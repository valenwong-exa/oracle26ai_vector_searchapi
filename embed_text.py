import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_PATH = BASE_DIR / "model_config.json"
DEFAULT_EMBEDDING_REPO_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_RERANKER_REPO_ID = "Qwen/Qwen3-VL-Reranker-2B"
PDF_PATH = BASE_DIR / "Application Containers In Physical Standby Data Guard Environment (Doc ID 2545894.1).pdf"
DEFAULT_PROXY = "http://127.0.0.1:7897"
DEFAULT_INSTRUCTION = "Represent the user's input for semantic retrieval."
DEFAULT_RERANK_INSTRUCTION = "Given a query, judge whether the document is relevant to the query."
DEFAULT_PDF_QUERY = (
    "What special requirements should be considered when using application containers "
    "in a physical standby Data Guard environment?"
)
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_RERANK_CHUNKS = 4
DEFAULT_RERANK_BATCH_SIZE_CUDA = 4
DEFAULT_RERANK_BATCH_SIZE_CPU = 1


def read_model_config(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(path_value: str | None, fallback: Path) -> Path:
    if not path_value:
        return fallback
    configured_path = Path(path_value)
    if not configured_path.is_absolute():
        configured_path = BASE_DIR / configured_path
    return configured_path


def get_env_or_config(config_key: str, env_key: str, default: str | None = None) -> str | None:
    env_value = os.environ.get(env_key)
    if env_value is not None and env_value != "":
        return env_value
    config_value = MODEL_CONFIG.get(config_key)
    if config_value is not None and config_value != "":
        return str(config_value)
    return default


MODEL_CONFIG = read_model_config()
MODEL_DIR = resolve_config_path(
    get_env_or_config("embedding_model_dir", "EMBEDDING_MODEL_DIR"),
    BASE_DIR / "Qwen3-VL-Embedding-2B",
)
RERANKER_MODEL_DIR = resolve_config_path(
    get_env_or_config("reranker_model_dir", "RERANKER_MODEL_DIR"),
    BASE_DIR / "Qwen3-VL-Reranker-2B",
)
DEFAULT_PROXY = str(get_env_or_config("proxy", "MODEL_PROXY", DEFAULT_PROXY) or DEFAULT_PROXY)
EMBEDDING_BACKEND = str(
    get_env_or_config("embedding_backend", "EMBEDDING_BACKEND", "auto") or "auto"
)
RERANKER_BACKEND = str(
    get_env_or_config("reranker_backend", "RERANKER_BACKEND", "auto") or "auto"
)
EMBEDDING_REPO_ID = str(
    get_env_or_config(
        "embedding_repo_id",
        "EMBEDDING_REPO_ID",
        DEFAULT_EMBEDDING_REPO_ID,
    )
    or DEFAULT_EMBEDDING_REPO_ID
)
RERANKER_REPO_ID = str(
    get_env_or_config(
        "reranker_repo_id",
        "RERANKER_REPO_ID",
        DEFAULT_RERANKER_REPO_ID,
    )
    or DEFAULT_RERANKER_REPO_ID
)


def has_model_weights(model_dir: Path) -> bool:
    weight_patterns = (
        "model.safetensors",
        "*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any(any(model_dir.glob(pattern)) for pattern in weight_patterns)


def resolve_local_model_dir(model_dir: Path) -> Path:
    if has_model_weights(model_dir):
        return model_dir

    nested_candidates = [
        path for path in model_dir.iterdir()
        if path.is_dir() and has_model_weights(path)
    ] if model_dir.exists() else []

    if len(nested_candidates) == 1:
        return nested_candidates[0]

    return model_dir


def list_found_weight_files(model_dir: Path) -> list[str]:
    model_dir = resolve_local_model_dir(model_dir)
    matched: list[str] = []
    for pattern in (
        "model.safetensors",
        "*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        matched.extend(str(path.name) for path in model_dir.glob(pattern))
    return sorted(set(matched))


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    import fitz

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as document:
        pages = [page.get_text("text") for page in document]

    text = normalize_text("\n\n".join(pages))
    if not text:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")

    return text, len(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start += step
    return chunks


def preview_text(text: str, max_chars: int = 180) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= max_chars:
        return single_line
    return single_line[:max_chars].rstrip() + "..."


def resolve_device(device_arg: str) -> str:
    normalized = device_arg.lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot use CUDA in the current environment.\n"
            "This usually means the current .venv has a CPU-only or broken PyTorch install.\n"
            "If `nvidia-smi` can see your NVIDIA GPU, reinstall CUDA-enabled PyTorch in this .venv,\n"
            "or run with `--device cpu` temporarily."
        )
    if normalized not in {"cuda", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, cpu")
    return normalized


def get_model_dtype(device: str):
    if device != "cuda":
        return "auto"
    if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def ensure_proxy(proxy_url: str | None) -> None:
    if not proxy_url:
        return
    os.environ.setdefault("HTTP_PROXY", proxy_url)
    os.environ.setdefault("HTTPS_PROXY", proxy_url)
    os.environ.setdefault("http_proxy", proxy_url)
    os.environ.setdefault("https_proxy", proxy_url)


def ensure_model(
    model_dir: Path,
    proxy_url: str | None,
    allow_download: bool,
    repo_id: str | None = None,
) -> Path:
    resolved_dir = resolve_local_model_dir(model_dir)
    if resolved_dir.exists() and has_model_weights(resolved_dir):
        return resolved_dir

    if not allow_download:
        found = list_found_weight_files(model_dir) if model_dir.exists() else []
        found_text = ", ".join(found) if found else "none"
        raise FileNotFoundError(
            f"Local model is incomplete in: {resolved_dir}\n"
            f"Found weight files: {found_text}\n"
            "Run again with --download to fetch missing files."
        )

    ensure_proxy(proxy_url)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    effective_repo_id = repo_id or DEFAULT_EMBEDDING_REPO_ID
    snapshot_download(repo_id=effective_repo_id, local_dir=str(model_dir))
    return model_dir


def build_messages(text: str, instruction: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    ]


def build_reranker_messages(query: str, document: str, instruction: str) -> list[dict]:
    rerank_system = (
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".'
    )
    rerank_user = (
        f"<Instruct>: {instruction}\n\n"
        f"<Query>: {query}\n\n"
        f"<Document>: {document}"
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": rerank_system}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": rerank_user}],
        },
    ]


def read_hf_config(model_dir: Path):
    return AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)


def resolve_model_max_length(tokenizer, fallback: int = 8192) -> int:
    max_length = getattr(tokenizer, "model_max_length", fallback)
    if not isinstance(max_length, int) or max_length <= 0 or max_length > 100000:
        return fallback
    return max_length


def read_pooling_config(model_dir: Path) -> dict:
    pooling_config_path = model_dir / "1_Pooling" / "config.json"
    if not pooling_config_path.exists():
        return {}
    with pooling_config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_embedding_backend(
    model_dir: Path,
    configured_backend: str = "auto",
) -> str:
    if configured_backend != "auto":
        return configured_backend
    config = read_hf_config(model_dir)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = [
        str(architecture).lower()
        for architecture in (getattr(config, "architectures", None) or [])
    ]
    if model_type.startswith("qwen3_vl"):
        return "qwen_vl"
    if model_type == "xlm-roberta" and any(
        architecture.endswith("model") for architecture in architectures
    ):
        return "bge_m3"
    if (model_dir / "1_Pooling" / "config.json").exists():
        return "sentence_transformer"
    raise ValueError(
        f"Unsupported embedding model backend for {model_dir}. "
        f"model_type={model_type!r}, architectures={architectures!r}"
    )


def infer_reranker_backend(
    model_dir: Path,
    configured_backend: str = "auto",
) -> str:
    if configured_backend != "auto":
        return configured_backend
    config = read_hf_config(model_dir)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = [
        str(architecture).lower()
        for architecture in (getattr(config, "architectures", None) or [])
    ]
    if model_type.startswith("qwen3_vl"):
        return "qwen_vl_yesno"
    if any(
        "sequenceclassification" in architecture for architecture in architectures
    ):
        return "bge_seq_cls"
    raise ValueError(
        f"Unsupported reranker model backend for {model_dir}. "
        f"model_type={model_type!r}, architectures={architectures!r}"
    )


def get_embedding_dimension(
    model_dir: Path,
    configured_backend: str = "auto",
) -> int:
    backend = infer_embedding_backend(model_dir, configured_backend)
    config = read_hf_config(model_dir)
    if backend == "qwen_vl":
        dimension = (
            getattr(config, "hidden_size", None)
            or getattr(getattr(config, "text_config", None), "hidden_size", None)
            or getattr(getattr(config, "vision_config", None), "out_hidden_size", None)
            or getattr(getattr(config, "vision_config", None), "hidden_size", None)
        )
        if not dimension:
            raise ValueError(
                f"Cannot determine embedding dimension for Qwen model: {model_dir}"
            )
        return int(dimension)
    if backend in ("bge_m3", "sentence_transformer"):
        pooling_config = read_pooling_config(model_dir)
        return int(
            pooling_config.get("word_embedding_dimension")
            or getattr(config, "hidden_size", 0)
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")


def move_inputs_to_device(model_inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def pool_sentence_transformer_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_config: dict,
) -> torch.Tensor:
    if pooling_config.get("pooling_mode_mean_tokens"):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float()
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    else:
        pooled = last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(pooled, dim=-1)


@dataclass
class EmbeddingRuntime:
    backend: str
    tokenizer: Any
    model: torch.nn.Module
    device: str
    instruction: str
    max_length: int
    model_dir: Path
    processor: Any | None = None

    @property
    def embedding_dimension(self) -> int:
        return get_embedding_dimension(self.model_dir, self.backend)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "qwen_vl":
            messages = [
                build_messages(normalize_text(text), self.instruction)
                for text in texts
            ]
            rendered_texts = [
                self.processor.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for message in messages
            ]
            model_inputs = self.processor(
                text=rendered_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            model_inputs = move_inputs_to_device(model_inputs, self.device)
            with torch.inference_mode():
                outputs = self.model.model(**model_inputs, return_dict=True)
                embeddings = torch.nn.functional.normalize(
                    outputs.last_hidden_state[:, -1, :], dim=-1
                )
            return embeddings.detach().cpu().tolist()

        if self.backend in ("bge_m3", "sentence_transformer"):
            normalized_texts = [normalize_text(text) for text in texts]
            model_inputs = self.tokenizer(
                normalized_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            model_inputs = move_inputs_to_device(model_inputs, self.device)
            with torch.inference_mode():
                outputs = self.model(**model_inputs, return_dict=True)
                embeddings = pool_sentence_transformer_embeddings(
                    outputs.last_hidden_state,
                    model_inputs["attention_mask"],
                    read_pooling_config(self.model_dir),
                )
            return embeddings.detach().cpu().tolist()

        raise ValueError(f"Unsupported embedding backend: {self.backend}")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass
class RerankerRuntime:
    backend: str
    tokenizer: Any
    model: torch.nn.Module
    device: str
    max_length: int
    processor: Any | None = None

    def score(
        self,
        query: str,
        documents: list[str],
        instruction: str,
        batch_size: int,
    ) -> list[float]:
        if self.backend == "qwen_vl_yesno":
            yes_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
            no_token_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
            scores: list[float] = []
            for start in range(0, len(documents), batch_size):
                batch_documents = documents[start : start + batch_size]
                rendered_texts = []
                for document in batch_documents:
                    messages = build_reranker_messages(query, document, instruction)
                    rendered_texts.append(
                        self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    )
                model_inputs = self.processor(
                    text=rendered_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                model_inputs = move_inputs_to_device(model_inputs, self.device)
                with torch.inference_mode():
                    outputs = self.model(**model_inputs, return_dict=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    yes_no_logits = next_token_logits[:, [yes_token_id, no_token_id]]
                    batch_scores = (
                        torch.softmax(yes_no_logits, dim=-1)[:, 0]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                scores.extend(batch_scores)
            return scores

        if self.backend == "bge_seq_cls":
            scores: list[float] = []
            normalized_query = normalize_text(query)
            for start in range(0, len(documents), batch_size):
                batch_documents = documents[start : start + batch_size]
                pairs = [
                    (normalized_query, normalize_text(document))
                    for document in batch_documents
                ]
                model_inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                model_inputs = move_inputs_to_device(model_inputs, self.device)
                with torch.inference_mode():
                    outputs = self.model(**model_inputs, return_dict=True)
                    logits = outputs.logits
                    if logits.ndim == 1:
                        batch_scores = torch.sigmoid(logits)
                    elif logits.shape[-1] == 1:
                        batch_scores = torch.sigmoid(logits[:, 0])
                    else:
                        batch_scores = torch.softmax(logits, dim=-1)[:, -1]
                scores.extend(batch_scores.detach().cpu().tolist())
            return scores

        raise ValueError(f"Unsupported reranker backend: {self.backend}")


def load_embedding_runtime(
    model_dir: Path,
    device: str,
    instruction: str,
    configured_backend: str = "auto",
) -> EmbeddingRuntime:
    backend = infer_embedding_backend(model_dir, configured_backend)
    if backend == "qwen_vl":
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            padding_side="right",
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            dtype=get_model_dtype(device),
            trust_remote_code=True,
        )
        model.eval()
        model.to(device)
        return EmbeddingRuntime(
            backend=backend,
            tokenizer=processor.tokenizer,
            processor=processor,
            model=model,
            device=device,
            instruction=instruction,
            max_length=8192,
            model_dir=model_dir,
        )

    if backend in ("bge_m3", "sentence_transformer"):
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        model.eval()
        model.to(device)
        return EmbeddingRuntime(
            backend=backend,
            tokenizer=tokenizer,
            model=model,
            device=device,
            instruction=instruction,
            max_length=resolve_model_max_length(tokenizer),
            model_dir=model_dir,
        )

    raise ValueError(f"Unsupported embedding backend: {backend}")


def load_reranker_runtime(
    model_dir: Path,
    device: str,
    configured_backend: str = "auto",
) -> RerankerRuntime:
    backend = infer_reranker_backend(model_dir, configured_backend)
    if backend == "qwen_vl_yesno":
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            padding_side="right",
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            dtype=get_model_dtype(device),
            trust_remote_code=True,
        )
        model.eval()
        model.to(device)
        return RerankerRuntime(
            backend=backend,
            tokenizer=processor.tokenizer,
            processor=processor,
            model=model,
            device=device,
            max_length=8192,
        )

    if backend == "bge_seq_cls":
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        model.eval()
        model.to(device)
        return RerankerRuntime(
            backend=backend,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=resolve_model_max_length(tokenizer),
        )

    raise ValueError(f"Unsupported reranker backend: {backend}")


def resolve_input_and_chunks(
    text_arg: str | None,
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
) -> tuple[str, list[str], str]:
    if text_arg is not None:
        text = normalize_text(text_arg)
        return text, chunk_text(text, chunk_size, overlap), "command line text"

    pdf_text, page_count = extract_pdf_text(pdf_path)
    chunks = chunk_text(pdf_text, chunk_size, overlap)
    if not chunks:
        raise ValueError(f"No chunks were generated from PDF: {pdf_path}")
    source = f"pdf: {pdf_path.name} ({page_count} pages, {len(chunks)} chunks)"
    return chunks[0], chunks, source


def print_embedding_example(
    model_dir: Path,
    input_text: str,
    instruction: str,
    device: str,
    backend: str,
) -> None:
    runtime = load_embedding_runtime(
        model_dir=model_dir,
        device=device,
        instruction=instruction,
        configured_backend=backend,
    )
    vector = runtime.embed_query(input_text)
    print(f"Embedding dimension: {len(vector)}")
    print("Embedding:")
    print(vector)

def print_reranker_example(
    reranker_model_dir: Path,
    query: str,
    documents: list[str],
    max_rerank_chunks: int,
    top_k: int,
    instruction: str,
    device: str,
    rerank_batch_size: int,
    backend: str,
) -> None:
    runtime = load_reranker_runtime(
        model_dir=reranker_model_dir,
        device=device,
        configured_backend=backend,
    )
    candidate_documents = documents[:max_rerank_chunks]
    scores = runtime.score(
        query=query,
        documents=candidate_documents,
        instruction=instruction,
        batch_size=max(rerank_batch_size, 1),
    )

    ranked = sorted(
        enumerate(scores, start=1),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    print("\nReranker example:")
    print(f"Query: {query}")
    print(f"Scored chunks: {len(candidate_documents)}")
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        print(f"Top {rank}: chunk {chunk_id}, score={score:.6f}")
        print(f"  {preview_text(candidate_documents[chunk_id - 1])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local embedding and reranker examples with Qwen3-VL or BGE models."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Optional text to embed. If omitted, extract text from the PDF.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction text used by the embedding model.",
    )
    parser.add_argument(
        "--proxy",
        default=DEFAULT_PROXY,
        help="HTTP/HTTPS proxy URL used for model download.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help="Local model directory.",
    )
    parser.add_argument(
        "--embedding-backend",
        default=EMBEDDING_BACKEND,
        help="Embedding backend: auto, qwen_vl, bge_m3, or sentence_transformer.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing model files if the local directory is incomplete.",
    )
    parser.add_argument(
        "--reranker-model-dir",
        default=str(RERANKER_MODEL_DIR),
        help="Local reranker model directory.",
    )
    parser.add_argument(
        "--reranker-backend",
        default=RERANKER_BACKEND,
        help="Reranker backend: auto, qwen_vl_yesno, or bge_seq_cls.",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip the reranker scoring example.",
    )
    parser.add_argument(
        "--reranker-instruction",
        default=DEFAULT_RERANK_INSTRUCTION,
        help="Instruction text used by the reranker model.",
    )
    parser.add_argument(
        "--pdf-path",
        default=str(PDF_PATH),
        help="Local PDF path used for text extraction.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_PDF_QUERY,
        help="Query used by the reranker example.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many reranked chunks to print.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in characters for PDF text splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters for PDF text splitting.",
    )
    parser.add_argument(
        "--max-rerank-chunks",
        type=int,
        default=DEFAULT_MAX_RERANK_CHUNKS,
        help="Maximum number of chunks to score in the reranker example.",
    )
    parser.add_argument(
        "--preview-pdf-only",
        action="store_true",
        help="Only extract and preview PDF text without running embedding or reranker.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=0,
        help="Batch size for reranker scoring. 0 means auto based on device.",
    )
    args = parser.parse_args()
    input_text, chunks, input_source = resolve_input_and_chunks(
        text_arg=args.text,
        pdf_path=Path(args.pdf_path),
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    if args.preview_pdf_only:
        print(f"Input source: {input_source}")
        print(f"Chunk count: {len(chunks)}")
        print("First chunk preview:")
        print(preview_text(chunks[0], max_chars=500))
        return

    model_dir = ensure_model(
        Path(args.model_dir),
        args.proxy,
        args.download,
        EMBEDDING_REPO_ID,
    )
    device = resolve_device(args.device)
    rerank_batch_size = args.rerank_batch_size
    if rerank_batch_size <= 0:
        rerank_batch_size = (
            DEFAULT_RERANK_BATCH_SIZE_CUDA
            if device == "cuda"
            else DEFAULT_RERANK_BATCH_SIZE_CPU
        )
    print(f"Loading embedding model from: {model_dir}")
    print(
        f"Embedding backend: "
        f"{infer_embedding_backend(model_dir, args.embedding_backend)}"
    )
    print(f"Using device: {device}")
    print(f"Rerank batch size: {rerank_batch_size}")
    print(f"Input source: {input_source}")
    print(f"Embedding text length: {len(input_text)} characters")
    print_embedding_example(
        model_dir=model_dir,
        input_text=input_text,
        instruction=args.instruction,
        device=device,
        backend=args.embedding_backend,
    )

    if not args.skip_reranker:
        reranker_model_dir = ensure_model(
            Path(args.reranker_model_dir),
            args.proxy,
            False,
            RERANKER_REPO_ID,
        )
        print(f"\nLoading reranker model from: {reranker_model_dir}")
        print(
            f"Reranker backend: "
            f"{infer_reranker_backend(reranker_model_dir, args.reranker_backend)}"
        )
        print_reranker_example(
            reranker_model_dir=reranker_model_dir,
            query=args.query,
            documents=chunks,
            max_rerank_chunks=max(args.max_rerank_chunks, 1),
            top_k=max(args.top_k, 1),
            instruction=args.reranker_instruction,
            device=device,
            rerank_batch_size=rerank_batch_size,
            backend=args.reranker_backend,
        )


if __name__ == "__main__":
    main()
