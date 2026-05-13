import array
import json
import logging
import tempfile
import time
import traceback
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

import oracledb
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from embed_text import (
    DEFAULT_INSTRUCTION,
    DEFAULT_PROXY,
    DEFAULT_RERANK_INSTRUCTION,
    EMBEDDING_BACKEND,
    EMBEDDING_REPO_ID,
    MODEL_DIR,
    RERANKER_BACKEND,
    RERANKER_REPO_ID,
    RERANKER_MODEL_DIR,
    ensure_model,
    ensure_proxy,
    load_reranker_runtime,
    normalize_text,
    resolve_device,
)
from oracle_langchain_demo import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_DOC_TYPE,
    DEFAULT_SOURCE_FILE,
    DEFAULT_TABLE_CHUNKS,
    DEFAULT_TABLE_DOCS,
    ChunkRow,
    LocalTextEmbeddings,
    build_retrieval_text,
    count_tokens,
    create_schema_if_needed,
    append_deepseek_summary_to_document,
    insert_chunks,
    insert_document_if_new,
    preview_text,
    read_text_file,
)


BASE_DIR = Path(__file__).resolve().parent
ORACLE_CONFIG_PATH = BASE_DIR / "oracle_config.json"
SCHEMA_SQL_PATH = BASE_DIR / "oracle_vector_schema.sql"
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 19000
DEFAULT_FETCH_MULTIPLIER = 10


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("oracle_vector_api")


class UploadResponse(BaseModel):
    document_name: str
    document_type: str
    doc_id: int
    inserted: bool
    deepseek_summary_applied: bool
    parsed_chunk_count: int
    inserted_chunk_count: int
    full_text_length: int
    full_text_preview: str


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    reranker_score: float = Field(0.0, ge=0.0, le=1.0)
    return_mode: Literal["full", "split", "title"] = "split"
    fetch_k: int | None = Field(None, ge=1, le=1000)
    document_type: str | None = None
    device: Literal["auto", "cuda", "cpu"] = "cuda"
    reranker_instruction: str = DEFAULT_RERANK_INSTRUCTION
    instruction: str = DEFAULT_INSTRUCTION


class SearchResultItem(BaseModel):
    txt: str
    score: float


app = FastAPI(
    title="Oracle Vector Search API",
    version="1.0.0",
    description="Upload documents, split/embed/store into Oracle, and run vector search with rerank.",
)


def load_oracle_config(config_path: Path = ORACLE_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Oracle config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    dsn = config.get("dsn")
    if not dsn:
        host = config.get("host")
        port = config.get("port", 1521)
        service_name = config.get("service_name")
        if not host or not service_name:
            raise ValueError("oracle_config.json must contain either dsn or host/port/service_name.")
        dsn = f"{host}:{port}/{service_name}"
        config["dsn"] = dsn
    return config


def mask_oracle_config(config: dict) -> dict:
    return {
        "user": config.get("user"),
        "dsn": config.get("dsn"),
        "config_dir": config.get("config_dir"),
        "wallet_location": config.get("wallet_location"),
    }


def make_request_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def text_preview(text: str, max_chars: int = 120) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}..."


def connect_oracle_from_json() -> oracledb.Connection:
    config = load_oracle_config()
    logger.info("Opening Oracle connection: %s", mask_oracle_config(config))
    connect_kwargs = {
        "user": config["user"],
        "password": config["password"],
        "dsn": config["dsn"],
    }
    if config.get("config_dir"):
        connect_kwargs["config_dir"] = config["config_dir"]
    if config.get("wallet_location"):
        connect_kwargs["wallet_location"] = config["wallet_location"]
    if config.get("wallet_password"):
        connect_kwargs["wallet_password"] = config["wallet_password"]
    return oracledb.connect(**connect_kwargs)


@lru_cache(maxsize=4)
def get_embedder(device: str, instruction: str) -> LocalTextEmbeddings:
    model_dir = ensure_model(MODEL_DIR, DEFAULT_PROXY, False, EMBEDDING_REPO_ID)
    return LocalTextEmbeddings(model_dir=model_dir, instruction=instruction, device=device)


@lru_cache(maxsize=4)
def get_reranker(device: str):
    model_dir = ensure_model(RERANKER_MODEL_DIR, DEFAULT_PROXY, False, RERANKER_REPO_ID)
    return load_reranker_runtime(
        model_dir=model_dir,
        device=device,
        configured_backend=RERANKER_BACKEND,
    )


def build_token_splitter_by_tokens(tokenizer_holder, chunk_size_tokens: int, chunk_overlap_tokens: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    overlap_tokens = max(0, min(chunk_overlap_tokens, max(chunk_size_tokens - 1, 0)))
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer_holder.tokenizer,
        chunk_size=max(chunk_size_tokens, 1),
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )


def load_document_text(input_path: Path) -> tuple[str, str]:
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        from embed_text import extract_pdf_text

        full_text, _ = extract_pdf_text(input_path)
        return normalize_text(full_text), "pdf"
    if suffix == ".txt":
        return read_text_file(input_path), "txt"
    raise ValueError(f"Unsupported file type: {input_path.suffix}. Only .pdf and .txt are supported.")


def save_upload_to_temp(upload: UploadFile) -> tuple[Path, int]:
    suffix = Path(upload.filename or "uploaded.bin").suffix
    temp_dir = BASE_DIR / ".upload_cache"
    temp_dir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as tmp:
        content = upload.file.read()
        tmp.write(content)
        return Path(tmp.name), len(content)


def build_chunk_rows(
    document_name: str,
    full_text: str,
    document_type: str,
    source_file: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    instruction: str,
    device: str,
) -> tuple[list[ChunkRow], LocalTextEmbeddings]:
    embedder = get_embedder(device, instruction)
    splitter = build_token_splitter_by_tokens(
        tokenizer_holder=embedder,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    split_documents = splitter.create_documents([full_text])
    texts = [normalize_text(doc.page_content) for doc in split_documents if normalize_text(doc.page_content)]
    if not texts:
        raise ValueError("No chunk text generated after splitting.")
    retrieval_texts = [build_retrieval_text(document_name, text) for text in texts]
    embeddings = embedder.embed_documents(retrieval_texts)
    rows: list[ChunkRow] = []
    for index, (text, embedding) in enumerate(zip(texts, embeddings), start=1):
        rows.append(
            ChunkRow(
                chunk_id=index,
                chunk_tokens=count_tokens(embedder, text),
                chunk_text=text,
                embedding=embedding,
                document_type=document_type,
                source_file=source_file if index == 1 else None,
            )
        )
    return rows, embedder


def search_chunk_candidates(
    connection: oracledb.Connection,
    docs_table: str,
    chunks_table: str,
    query_embedding: list[float],
    fetch_k: int,
    document_type: str | None,
) -> list[dict]:
    cursor = connection.cursor()
    sql = f"""
        SELECT
            d.doc_id,
            d.document_name,
            d.document_text,
            c.chunk_id,
            c.chunk_tokens,
            c.chunk_text,
            c.document_type,
            c.source_file,
            VECTOR_DISTANCE(c.embedding, :query_vector, COSINE) AS distance
        FROM {chunks_table} c
        JOIN {docs_table} d
          ON d.doc_id = c.doc_id
        WHERE (:document_type IS NULL OR c.document_type = :document_type)
        ORDER BY VECTOR_DISTANCE(c.embedding, :query_vector, COSINE)
        FETCH FIRST {fetch_k} ROWS ONLY
    """
    cursor.execute(
        sql,
        query_vector=array.array("f", query_embedding),
        document_type=document_type,
    )
    rows = []
    for result in cursor:
        rows.append(
            {
                "doc_id": int(result[0]),
                "document_name": result[1],
                "document_text": result[2].read() if hasattr(result[2], "read") else result[2],
                "chunk_id": int(result[3]),
                "chunk_tokens": int(result[4]),
                "chunk_text": result[5].read() if hasattr(result[5], "read") else result[5],
                "document_type": result[6],
                "source_file": result[7],
                "distance": float(result[8]),
            }
        )
    return rows


def format_search_results(
    candidates: list[dict],
    rerank_score_list: list[float],
    top_k: int,
    reranker_score: float,
    return_mode: str,
) -> list[SearchResultItem]:
    grouped: dict[int, dict] = {}
    for row, score in zip(candidates, rerank_score_list):
        if score < reranker_score:
            continue
        doc_group = grouped.setdefault(
            row["doc_id"],
            {
                "document_name": row["document_name"],
                "document_text": row["document_text"] or "",
                "items": [],
                "max_score": 0.0,
            },
        )
        doc_group["items"].append({"chunk_id": row["chunk_id"], "chunk_text": row["chunk_text"], "score": score})
        doc_group["max_score"] = max(doc_group["max_score"], score)

    ranked_docs = sorted(grouped.values(), key=lambda item: item["max_score"], reverse=True)[:top_k]
    results: list[SearchResultItem] = []
    for doc in ranked_docs:
        sorted_chunks = sorted(doc["items"], key=lambda item: item["chunk_id"])
        if return_mode == "title":
            text_payload = doc["document_name"]
        elif return_mode == "full":
            content = doc["document_text"] or "\n\n".join(chunk["chunk_text"] for chunk in sorted_chunks)
            text_payload = f"{doc['document_name']}\n{content}"
        else:
            combined_chunks = "\n\n".join(chunk["chunk_text"] for chunk in sorted_chunks)
            text_payload = f"{doc['document_name']}\n{combined_chunks}"
        results.append(SearchResultItem(txt=text_payload, score=round(doc["max_score"], 6)))
    return results


@app.get("/health")
def health_check() -> dict:
    logger.info("Health check ok")
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    chunk_size_tokens: int = Form(DEFAULT_CHUNK_TOKENS),
    chunk_overlap_tokens: int = Form(int(DEFAULT_CHUNK_TOKENS * 0.1)),
    source_file: str = Form(DEFAULT_SOURCE_FILE),
    document_type: str | None = Form(None),
    instruction: str = Form(DEFAULT_INSTRUCTION),
    use_deepseek_summary: bool = Form(True),
    device: Literal["auto", "cuda", "cpu"] = Form("cuda"),
    docs_table: str = Form(DEFAULT_TABLE_DOCS),
    chunks_table: str = Form(DEFAULT_TABLE_CHUNKS),
) -> UploadResponse:
    request_id = make_request_id("upload")
    started_at = time.perf_counter()
    if not file.filename:
        raise HTTPException(status_code=400, detail="file name is required")
    logger.info(
        "[%s] Upload request received: filename=%s chunk_size_tokens=%s chunk_overlap_tokens=%s source_file=%s document_type=%s use_deepseek_summary=%s device=%s docs_table=%s chunks_table=%s",
        request_id,
        file.filename,
        chunk_size_tokens,
        chunk_overlap_tokens,
        source_file,
        document_type,
        use_deepseek_summary,
        device,
        docs_table,
        chunks_table,
    )
    ensure_proxy(DEFAULT_PROXY)
    resolved_device = resolve_device(device)
    logger.info("[%s] Resolved device: %s", request_id, resolved_device)
    temp_path, upload_size = save_upload_to_temp(file)
    logger.info("[%s] Upload cached: temp_path=%s bytes=%s", request_id, temp_path, upload_size)
    try:
        parse_started_at = time.perf_counter()
        full_text, detected_type = load_document_text(temp_path)
        logger.info(
            "[%s] Document parsed: detected_type=%s full_text_length=%s preview=%s elapsed=%.3fs",
            request_id,
            detected_type,
            len(full_text),
            text_preview(full_text),
            time.perf_counter() - parse_started_at,
        )
        effective_document_type = document_type or detected_type or DEFAULT_DOC_TYPE
        deepseek_summary_applied = False
        if use_deepseek_summary:
            deepseek_started_at = time.perf_counter()
            full_text = append_deepseek_summary_to_document(
                document_name=file.filename,
                document_type=effective_document_type,
                full_text=full_text,
                logger=logger,
            )
            deepseek_summary_applied = True
            logger.info(
                "[%s] DeepSeek summary appended: full_text_length=%s preview=%s elapsed=%.3fs",
                request_id,
                len(full_text),
                text_preview(full_text, max_chars=180),
                time.perf_counter() - deepseek_started_at,
            )
        chunk_started_at = time.perf_counter()
        chunk_rows, embedder = build_chunk_rows(
            document_name=file.filename,
            full_text=full_text,
            document_type=effective_document_type,
            source_file=source_file,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            instruction=instruction,
            device=resolved_device,
        )
        chunk_token_total = sum(row.chunk_tokens for row in chunk_rows)
        logger.info(
            "[%s] Chunks built: count=%s total_tokens=%s first_chunk_tokens=%s elapsed=%.3fs",
            request_id,
            len(chunk_rows),
            chunk_token_total,
            chunk_rows[0].chunk_tokens if chunk_rows else 0,
            time.perf_counter() - chunk_started_at,
        )
        db_started_at = time.perf_counter()
        connection = connect_oracle_from_json()
        try:
            create_schema_if_needed(
                connection,
                SCHEMA_SQL_PATH,
                docs_table,
                chunks_table,
                embedder.embedding_dimension,
            )
            logger.info(
                "[%s] Schema ensured: docs_table=%s chunks_table=%s embedding_dimension=%s",
                request_id,
                docs_table,
                chunks_table,
                embedder.embedding_dimension,
            )
            doc_id, inserted = insert_document_if_new(
                connection=connection,
                document_name=file.filename,
                document_text=full_text,
                document_type=effective_document_type,
                source_file=source_file,
                docs_table=docs_table,
            )
            inserted_chunk_count = 0
            if inserted:
                insert_chunks(connection, doc_id, chunk_rows, chunks_table)
                inserted_chunk_count = len(chunk_rows)
                logger.info(
                    "[%s] Document inserted: doc_id=%s inserted_chunks=%s db_elapsed=%.3fs",
                    request_id,
                    doc_id,
                    inserted_chunk_count,
                    time.perf_counter() - db_started_at,
                )
            else:
                logger.info(
                    "[%s] Document already exists, skip chunk insert: doc_id=%s db_elapsed=%.3fs",
                    request_id,
                    doc_id,
                    time.perf_counter() - db_started_at,
                )
        finally:
            connection.close()
        response = UploadResponse(
            document_name=file.filename,
            document_type=effective_document_type,
            doc_id=doc_id,
            inserted=inserted,
            deepseek_summary_applied=deepseek_summary_applied,
            parsed_chunk_count=len(chunk_rows),
            inserted_chunk_count=inserted_chunk_count,
            full_text_length=len(full_text),
            full_text_preview=preview_text(full_text, max_chars=300),
        )
        logger.info(
            "[%s] Upload request completed: inserted=%s parsed_chunk_count=%s inserted_chunk_count=%s total_elapsed=%.3fs",
            request_id,
            response.inserted,
            response.parsed_chunk_count,
            response.inserted_chunk_count,
            time.perf_counter() - started_at,
        )
        return response
    except Exception as exc:
        logger.error("[%s] Upload request failed: %s", request_id, exc)
        logger.error("[%s] Upload traceback:\n%s", request_id, traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
        logger.info("[%s] Temp file removed: %s", request_id, temp_path)


@app.post("/search", response_model=list[SearchResultItem])
def vector_search(request: SearchRequest) -> list[SearchResultItem]:
    request_id = make_request_id("search")
    started_at = time.perf_counter()
    logger.info(
        "[%s] Search request received: prompt=%s top_k=%s fetch_k=%s reranker_score=%s return_mode=%s document_type=%s device=%s",
        request_id,
        text_preview(request.prompt),
        request.top_k,
        request.fetch_k,
        request.reranker_score,
        request.return_mode,
        request.document_type,
        request.device,
    )
    ensure_proxy(DEFAULT_PROXY)
    resolved_device = resolve_device(request.device)
    fetch_k = request.fetch_k or max(request.top_k * DEFAULT_FETCH_MULTIPLIER, request.top_k)
    logger.info("[%s] Search resolved device=%s effective_fetch_k=%s", request_id, resolved_device, fetch_k)
    try:
        embed_started_at = time.perf_counter()
        embedder = get_embedder(resolved_device, request.instruction)
        logger.info("[%s] Search embedder ready: elapsed=%.3fs", request_id, time.perf_counter() - embed_started_at)
        query_embed_started_at = time.perf_counter()
        query_embedding = embedder.embed_query(request.prompt)
        logger.info(
            "[%s] Query embedding generated: dimension=%s elapsed=%.3fs",
            request_id,
            len(query_embedding),
            time.perf_counter() - query_embed_started_at,
        )
        search_started_at = time.perf_counter()
        connection = connect_oracle_from_json()
        try:
            create_schema_if_needed(
                connection,
                SCHEMA_SQL_PATH,
                DEFAULT_TABLE_DOCS,
                DEFAULT_TABLE_CHUNKS,
                embedder.embedding_dimension,
            )
            candidates = search_chunk_candidates(
                connection=connection,
                docs_table=DEFAULT_TABLE_DOCS,
                chunks_table=DEFAULT_TABLE_CHUNKS,
                query_embedding=query_embedding,
                fetch_k=fetch_k,
                document_type=request.document_type,
            )
        finally:
            connection.close()
        logger.info(
            "[%s] Oracle vector search finished: candidate_count=%s best_distance=%s elapsed=%.3fs",
            request_id,
            len(candidates),
            round(candidates[0]["distance"], 6) if candidates else None,
            time.perf_counter() - search_started_at,
        )
        if not candidates:
            logger.info("[%s] Search request completed with no candidates: total_elapsed=%.3fs", request_id, time.perf_counter() - started_at)
            return []
        rerank_started_at = time.perf_counter()
        reranker = get_reranker(resolved_device)
        rerank_score_list = reranker.score(
            query=request.prompt,
            documents=[
                build_retrieval_text(row["document_name"], row["chunk_text"])
                for row in candidates
            ],
            instruction=request.reranker_instruction,
            batch_size=1 if resolved_device == "cpu" else 4,
        )
        passed_count = sum(1 for score in rerank_score_list if score >= request.reranker_score)
        logger.info(
            "[%s] Rerank finished: rerank_count=%s passed_threshold=%s max_score=%.6f min_score=%.6f elapsed=%.3fs",
            request_id,
            len(rerank_score_list),
            passed_count,
            max(rerank_score_list) if rerank_score_list else 0.0,
            min(rerank_score_list) if rerank_score_list else 0.0,
            time.perf_counter() - rerank_started_at,
        )
        results = format_search_results(
            candidates=candidates,
            rerank_score_list=rerank_score_list,
            top_k=request.top_k,
            reranker_score=request.reranker_score,
            return_mode=request.return_mode,
        )
        logger.info(
            "[%s] Search request completed: result_count=%s top_score=%s total_elapsed=%.3fs",
            request_id,
            len(results),
            results[0].score if results else None,
            time.perf_counter() - started_at,
        )
        return results
    except Exception as exc:
        logger.error("[%s] Search request failed: %s", request_id, exc)
        logger.error("[%s] Search traceback:\n%s", request_id, traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("oracle_vector_api:app", host=DEFAULT_API_HOST, port=DEFAULT_API_PORT, reload=False)
