import argparse
import array
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import oracledb
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embed_text import (
    DEFAULT_INSTRUCTION,
    DEFAULT_PDF_QUERY,
    DEFAULT_PROXY,
    DEFAULT_RERANK_INSTRUCTION,
    EMBEDDING_BACKEND,
    EMBEDDING_REPO_ID,
    MODEL_DIR,
    PDF_PATH,
    RERANKER_BACKEND,
    RERANKER_REPO_ID,
    RERANKER_MODEL_DIR,
    ensure_model,
    ensure_proxy,
    extract_pdf_text,
    load_embedding_runtime,
    load_reranker_runtime,
    normalize_text,
    preview_text,
    resolve_device,
)


DEFAULT_DOC_TYPE = "pdf"
DEFAULT_SOURCE_FILE = "blog"
DEFAULT_TABLE_DOCS = os.environ.get("VECTOR_DOCS_TABLE", "LC_DEMO_DOCUMENTS")
DEFAULT_TABLE_CHUNKS = os.environ.get("VECTOR_CHUNKS_TABLE", "LC_DEMO_CHUNKS")
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_RATIO = 0.1
DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 10
DEFAULT_ORACLE_USER = "valen"
DEFAULT_ORACLE_PASSWORD = "oracle"
DEFAULT_ORACLE_DSN = "192.168.56.101:1521/aidemo_pdb"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_SUMMARY_HEADER = "# Deepseek Summary"
DEFAULT_DEEPSEEK_SYSTEM_PROMPT = (
    "You are a senior technical documentation analyst. "
    "Extract only facts supported by the document. "
    "Do not invent information. If something is unclear, keep the wording cautious. "
    "Write in the same primary language as the document when possible, "
    "but preserve product names, acronyms, commands, and error codes exactly."
)


@dataclass
class ChunkRow:
    chunk_id: int
    chunk_tokens: int
    chunk_text: str
    embedding: list[float]
    document_type: str
    source_file: str | None


TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


class LocalTextEmbeddings(Embeddings):
    def __init__(self, model_dir: Path, instruction: str, device: str):
        self.model_dir = model_dir
        self.instruction = instruction
        self.device = device
        self.runtime = load_embedding_runtime(
            model_dir=model_dir,
            device=device,
            instruction=instruction,
            configured_backend=EMBEDDING_BACKEND,
        )
        self.tokenizer = self.runtime.tokenizer
        self.embedding_dimension = self.runtime.embedding_dimension

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.runtime.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def resolve_tokenizer(tokenizer_holder):
    return getattr(tokenizer_holder, "tokenizer", tokenizer_holder)


def count_tokens(tokenizer_holder, text: str) -> int:
    tokenizer = resolve_tokenizer(tokenizer_holder)
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_token_splitter(tokenizer_holder, chunk_tokens: int, overlap_ratio: float):
    chunk_overlap = max(1, int(chunk_tokens * overlap_ratio))
    tokenizer = resolve_tokenizer(tokenizer_holder)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_tokens,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )


def read_text_file(text_path: Path) -> str:
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    return normalize_text(text_path.read_text(encoding="utf-8"))


def build_deepseek_summary_user_prompt(
    document_name: str,
    document_type: str,
    full_text: str,
) -> str:
    return (
        "Please analyze the following technical document and generate a concise summary section.\n\n"
        "Requirements:\n"
        "1. Output markdown only.\n"
        f"2. The first line must be exactly: {DEEPSEEK_SUMMARY_HEADER}\n"
        "3. Then output a section named `## Document Overview` with 3-6 bullet points.\n"
        "4. Then output a section named `## Questions This Document Can Answer`.\n"
        "5. Under that section, list 1-3 realistic questions that technical engineers are likely to ask and that this document can answer.\n"
        "6. Do not answer the questions, only list the questions.\n"
        "7. Keep the summary concise, factual, and grounded in the source text.\n"
        "8. Do not include any extra introduction, explanation, or code block fences.\n\n"
        f"Document name: {document_name}\n"
        f"Document type: {document_type}\n\n"
        "Document content:\n"
        f"{full_text}"
    )


def generate_deepseek_summary_section(
    document_name: str,
    document_type: str,
    full_text: str,
    logger: logging.Logger | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek summary requires the OpenAI SDK. Please install `openai` first."
        ) from exc

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DeepSeek summary requires the environment variable `DEEPSEEK_API_KEY`."
        )

    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    model_name = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    client = OpenAI(api_key=api_key, base_url=base_url)
    if logger:
        logger.info(
            "Calling DeepSeek summary model=%s base_url=%s document=%s text_length=%s",
            model_name,
            base_url,
            document_name,
            len(full_text),
        )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": DEFAULT_DEEPSEEK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_deepseek_summary_user_prompt(
                    document_name=document_name,
                    document_type=document_type,
                    full_text=full_text,
                ),
            },
        ],
        stream=False,
        reasoning_effort=os.environ.get("DEEPSEEK_REASONING_EFFORT", "high"),
        extra_body={"thinking": {"type": "enabled"}},
    )
    content = normalize_text(response.choices[0].message.content or "")
    if not content:
        raise RuntimeError("DeepSeek returned empty summary content.")
    if not content.startswith(DEEPSEEK_SUMMARY_HEADER):
        content = f"{DEEPSEEK_SUMMARY_HEADER}\n\n{content}"
    return content


def append_deepseek_summary_to_document(
    document_name: str,
    document_type: str,
    full_text: str,
    logger: logging.Logger | None = None,
) -> str:
    normalized_text = normalize_text(full_text)
    if not normalized_text:
        return normalized_text
    if re.search(r"(?im)^# Deepseek Summary\s*$", normalized_text):
        if logger:
            logger.info(
                "Document already contains a Deepseek Summary section, skip regeneration: %s",
                document_name,
            )
        return normalized_text
    summary_section = generate_deepseek_summary_section(
        document_name=document_name,
        document_type=document_type,
        full_text=normalized_text,
        logger=logger,
    )
    return normalize_text(f"{normalized_text}\n\n{summary_section}")


def load_source_documents(input_path: Path) -> tuple[list[Document], int | None, str]:
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        full_text, page_count = extract_pdf_text(input_path)
    elif suffix == ".txt":
        full_text = read_text_file(input_path)
        page_count = None
    else:
        raise ValueError(f"Unsupported input type: {input_path.suffix}. Only .pdf and .txt are supported.")

    document = Document(
        page_content=full_text,
        metadata={
            "document_name": input_path.name,
            "document_type": suffix.lstrip(".").lower() or DEFAULT_DOC_TYPE,
        },
    )
    return [document], page_count, full_text


def split_documents(
    documents: list[Document],
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    return splitter.split_documents(documents)


def build_retrieval_text(document_name: str, chunk_text: str) -> str:
    title = normalize_text(document_name)
    content = normalize_text(chunk_text)
    if not title:
        return content
    return f"Title: {title}\nContent:\n{content}"


def chunk_rows_from_documents(
    chunks: list[Document],
    embedder: LocalTextEmbeddings,
    tokenizer_holder,
    document_name: str,
    document_type: str,
    source_file: str,
) -> list[ChunkRow]:
    texts = [normalize_text(chunk.page_content) for chunk in chunks if normalize_text(chunk.page_content)]
    if not texts:
        raise ValueError("No chunk text generated after splitting.")
    retrieval_texts = [build_retrieval_text(document_name, text) for text in texts]
    embeddings = embedder.embed_documents(retrieval_texts)
    rows: list[ChunkRow] = []
    for index, (text, embedding) in enumerate(zip(texts, embeddings), start=1):
        rows.append(
            ChunkRow(
                chunk_id=index,
                chunk_tokens=count_tokens(tokenizer_holder, text),
                chunk_text=text,
                embedding=embedding,
                document_type=document_type,
                source_file=source_file if index == 1 else None,
            )
        )
    return rows


def connect_oracle_from_env() -> oracledb.Connection:
    user = os.environ.get("ORACLE_USER", DEFAULT_ORACLE_USER)
    password = os.environ.get("ORACLE_PASSWORD", DEFAULT_ORACLE_PASSWORD)
    dsn = os.environ.get("ORACLE_DSN", DEFAULT_ORACLE_DSN)
    config_dir = os.environ.get("ORACLE_CONFIG_DIR")
    wallet_location = os.environ.get("ORACLE_WALLET_LOCATION")
    wallet_password = os.environ.get("ORACLE_WALLET_PASSWORD")
    connect_kwargs = {
        "user": user,
        "password": password,
        "dsn": dsn,
    }
    if config_dir:
        connect_kwargs["config_dir"] = config_dir
    if wallet_location:
        connect_kwargs["wallet_location"] = wallet_location
    if wallet_password:
        connect_kwargs["wallet_password"] = wallet_password
    return oracledb.connect(**connect_kwargs)


def normalize_table_name(table_name: str) -> str:
    normalized = table_name.strip().upper()
    if not TABLE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid Oracle table name: {table_name}")
    return normalized


def table_exists(connection: oracledb.Connection, table_name: str) -> bool:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM user_tables WHERE table_name = :table_name",
        table_name=normalize_table_name(table_name),
    )
    return int(cursor.fetchone()[0]) > 0


def get_existing_vector_dimension(
    connection: oracledb.Connection,
    table_name: str,
    column_name: str = "EMBEDDING",
) -> int | None:
    if not table_exists(connection, table_name):
        return None
    cursor = connection.cursor()
    cursor.execute(
        "SELECT DBMS_METADATA.GET_DDL('TABLE', :table_name) FROM dual",
        table_name=normalize_table_name(table_name),
    )
    row = cursor.fetchone()
    if not row or row[0] is None:
        return None
    ddl = row[0].read() if hasattr(row[0], "read") else row[0]
    pattern = rf'"?{column_name.upper()}"?\s+VECTOR\((\d+)\s*,\s*FLOAT32\)'
    match = re.search(pattern, ddl, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def build_schema_statements(
    docs_table: str,
    chunks_table: str,
    embedding_dimension: int,
) -> list[str]:
    docs_table = normalize_table_name(docs_table)
    chunks_table = normalize_table_name(chunks_table)
    index_name = f"{chunks_table[:24]}_DT_IX"
    return [
        f"""
        CREATE TABLE {docs_table} (
            doc_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            document_name VARCHAR2(512) NOT NULL,
            document_text CLOB NOT NULL,
            document_type VARCHAR2(128) NOT NULL,
            source_file VARCHAR2(128),
            created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            UNIQUE (document_name)
        )
        """,
        f"""
        CREATE TABLE {chunks_table} (
            doc_id NUMBER NOT NULL,
            chunk_id NUMBER NOT NULL,
            chunk_tokens NUMBER NOT NULL,
            chunk_text CLOB NOT NULL,
            embedding VECTOR({embedding_dimension}, FLOAT32) NOT NULL,
            document_type VARCHAR2(128) NOT NULL,
            source_file VARCHAR2(128),
            created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            PRIMARY KEY (doc_id, chunk_id),
            FOREIGN KEY (doc_id)
                REFERENCES {docs_table} (doc_id)
                ON DELETE CASCADE,
            CHECK ((chunk_id = 1 AND source_file IS NOT NULL) OR (chunk_id > 1 AND source_file IS NULL))
        )
        """,
        f"CREATE INDEX {index_name} ON {chunks_table} (document_type)",
    ]


def create_schema_if_needed(
    connection: oracledb.Connection,
    sql_path: Path,
    docs_table: str,
    chunks_table: str = DEFAULT_TABLE_CHUNKS,
    embedding_dimension: int = 2048,
) -> None:
    cursor = connection.cursor()
    docs_table = normalize_table_name(docs_table)
    chunks_table = normalize_table_name(chunks_table)
    del sql_path
    for statement in build_schema_statements(
        docs_table=docs_table,
        chunks_table=chunks_table,
        embedding_dimension=embedding_dimension,
    ):
        try:
            cursor.execute(statement)
        except oracledb.DatabaseError as exc:
            error_obj = exc.args[0]
            if getattr(error_obj, "code", None) not in {955, 2261, 1408}:
                raise
    existing_dimension = get_existing_vector_dimension(connection, chunks_table)
    if existing_dimension is not None and existing_dimension != embedding_dimension:
        raise RuntimeError(
            f"Oracle vector column dimension mismatch for {chunks_table}. "
            f"Expected {embedding_dimension}, found {existing_dimension}. "
            f"Please use a different table name or recreate the table for the new embedding model."
        )
    try:
        cursor.execute(
            f"ALTER TABLE {docs_table} ADD (document_text CLOB)"
        )
    except oracledb.DatabaseError as exc:
        error_obj = exc.args[0]
        if getattr(error_obj, "code", None) != 1430:
            raise
    connection.commit()


def insert_document_if_new(
    connection: oracledb.Connection,
    document_name: str,
    document_text: str,
    document_type: str,
    source_file: str,
    docs_table: str,
) -> tuple[int, bool]:
    cursor = connection.cursor()
    inserted = False
    try:
        cursor.execute(
            f"""
            INSERT INTO {docs_table} (
                document_name,
                document_text,
                document_type,
                source_file
            ) VALUES (
                :document_name,
                :document_text,
                :document_type,
                :source_file
            )
            """,
            document_name=document_name,
            document_text=document_text,
            document_type=document_type,
            source_file=source_file,
        )
        connection.commit()
        inserted = True
    except oracledb.DatabaseError as exc:
        error_obj = exc.args[0]
        if getattr(error_obj, "code", None) == 1:
            print(
                f"Document already exists in Oracle, skip insert: {document_name}"
            )
        else:
            raise

    cursor.execute(
        f"SELECT doc_id FROM {docs_table} WHERE document_name = :document_name",
        document_name=document_name,
    )
    doc_id = int(cursor.fetchone()[0])
    return doc_id, inserted


def insert_chunks(
    connection: oracledb.Connection,
    doc_id: int,
    rows: Iterable[ChunkRow],
    chunks_table: str,
) -> None:
    cursor = connection.cursor()
    data = []
    for row in rows:
        data.append(
            {
                "doc_id": doc_id,
                "chunk_id": row.chunk_id,
                "chunk_tokens": row.chunk_tokens,
                "chunk_text": row.chunk_text,
                "embedding": array.array("f", row.embedding),
                "document_type": row.document_type,
                "source_file": row.source_file,
            }
        )
    try:
        cursor.executemany(
            f"""
            INSERT INTO {chunks_table} (
                doc_id,
                chunk_id,
                chunk_tokens,
                chunk_text,
                embedding,
                document_type,
                source_file
            ) VALUES (
                :doc_id,
                :chunk_id,
                :chunk_tokens,
                :chunk_text,
                :embedding,
                :document_type,
                :source_file
            )
            """,
            data,
        )
        connection.commit()
    except oracledb.DatabaseError as exc:
        error_obj = exc.args[0]
        if getattr(error_obj, "code", None) == 1:
            print(f"Chunk insert skipped because rows already exist for doc_id={doc_id}")
        else:
            raise


def search_chunks(
    connection: oracledb.Connection,
    query_embedding: list[float],
    top_k: int,
    chunks_table: str,
    docs_table: str,
    document_type: str | None,
) -> list[dict]:
    cursor = connection.cursor()
    sql = f"""
        SELECT
            d.doc_id,
            d.document_name,
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
        FETCH FIRST {top_k} ROWS ONLY
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
                "chunk_id": int(result[2]),
                "chunk_tokens": int(result[3]),
                "chunk_text": result[4].read() if hasattr(result[4], "read") else result[4],
                "document_type": result[5],
                "source_file": result[6],
                "distance": float(result[7]),
            }
        )
    return rows


def print_ingest_summary(
    document_name: str,
    page_count: int | None,
    document_type: str,
    full_text: str,
    chunk_rows: list[ChunkRow],
    docs_table: str,
    chunks_table: str,
) -> None:
    print(f"Document: {document_name}")
    print(f"Document type: {document_type}")
    print(f"Pages: {page_count if page_count is not None else 'N/A for txt'}")
    print(f"Full text length: {len(full_text)} characters")
    print(f"Chunks inserted: {len(chunk_rows)}")
    print(f"Docs table: {docs_table}")
    print(f"Chunks table: {chunks_table}")
    if chunk_rows:
        print("First chunk preview:")
        print(preview_text(chunk_rows[0].chunk_text, max_chars=400))


def print_search_results(results: list[dict], rerank_scores_list: list[float], top_k: int) -> None:
    print("\nVector search results after rerank:")
    ranked = sorted(
        zip(results, rerank_scores_list),
        key=lambda item: item[1],
        reverse=True,
    )
    for rank, (row, score) in enumerate(ranked[:top_k], start=1):
        print(
            f"Top {rank}: doc={row['document_name']} chunk={row['chunk_id']} "
            f"distance={row['distance']:.6f} rerank_score={score:.6f}"
        )
        print(f"  tokens={row['chunk_tokens']} source_file={row['source_file']}")
        print(f"  {preview_text(row['chunk_text'], max_chars=260)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangChain + Oracle vector search demo using local embedding and reranker models."
    )
    parser.add_argument("--device", default="cpu", help="auto, cuda, or cpu")
    parser.add_argument("--proxy", default=DEFAULT_PROXY, help="HTTP proxy for model download")
    parser.add_argument("--download", action="store_true", help="Download missing embedding model")
    parser.add_argument("--input-path", "--pdf-path", default=str(PDF_PATH), help="Input file path. Supports .pdf and .txt")
    parser.add_argument("--query", default=DEFAULT_PDF_QUERY, help="Semantic search query")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Embedding instruction")
    parser.add_argument(
        "--reranker-instruction",
        default=DEFAULT_RERANK_INSTRUCTION,
        help="Reranker instruction",
    )
    parser.add_argument(
        "--embedding-model-dir",
        default=str(MODEL_DIR),
        help="Local embedding model directory",
    )
    parser.add_argument(
        "--reranker-model-dir",
        default=str(RERANKER_MODEL_DIR),
        help="Local reranker model directory",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help="Token size per chunk",
    )
    parser.add_argument(
        "--chunk-overlap-ratio",
        type=float,
        default=DEFAULT_CHUNK_OVERLAP_RATIO,
        help="Chunk overlap ratio, default 0.1 means 10 percent",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Final output count")
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=DEFAULT_FETCH_K,
        help="Oracle vector search fetch count before rerank",
    )
    parser.add_argument(
        "--document-type",
        default=None,
        help="Document type metadata stored with each chunk. Defaults to the input file extension.",
    )
    parser.add_argument(
        "--source-file",
        default=DEFAULT_SOURCE_FILE,
        help="Source file value stored only on chunk_id=1",
    )
    parser.add_argument(
        "--schema-sql",
        default=str(Path(__file__).resolve().parent / "oracle_vector_schema.sql"),
        help="DDL file for table creation",
    )
    parser.add_argument(
        "--docs-table",
        default=DEFAULT_TABLE_DOCS,
        help="Document master table name",
    )
    parser.add_argument(
        "--chunks-table",
        default=DEFAULT_TABLE_CHUNKS,
        help="Chunk table name with vector column",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip PDF split and insert, only run query + rerank against existing data",
    )
    parser.add_argument(
        "--use-deepseek-summary",
        action="store_true",
        help="Call DeepSeek once on the full document and append `# Deepseek Summary` before split.",
    )
    args = parser.parse_args()

    ensure_proxy(args.proxy)
    device = resolve_device(args.device)
    embedding_model_dir = ensure_model(
        Path(args.embedding_model_dir),
        args.proxy,
        args.download,
        EMBEDDING_REPO_ID,
    )
    reranker_model_dir = ensure_model(
        Path(args.reranker_model_dir),
        args.proxy,
        False,
        RERANKER_REPO_ID,
    )

    embedder = LocalTextEmbeddings(
        model_dir=embedding_model_dir,
        instruction=args.instruction,
        device=device,
    )
    splitter = build_token_splitter(
        tokenizer_holder=embedder,
        chunk_tokens=max(args.chunk_size_tokens, 1),
        overlap_ratio=max(min(args.chunk_overlap_ratio, 0.5), 0.0),
    )

    connection = connect_oracle_from_env()
    create_schema_if_needed(
        connection,
        Path(args.schema_sql),
        args.docs_table,
        args.chunks_table,
        embedder.embedding_dimension,
    )

    if not args.skip_ingest:
        input_path = Path(args.input_path)
        documents, page_count, full_text = load_source_documents(input_path)
        if args.use_deepseek_summary:
            full_text = append_deepseek_summary_to_document(
                document_name=input_path.name,
                document_type=input_path.suffix.lstrip(".").lower() or DEFAULT_DOC_TYPE,
                full_text=full_text,
            )
            documents = [
                Document(
                    page_content=full_text,
                    metadata=documents[0].metadata if documents else {
                        "document_name": input_path.name,
                        "document_type": input_path.suffix.lstrip(".").lower() or DEFAULT_DOC_TYPE,
                    },
                )
            ]
        document_type = args.document_type or input_path.suffix.lstrip(".").lower() or DEFAULT_DOC_TYPE
        split_docs = split_documents(documents, splitter)
        chunk_rows = chunk_rows_from_documents(
            chunks=split_docs,
            embedder=embedder,
            tokenizer_holder=embedder,
            document_name=input_path.name,
            document_type=document_type,
            source_file=args.source_file,
        )
        doc_id, inserted = insert_document_if_new(
            connection=connection,
            document_name=input_path.name,
            document_text=full_text,
            document_type=document_type,
            source_file=args.source_file,
            docs_table=args.docs_table,
        )
        if inserted:
            insert_chunks(connection, doc_id, chunk_rows, args.chunks_table)
            print_ingest_summary(
                document_name=input_path.name,
                page_count=page_count,
                document_type=document_type,
                full_text=full_text,
                chunk_rows=chunk_rows,
                docs_table=args.docs_table,
                chunks_table=args.chunks_table,
            )
        else:
            print("Skip ingest and reuse existing Oracle rows.")

    query_document_type = args.document_type
    if query_document_type is None and not args.skip_ingest:
        query_document_type = document_type

    query_embedding = embedder.embed_query(args.query)
    candidates = search_chunks(
        connection=connection,
        query_embedding=query_embedding,
        top_k=max(args.fetch_k, args.top_k),
        chunks_table=args.chunks_table,
        docs_table=args.docs_table,
        document_type=query_document_type,
    )
    if not candidates:
        print("No Oracle vector search results found.")
        return

    reranker = load_reranker_runtime(
        reranker_model_dir,
        device,
        configured_backend=RERANKER_BACKEND,
    )
    rerank_score_list = reranker.score(
        query=args.query,
        documents=[
            build_retrieval_text(row["document_name"], row["chunk_text"])
            for row in candidates
        ],
        instruction=args.reranker_instruction,
        batch_size=1 if device == "cpu" else 4,
    )
    print_search_results(candidates, rerank_score_list, args.top_k)


if __name__ == "__main__":
    main()
