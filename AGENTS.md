# AGENTS.md — oracle26ai_vector_searchapi

## Project

RAG demo: local embedding/reranker → Oracle Vector (23ai) → FastAPI. Two model stacks that **cannot** share vector tables.

## Stacks

| Stack | Embedding | Reranker | Port | Dim | Docs table | Chunks table |
|-------|-----------|----------|------|-----|------------|--------------|
| Qwen  | Qwen3-VL-Embedding-2B | Qwen3-VL-Reranker-2B | 19000 | 2048 | LC_DEMO_DOCUMENTS | LC_DEMO_CHUNKS |
| BGE   | bge-m3 | bge-reranker-v2-m3 | 19001 | 1024 | LC_DEMO_DOCUMENTS_BGE | LC_DEMO_CHUNKS_BGE |

- Upload and search on the same stack. Mixing tables will crash (dimension mismatch).

## Config Priority

Env var > `model_config.json` > code default. Key env vars: `EMBEDDING_MODEL_DIR`, `RERANKER_MODEL_DIR`, `VECTOR_DOCS_TABLE`, `VECTOR_CHUNKS_TABLE`, `MODEL_PROXY`, `EMBEDDING_BACKEND`, `RERANKER_BACKEND`, `DEEPSEEK_API_KEY`.

## Commands (PowerShell, project root)

```powershell
.venv\Scripts\Activate.ps1

# Start API
.\start_qwen_oracle_vector_api.bat   # port 19000
.\start_bge_oracle_vector_api.bat    # port 19001

# CLI demos
.\.venv\Scripts\python.exe .\oracle_langchain_demo.py --device cuda
.\.venv\Scripts\python.exe .\embed_text.py --device cuda

# Test scripts (API must be running)
.\.venv\Scripts\python.exe .\test_upload_doc.py .\test.txt --port 19000
.\.venv\Scripts\python.exe .\test_vectoer_search.py --port 19000 --prompt "query"
.\.venv\Scripts\python.exe .\test_api_upload.py              # hardcoded port 19000
.\.venv\Scripts\python.exe .\test_api_search.py              # hardcoded port 19000

# Health
curl http://127.0.0.1:19000/health
# Swagger
http://127.0.0.1:19000/docs
```

## DeepSeek Summary

Requires `$env:DEEPSEEK_API_KEY="..."`. Optional: `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `DEEPSEEK_REASONING_EFFORT`. Appends `# Deepseek Summary` section to full text before splitting.

## Startup Scripts (.bat)

Both scripts kill the port's old process, set env vars (model dirs, table names, proxy), then start via `uvicorn oracle_vector_api:app`. The `.bat` env vars override `model_config.json`.

## Device

`auto` → `cuda` if `torch.cuda.is_available()` else `cpu`. Specifying `cuda` without GPU raises `RuntimeError`.

## Search Internals

1. Embed query with local model
2. Oracle vector search (`COSINE` distance) on chunks table
3. Rerank with local model (batch_size = 4 CUDA, 1 CPU)
4. Aggregate by document, filter by `reranker_score` threshold

Retrieval text format: `"Title: {doc_name}\nContent:\n{chunk_text}"` — used for both embedding (upload) and reranker (search).

## Oracle Schema

- Tables auto-created at runtime via Python DDL
- Dedup by `document_name` UNIQUE constraint (no content hash)
- `chunk_id=1` stores `source_file`, others get NULL (CHECK constraint enforces this)
- `doc_id` + `chunk_id` composite PK

## Return Modes

- `title` — document name only
- `split` — doc name + passing chunks (sorted by chunk_id)
- `full` — doc name + full `document_text`

## Key Files

| File | Role |
|------|------|
| `embed_text.py` | Core: embedding/reranker runtime, backends (qwen_vl, bge_m3, qwen_vl_yesno, bge_seq_cls) |
| `oracle_langchain_demo.py` | CLI demo: Oracle + LangChain + embedding + reranker |
| `oracle_vector_api.py` | FastAPI app: upload + search endpoints |
| `model_config.json` | Default model paths/proxy/backends |
| `oracle_config.json` | Oracle connection (dsn, user, password) |
| `requirements.txt` | Frozen deps (torch 2.11.0+cu128, transformers 5.7.0, FastAPI, etc.) |

## No test framework

Standalone Python test scripts only. No pytest, no lint/typecheck config.
