# Oracle Vector Search Demo

A lightweight RAG demo built with local embedding/reranker models, Oracle Vector, and FastAPI.

This project supports two retrieval stacks:

- `Qwen3-VL-Embedding-2B` + `Qwen3-VL-Reranker-2B`
- `bge-m3` + `bge-reranker-v2-m3`

## What It Does

- Upload `PDF` or `TXT` documents
- Parse and split text into chunks
- Generate embeddings with local models
- Store full documents and vectors in Oracle
- Run vector search with reranking
- Expose the workflow through HTTP APIs

## Model Options

### Qwen

- Port: `19000`
- Docs table: `LC_DEMO_DOCUMENTS`
- Chunks table: `LC_DEMO_CHUNKS`
- Vector dimension: `2048`

### BGE

- Port: `19001`
- Docs table: `LC_DEMO_DOCUMENTS_BGE`
- Chunks table: `LC_DEMO_CHUNKS_BGE`
- Vector dimension: `1024`

Because the vector dimensions are different, Qwen and BGE should not share the same Oracle vector table.

## Main Files

- `oracle_vector_api.py`: FastAPI service for upload and search
- `oracle_langchain_demo.py`: end-to-end Oracle + embedding + rerank demo
- `embed_text.py`: local embedding and reranker demo
- `start_qwen_oracle_vector_api.bat`: start Qwen API
- `start_bge_oracle_vector_api.bat`: start BGE API
- `test_upload_doc.py`: upload API test script
- `test_vectoer_search.py`: search API test script

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

Optional helper scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
powershell -ExecutionPolicy Bypass -File .\install_cuda_torch.ps1
```

### 3. Configure Oracle and models

Check these files before running:

- `oracle_config.json`
- `model_config.json`

If you want document summary enhancement, set:

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
```

### 4. Start the API

Qwen:

```powershell
.\start_qwen_oracle_vector_api.bat
```

BGE:

```powershell
.\start_bge_oracle_vector_api.bat
```

## API Endpoints

### `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

### `POST /documents/upload`

Uploads a document, splits it, creates embeddings, and writes the data to Oracle.

Example:

```powershell
curl -X POST "http://127.0.0.1:19000/documents/upload" `
  -F "file=@E:\embeding_example_2026\test.txt" `
  -F "chunk_size_tokens=500" `
  -F "chunk_overlap_tokens=50" `
  -F "source_file=demo" `
  -F "use_deepseek_summary=true" `
  -F "device=auto"
```

### `POST /search`

Runs Oracle vector search and reranking.

Example request body:

```json
{
  "prompt": "how to avoid oracle corruption",
  "top_k": 5,
  "reranker_score": 0.3,
  "return_mode": "split",
  "device": "auto",
  "document_type": "pdf"
}
```

`return_mode` values:

- `title`: document name only
- `split`: matched chunks
- `full`: full document text

## Demo Commands

Local embedding/reranker demo:

```powershell
.\.venv\Scripts\python.exe .\embed_text.py --device cuda
```

End-to-end Oracle demo:

```powershell
.\.venv\Scripts\python.exe .\oracle_langchain_demo.py --device cuda
```

Skip ingest and search existing Oracle data only:

```powershell
.\.venv\Scripts\python.exe .\oracle_langchain_demo.py --device cuda --skip-ingest
```

## Test Scripts

Upload test:

```powershell
.\.venv\Scripts\python.exe .\test_upload_doc.py .\test.txt --port 19000
.\.venv\Scripts\python.exe .\test_upload_doc.py .\test.txt --port 19001
```

Search test:

```powershell
.\.venv\Scripts\python.exe .\test_vectoer_search.py --port 19000 --prompt "ASM instance crashed during sync"
.\.venv\Scripts\python.exe .\test_vectoer_search.py --port 19001 --prompt "ASM instance crashed during sync"
```

## Swagger

- Qwen: `http://127.0.0.1:19000/docs`
- BGE: `http://127.0.0.1:19001/docs`

## Notes

- Documents are deduplicated by `document_name`
- Tables can be created automatically at runtime
- `chunk_id = 1` stores `source_file`
- CPU mode works, but GPU is strongly recommended

## More Docs

- Chinese README: `README.md`
- Installation guide: `INSTALL.md`
- API guide: `API调用说明.md`
- Schema notes: `DDL.md`

## License

This project is licensed under the MIT License. See `LICENSE`.
