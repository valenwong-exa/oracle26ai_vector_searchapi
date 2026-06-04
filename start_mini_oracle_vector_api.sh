#!/bin/bash
# ================================================================
# Start Oracle Vector API with all-MiniLM-L6-v2 (384-dim)
# Port 19002 — independent from BGE (19001)
# ================================================================

# Switch to the directory where this script lives
cd "$(dirname "$0")" || exit 1

# Kill old process on port 19002
fuser -k 19002/tcp 2>/dev/null; sleep 1

# ----- 1. Download models (run once) -----
#   huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
#     --local-dir /u01/app/embeding/all-MiniLM-L6-v2
#   huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
#     --local-dir /u01/app/embeding/ms-marco-MiniLM-L-6-v2

# ----- 2. Environment variables -----
export EMBEDDING_MODEL_DIR="/u01/app/embeding/all-MiniLM-L6-v2"
export EMBEDDING_REPO_ID="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_BACKEND="auto"

# Fast CPU reranker: ms-marco-MiniLM-L-6-v2 (CrossEncoder, bert)
export RERANKER_MODEL_DIR="/u01/app/embeding/ms-marco-MiniLM-L-6-v2"
export RERANKER_REPO_ID="cross-encoder/ms-marco-MiniLM-L-6-v2"
export RERANKER_BACKEND="auto"

# New Oracle tables for 384-dim vectors (will be auto-created)
export VECTOR_DOCS_TABLE="LC_DEMO_DOCUMENTS_MINI"
export VECTOR_CHUNKS_TABLE="LC_DEMO_CHUNKS_MINI"

# Proxy for model download (if needed)
export MODEL_PROXY="http://127.0.0.1:7897"

# ----- 3. Start MiniLM API on port 19002 -----
nohup uvicorn oracle_vector_api_mini:app \
  --host 0.0.0.0 --port 19002 \
  >> api_minilm.log 2>&1 &

echo "all-MiniLM-L6-v2 API starting on http://192.168.56.27:19002"
echo "Logs: tail -f api_minilm.log"
