export EMBEDDING_MODEL_DIR="/u01/app/embeding/Qwen3-VL-Embedding-2B"
export RERANKER_MODEL_DIR="/u01/app/embeding/Qwen3-VL-Reranker-2B"
export EMBEDDING_BACKEND="auto"
export RERANKER_BACKEND="auto"
export MODEL_PROXY="http://127.0.0.1:7897"
export VECTOR_DOCS_TABLE="LC_DEMO_DOCUMENTS"
export VECTOR_CHUNKS_TABLE="LC_DEMO_CHUNKS"

nohup uvicorn oracle_vector_api:app --host 0.0.0.0 --port 19000 >> api.log 2>&1 &
