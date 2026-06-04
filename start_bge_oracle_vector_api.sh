# 杀旧进程
fuser -k 19001/tcp 2>/dev/null; sleep 1

# 启动 BGE API
export EMBEDDING_MODEL_DIR="/u01/app/embeding/bge-m3"
export RERANKER_MODEL_DIR="/u01/app/embeding/bge-reranker-v2-m3"
export EMBEDDING_BACKEND="auto"
export RERANKER_BACKEND="auto"
export VECTOR_DOCS_TABLE="LC_DEMO_DOCUMENTS_BGE"
export VECTOR_CHUNKS_TABLE="LC_DEMO_CHUNKS_BGE"

nohup uvicorn oracle_vector_api_bge:app --host 0.0.0.0 --port 19001 >> api_bge.log 2>&1 &
