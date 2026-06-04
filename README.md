# Oracle Vector Search Demo

一个基于本地 Embedding / Reranker 模型、Oracle Vector、FastAPI 的 RAG 示例项目。

项目支持三套检索链路：

- `Qwen3-VL-Embedding-2B` + `Qwen3-VL-Reranker-2B`
- `bge-m3` + `bge-reranker-v2-m3`
- `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`（轻量快速，适合 CPU）

你可以用它完成以下流程：

- 上传 `PDF` / `TXT` 文档
- 进行文本解析与分片
- 生成向量并写入 Oracle Vector
- 通过向量检索召回相关分片
- 使用本地 reranker 进行重排
- 通过 FastAPI 对外提供统一接口

## Features

- 支持本地 Embedding 与 Reranker 推理
- 支持 `Qwen`、`BGE`、`MiniLM` 三套模型方案切换
- 支持 `PDF` / `TXT` 文档上传
- 支持 Oracle 原生 `VECTOR(dim, FLOAT32)` 向量列
- 支持 `title` / `split` / `full` 三种检索返回模式
- 支持使用 DeepSeek 对全文做一次摘要增强
- 支持命令行 Demo 和 HTTP API 两种使用方式

## Architecture

整体流程如下：

1. 上传文档到 `POST /documents/upload`
2. 服务解析文档全文
3. 按 token 进行分片
4. 使用本地 embedding 模型生成向量
5. 将完整文档写入 `*_DOCUMENTS`
6. 将分片与向量写入 `*_CHUNKS`
7. 调用 `POST /search` 时生成 query embedding
8. 在 Oracle 中进行向量召回
9. 使用本地 reranker 对召回结果重排
10. 按 `title` / `split` / `full` 返回结果

## Model Variants

### Qwen 链路

- Embedding: `Qwen3-VL-Embedding-2B`
- Reranker: `Qwen3-VL-Reranker-2B`
- 默认端口: `19000`
- 默认文档表: `LC_DEMO_DOCUMENTS`
- 默认分片表: `LC_DEMO_CHUNKS`
- 默认向量维度: `2048`

### BGE 链路

- Embedding: `bge-m3`
- Reranker: `bge-reranker-v2-m3`
- 默认端口: `19001`
- 默认文档表: `LC_DEMO_DOCUMENTS_BGE`
- 默认分片表: `LC_DEMO_CHUNKS_BGE`
- 默认向量维度: `1024`

### MiniLM 链路

- Embedding: `all-MiniLM-L6-v2`（22.7M 参数，CPU 约 5 秒）
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`（CrossEncoder，CPU 约 15 秒）
- 默认端口: `19002`
- 默认文档表: `LC_DEMO_DOCUMENTS_MINI`
- 默认分片表: `LC_DEMO_CHUNKS_MINI`
- 默认向量维度: `384`

说明：

- 三套模型的向量维度不同，不能共用同一张 Oracle 向量表
- 文档上传到哪一套 API，搜索也建议走同一套 API
- MiniLM 专为 CPU 环境优化，embedding 耗时约 5 秒（BGE-M3 约 188 秒）

## Project Structure

```text
.
├── README.md
├── INSTALL.md
├── API调用说明.md
├── DDL.md
├── requirements.txt
├── model_config.json
├── oracle_config.json
├── embed_text.py
├── oracle_langchain_demo.py
├── oracle_vector_api.py
├── oracle_vector_api_bge.py
├── oracle_vector_api_mini.py
├── oracle_vector_schema.sql
├── start_qwen_oracle_vector_api.bat
├── start_bge_oracle_vector_api.bat
├── start_mini_oracle_vector_api.sh
├── test_upload_doc.py
├── test_vectoer_search.py
├── test_api_upload.py
├── test_api_search.py
├── test_api_upload_mini.py
├── test_api_search_mini.py
├── Qwen3-VL-Embedding-2B/
├── Qwen3-VL-Reranker-2B/
├── bge-m3/
├── bge-reranker-v2-m3/
├── all-MiniLM-L6-v2/
└── ms-marco-MiniLM-L-6-v2/
```

## Requirements

- Windows
- Python `3.12` 推荐
- Oracle Database，且支持向量能力
- 本地模型目录已准备完成，或可联网下载模型
- 如果启用摘要增强，需要配置 `DEEPSEEK_API_KEY`
- 如果需要 GPU 推理，建议安装 CUDA 版 PyTorch

## Quick Start

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

如果你只想快速安装基础依赖，也可以执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

如果你需要 GPU：

```powershell
powershell -ExecutionPolicy Bypass -File .\install_cuda_torch.ps1
```

### 3. 配置模型和数据库

先检查两个配置文件：

- `model_config.json`
- `oracle_config.json`

推荐至少确认以下字段：

```json
{
  "embedding_model_dir": "./Qwen3-VL-Embedding-2B",
  "reranker_model_dir": "./Qwen3-VL-Reranker-2B",
  "proxy": "http://127.0.0.1:7897"
}
```

```json
{
  "user": "your_user",
  "password": "your_password",
  "dsn": "host:1521/service_name"
}
```

如果要启用 DeepSeek 摘要增强：

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
```

可选：

```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

### 4. 启动服务

启动 Qwen 版本：

```powershell
.\start_qwen_oracle_vector_api.bat
```

启动 BGE 版本：

```powershell
.\start_bge_oracle_vector_api.bat
```

启动 MiniLM 版本（Linux）：

```bash
chmod +x start_mini_oracle_vector_api.sh
./start_mini_oracle_vector_api.sh
```

启动成功后，默认访问地址：

- Qwen: `http://127.0.0.1:19000`
- BGE: `http://127.0.0.1:19001`
- MiniLM: `http://127.0.0.1:19002`

健康检查：

```powershell
curl http://127.0.0.1:19000/health
curl http://127.0.0.1:19001/health
curl http://127.0.0.1:19002/health
```

## API Overview

### `GET /health`

健康检查接口。

返回示例：

```json
{
  "status": "ok"
}
```

### `POST /documents/upload`

上传文档、切分、向量化并写入 Oracle。

支持参数：

- `file`
- `chunk_size_tokens`
- `chunk_overlap_tokens`
- `source_file`
- `document_type`
- `instruction`
- `use_deepseek_summary`
- `device`
- `docs_table`
- `chunks_table`

示例：

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

执行 Oracle 向量检索并使用 reranker 重排。

请求体示例：

```json
{
  "prompt": "如何避免 oracle corruption",
  "top_k": 5,
  "reranker_score": 0.3,
  "return_mode": "split",
  "device": "auto",
  "document_type": "pdf"
}
```

`return_mode` 说明：

- `title`: 只返回文档名
- `split`: 返回命中的相关分片
- `full`: 返回完整文档内容

## CLI Demos

### 本地 Embedding / Reranker 演示

```powershell
.\.venv\Scripts\python.exe .\embed_text.py --device cuda
```

只预览 PDF 抽取和分片：

```powershell
.\.venv\Scripts\python.exe .\embed_text.py --preview-pdf-only
```

### Oracle + LangChain 全流程演示

```powershell
.\.venv\Scripts\python.exe .\oracle_langchain_demo.py --device cuda
```

只搜索、不重新入库：

```powershell
.\.venv\Scripts\python.exe .\oracle_langchain_demo.py --device cuda --skip-ingest
```

启用 DeepSeek 摘要增强：

```powershell
.\.venv\Scripts\python.exe .\oracle_langchain_demo.py --device cuda --use-deepseek-summary
```

## Test Scripts

项目自带几个简单测试脚本，方便快速验证 API：

### 上传测试

```powershell
.\.venv\Scripts\python.exe .\test_upload_doc.py .\test.txt --port 19000
.\.venv\Scripts\python.exe .\test_upload_doc.py .\test.txt --port 19001
```

MiniLM 上传（Linux）：

```bash
python test_api_upload_mini.py ./test.txt
```

### 搜索测试

```powershell
.\.venv\Scripts\python.exe .\test_vectoer_search.py --port 19000 --prompt "ASM 实例同步时发生崩溃"
.\.venv\Scripts\python.exe .\test_vectoer_search.py --port 19001 --prompt "ASM 实例同步时发生崩溃"
```

MiniLM 搜索（Linux）：

```bash
python test_api_search_mini.py "你的查询" --top-k 5 --device cpu
```

### 固定脚本示例

```powershell
.\.venv\Scripts\python.exe .\test_api_upload.py
.\.venv\Scripts\python.exe .\test_api_search.py
```

## Swagger Docs

FastAPI 默认提供在线文档：

- Qwen Swagger: `http://127.0.0.1:19000/docs`
- Qwen ReDoc: `http://127.0.0.1:19000/redoc`
- BGE Swagger: `http://127.0.0.1:19001/docs`
- BGE ReDoc: `http://127.0.0.1:19001/redoc`
- MiniLM Swagger: `http://127.0.0.1:19002/docs`
- MiniLM ReDoc: `http://127.0.0.1:19002/redoc`

## Database Notes

当前默认使用两类表：

- 文档主表：保存完整文档内容
- 分片表：保存 chunk 文本和 embedding 向量

典型表名如下：

- Qwen: `LC_DEMO_DOCUMENTS` / `LC_DEMO_CHUNKS`（2048 维）
- BGE: `LC_DEMO_DOCUMENTS_BGE` / `LC_DEMO_CHUNKS_BGE`（1024 维）
- MiniLM: `LC_DEMO_DOCUMENTS_MINI` / `LC_DEMO_CHUNKS_MINI`（384 维）

当前实现特点：

- 按 `document_name` 去重
- 文档主表与分片表通过 `doc_id` 关联
- `chunk_id = 1` 时才写入 `source_file`
- 会在启动或调用时自动尝试建表

## Common Issues

### 1. `DEEPSEEK_API_KEY` 未设置

如果上传时启用了 `use_deepseek_summary=true`，但没有设置环境变量，会报错。

解决：

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
```

### 2. Oracle 向量维度不匹配

常见原因：

- 用 BGE 表去存 Qwen 向量
- 用 Qwen 表去存 BGE 向量
- 往 MiniLM 表写入其他模型的向量（维度不匹配）

解决方式：

- Qwen、BGE、MiniLM 分开使用各自默认表
- 或者为新模型显式指定新的表名

### 3. CUDA 不可用

如果强制指定 `--device cuda`，但当前环境没有正确安装 CUDA 版 PyTorch，会报错。

可先临时改为：

```powershell
--device cpu
```

### 4. 重复上传同名文档

当前按 `document_name` 去重。

如果上传同名文件：

- 文档主表不会重复插入
- 分片也不会重复写入
- 返回结果中的 `inserted` 会是 `false`

## Related Docs

- 安装部署说明: [INSTALL.md](./INSTALL.md)
- API 调用说明: [API调用说明.md](./API调用说明.md)
- 数据结构说明: [DDL.md](./DDL.md)
- Oracle 建表 SQL: [oracle_vector_schema.sql](./oracle_vector_schema.sql)

## Recommended Scenarios

- 想验证本地向量化和 rerank 效果：使用 `embed_text.py`
- 想验证 Oracle 入库和检索链路：使用 `oracle_langchain_demo.py`
- 想给外部系统提供统一 HTTP 能力：启动 `oracle_vector_api.py`
- 想比较不同模型方案：分别启动 Qwen、BGE、MiniLM 三套 API
- CPU 环境优先推荐 MiniLM 链路，embedding ~5 秒，搜索全程 ~20 秒

## License

当前仓库未提供单独许可证文件。如需开源发布，建议补充 `LICENSE`。
