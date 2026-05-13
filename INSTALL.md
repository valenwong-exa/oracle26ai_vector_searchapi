项目安装部署说明

本文档用于说明本项目在 Windows + PowerShell 环境下的 step by step 安装、配置、启动和验证方法。


一、项目功能概览

本项目包含以下主要能力：

1. 本地 embedding / reranker 演示
   - 脚本：`embed_text.py`
   - 支持 `Qwen3-VL-Embedding-2B` 和 `bge-m3`
   - 支持 `Qwen3-VL-Reranker-2B` 和 `bge-reranker-v2-m3`
   - 支持读取 PDF 或 TXT

2. Oracle + LangChain 全流程演示
   - 脚本：`oracle_langchain_demo.py`
   - 支持文档切分、embedding、写入 Oracle、向量检索、rerank

3. FastAPI 服务
   - 脚本：`oracle_vector_api.py`
   - 提供上传文档接口和向量搜索接口
   - 支持 DeepSeek 对全文先做一次摘要增强
   - 支持 Qwen / BGE 两套模型分别启动


二、目录中关键文件

- `requirements.txt`
  - 已由当前 `.venv` 实际安装的 Python 依赖导出

- `model_config.json`
  - 默认模型路径配置

- `oracle_config.json`
  - Oracle 连接配置

- `start_qwen_oracle_vector_api.bat`
  - 启动 Qwen API

- `start_bge_oracle_vector_api.bat`
  - 启动 BGE API

- `API调用说明.md`
  - API 调用文档

- `数据结构说明.md`
  - Oracle 表结构说明


三、安装前准备

请先准备好以下环境：

1. 操作系统
   - Windows

2. Python
   - 建议 Python 3.12

3. Oracle 数据库
   - 当前项目测试连接信息写在 `oracle_config.json`
   - 当前数据库地址：
     - 用户名：`valen`
     - 密码：`oracle`
     - DSN：`192.168.56.101:1521/aidemo_pdb`

4. 模型目录
   - Qwen embedding：`Qwen3-VL-Embedding-2B`
   - Qwen reranker：`Qwen3-VL-Reranker-2B`
   - BGE embedding：`bge-m3`
   - BGE reranker：`bge-reranker-v2-m3`

5. DeepSeek Key
   - 需要设置环境变量 `DEEPSEEK_API_KEY`

6. 代理
   - 如需联网下载模型或访问外网，默认代理为：
     - `http://127.0.0.1:7897`


四、Step 1：创建虚拟环境

在项目根目录执行：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```


五、Step 2：升级 pip

```powershell
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
```


六、Step 3：安装 Python 依赖

当前项目依赖已经导出到 `requirements.txt`。

直接安装：

```powershell
pip install -r .\requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

说明：

- `requirements.txt` 是从当前可运行环境导出的实际依赖清单
- 如果你只想快速装基础依赖，也可以执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```


七、Step 4：安装或校验 CUDA 版 PyTorch

如果你需要使用 GPU，请执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install_cuda_torch.ps1
```

校验 CUDA 是否可用：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
```

如果输出中 `torch.cuda.is_available()` 为 `True`，说明 GPU 可用。

如果只想使用 CPU，可以跳过这一步。


八、Step 5：检查配置文件

1. 检查模型配置文件 `model_config.json`

当前默认读取该文件中的模型路径和代理配置。

补充说明：

- 如果你是通过 `start_qwen_oracle_vector_api.bat` 或 `start_bge_oracle_vector_api.bat` 启动 API，这两个 bat 会先通过环境变量显式指定模型目录、backend 和默认表名，因此会优先覆盖 `model_config.json`
- `model_config.json` 仍然有用，但主要作为非 bat 启动场景下的默认配置，例如手工运行 `embed_text.py`、`oracle_langchain_demo.py` 或直接启动 `oracle_vector_api.py`
- 当前配置优先级为：环境变量 > `model_config.json` > 代码内置默认值

示例：

```json
{
  "embedding_model_dir": "./Qwen3-VL-Embedding-2B",
  "embedding_backend": "auto",
  "embedding_repo_id": "Qwen/Qwen3-VL-Embedding-2B",
  "reranker_model_dir": "./Qwen3-VL-Reranker-2B",
  "reranker_backend": "auto",
  "reranker_repo_id": "Qwen/Qwen3-VL-Reranker-2B",
  "proxy": "http://127.0.0.1:7897"
}
```

2. 检查 Oracle 配置文件 `oracle_config.json`

当前 API 服务会读取该文件连接数据库。

请确认至少包含：

- `user`
- `password`
- `dsn`


九、Step 6：设置 DeepSeek 环境变量

在当前 PowerShell 窗口执行：

```powershell
$env:DEEPSEEK_API_KEY="你的DeepSeekKey"
```

可选环境变量：

```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

说明：

- 上传文档时，系统会在 split 之前先对全文调用一次 DeepSeek
- DeepSeek 返回的摘要会追加到文档末尾，章节名固定为 `# Deepseek Summary`
- 不是给每个 chunk 单独加 summary，而是全文只加一次


十、Step 7：启动 API

本项目支持两种启动方式：

1. 启动 Qwen API

```powershell
.\start_qwen_oracle_vector_api.bat
```

2. 启动 BGE API

```powershell
.\start_bge_oracle_vector_api.bat
```

说明：

- 两个启动脚本都会先 kill 掉 `19000` 端口占用进程
- 两个脚本都会启动 `oracle_vector_api.py`
- 两者区别在于加载的模型不同、默认 Oracle 表名不同

Qwen 默认表：

- `LC_DEMO_DOCUMENTS`
- `LC_DEMO_CHUNKS`

BGE 默认表：

- `LC_DEMO_DOCUMENTS_BGE`
- `LC_DEMO_CHUNKS_BGE`

这样做的原因是：

- Qwen 向量维度是 `2048`
- BGE 向量维度是 `1024`
- 两种维度不能写入同一个 Oracle 向量列


十一、Step 8：验证服务是否启动成功

浏览器访问或命令行执行：

```powershell
curl http://127.0.0.1:19000/health
```

正常返回：

```json
{"status":"ok"}
```


十二、Step 9：测试上传文档

上传接口：

- `POST /documents/upload`

关键参数：

- `file`
- `chunk_size_tokens`
- `chunk_overlap_tokens`
- `source_file`
- `document_type`
- `instruction`
- `use_deepseek_summary`
- `device`

示例 `curl`：

```powershell
curl -X POST "http://127.0.0.1:19000/documents/upload" `
  -F "file=@E:\embeding_example_2026\Application Containers In Physical Standby Data Guard Environment (Doc ID 2545894.1).pdf" `
  -F "chunk_size_tokens=500" `
  -F "chunk_overlap_tokens=50" `
  -F "source_file=webui" `
  -F "document_type=pdf" `
  -F "use_deepseek_summary=true" `
  -F "device=auto"
```


十三、Step 10：测试搜索

搜索接口：

- `POST /search`

示例请求体：

```json
{
  "prompt": "how to Tuning Network Performance for Data Guard",
  "top_k": 5,
  "reranker_score": 0.5,
  "return_mode": "full",
  "device": "auto"
}
```

PowerShell 示例：

```powershell
$body = @{
  prompt = "how to Tuning Network Performance for Data Guard"
  top_k = 5
  reranker_score = 0.5
  return_mode = "full"
  device = "auto"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:19000/search" -Method Post -ContentType "application/json" -Body $body
```


十四、Step 11：命令行脚本验证

1. 测试本地 embedding / rerank

```powershell
.\.venv\Scripts\python.exe .\embed_text.py --device cuda
```

2. 跑 Oracle + LangChain + embedding + rerank 全流程

```powershell
.\.venv\Scripts\python.exe -u .\oracle_langchain_demo.py --device cuda
```

3. 如果想启用 DeepSeek 摘要增强

```powershell
.\.venv\Scripts\python.exe -u .\oracle_langchain_demo.py --device cuda --use-deepseek-summary
```

4. 只查库，不重新入库

```powershell
.\.venv\Scripts\python.exe -u .\oracle_langchain_demo.py --device cuda --skip-ingest
```

5. 如果临时想用 CPU

```powershell
.\.venv\Scripts\python.exe -u .\oracle_langchain_demo.py --device cpu
```


十五、Step 12：常见问题

1. 报 `No module named 'openai'`
   - 说明当前 `.venv` 没装 `openai`
   - 执行：

```powershell
pip install openai -i https://mirrors.aliyun.com/pypi/simple/
```

2. 报 `DEEPSEEK_API_KEY` 未设置
   - 请先设置：

```powershell
$env:DEEPSEEK_API_KEY="你的DeepSeekKey"
```

3. 报 Oracle 向量维度不匹配
   - 原因通常是：
     - 用 BGE API 写到了 Qwen 的表
     - 或用 Qwen API 写到了 BGE 的表
   - 解决方式：
     - 使用对应的启动脚本
     - 不要混用表

4. 报 CUDA 不可用
   - 先执行 `install_cuda_torch.ps1`
   - 再用 `torch.cuda.is_available()` 验证


十六、当前默认运行约定

1. API 端口
   - `19000`

2. LangChain 切分参数默认值
   - `500 tokens`
   - `10% overlap`

3. 文档去重规则
   - 默认按 `document_name` 去重

4. DeepSeek 摘要策略
   - 对全文调用一次
   - 结果追加到文档末尾
   - 章节名为 `# Deepseek Summary`


十七、部署完成后的建议检查项

部署完成后，建议至少检查以下内容：

1. `GET /health` 返回 200
2. 上传 PDF 成功
3. Oracle 中成功看到文档主表和分片表数据
4. 搜索接口能返回结果
5. Qwen 和 BGE 分别能使用各自表名运行


十八、相关文档

- API 调用说明：`API调用说明.md`
- 数据结构说明：`数据结构说明.md`
- Python 依赖清单：`requirements.txt`


十九、结论

按本文档步骤执行后，可以完成以下部署目标：

1. 创建并激活 Python 虚拟环境
2. 从 `requirements.txt` 安装项目依赖
3. 配置模型、Oracle 和 DeepSeek
4. 启动 Qwen 或 BGE 版本的 API
5. 完成上传、embedding、Oracle 入库、向量检索和 rerank 全流程验证
