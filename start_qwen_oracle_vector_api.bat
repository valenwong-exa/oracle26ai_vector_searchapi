@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PORT=19000"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

echo Checking port %PORT% ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo Killing process %%a on port %PORT% ...
    taskkill /F /PID %%a >nul 2>nul
)

if not exist "%PYTHON_EXE%" (
    echo Python not found: %PYTHON_EXE%
    exit /b 1
)

set "EMBEDDING_MODEL_DIR=%SCRIPT_DIR%Qwen3-VL-Embedding-2B"
set "RERANKER_MODEL_DIR=%SCRIPT_DIR%Qwen3-VL-Reranker-2B"
set "EMBEDDING_BACKEND=auto"
set "RERANKER_BACKEND=auto"
set "MODEL_PROXY=http://127.0.0.1:7897"
set "VECTOR_DOCS_TABLE=LC_DEMO_DOCUMENTS"
set "VECTOR_CHUNKS_TABLE=LC_DEMO_CHUNKS"

echo Starting Oracle Vector API with Qwen models on port %PORT% ...
echo EMBEDDING_MODEL_DIR=%EMBEDDING_MODEL_DIR%
echo RERANKER_MODEL_DIR=%RERANKER_MODEL_DIR%
echo VECTOR_DOCS_TABLE=%VECTOR_DOCS_TABLE%
echo VECTOR_CHUNKS_TABLE=%VECTOR_CHUNKS_TABLE%
start "oracle-vector-api-qwen" cmd /k ""%PYTHON_EXE%" -m uvicorn oracle_vector_api:app --host 0.0.0.0 --port %PORT%"

echo Done.
endlocal
