@echo off
rem CLI launcher: starts Ollama (shared install preferred) then the app.
setlocal
set "ROOT=%~dp0"
set "OLLAMA_EXE=%ROOT%..\shared\ollama\ollama.exe"
set "OLLAMA_MODELS=%ROOT%..\shared\ollama-models"
if not exist "%OLLAMA_EXE%" (
    set "OLLAMA_EXE=%ROOT%libs\ollama\ollama.exe"
    set "OLLAMA_MODELS=%ROOT%models\ollama"
)

powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [run] starting Ollama server ...
    start "" /min "%OLLAMA_EXE%" serve
    timeout /t 4 /nobreak >nul
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%app.py" %*
endlocal
