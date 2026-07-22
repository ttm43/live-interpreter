@echo off
rem Live interpreter launcher: starts local Ollama (portable) then the app.
setlocal
set "ROOT=%~dp0"
set "OLLAMA_MODELS=%ROOT%models\ollama"

rem Start Ollama server if it is not already listening on 11434
powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [run] starting Ollama server ...
    start "" /min "%ROOT%libs\ollama\ollama.exe" serve
    timeout /t 4 /nobreak >nul
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%app.py" %*
endlocal
