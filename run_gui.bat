@echo off
rem GUI launcher: starts local Ollama (portable) then the Tk interface.
setlocal
set "ROOT=%~dp0"
set "OLLAMA_MODELS=%ROOT%models\ollama"

powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [run] starting Ollama server ...
    start "" /min "%ROOT%libs\ollama\ollama.exe" serve
    timeout /t 4 /nobreak >nul
)

start "" "%ROOT%.venv\Scripts\pythonw.exe" "%ROOT%gui.py"
endlocal
