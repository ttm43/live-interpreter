"""Make `python gui.py` work with the system Python.

Dependencies live in the project-local .venv. When the entry scripts are run
with a bare system Python (no venv activated), this shim appends the venv's
site-packages to sys.path so imports resolve. Must be imported before any
third-party import.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_deps() -> None:
    try:
        import sherpa_onnx  # noqa: F401
        import pyaudiowpatch  # noqa: F401
        import numpy  # noqa: F401
        import requests  # noqa: F401
    except ImportError:
        venv_site = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
        if venv_site.is_dir():
            sys.path.append(str(venv_site))
        else:
            raise SystemExit(
                "依赖缺失且未找到 .venv。请在项目目录执行:\n"
                "  python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt"
            )


def ensure_ollama(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Start the bundled portable Ollama if it isn't already running."""
    import requests

    def alive() -> bool:
        try:
            return requests.get(f"{base_url}/api/version", timeout=2).ok
        except requests.RequestException:
            return False

    if alive():
        return True
    exe = PROJECT_ROOT / "libs" / "ollama" / "ollama.exe"
    if not exe.exists():
        return False
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(PROJECT_ROOT / "models" / "ollama")
    subprocess.Popen(
        [str(exe), "serve"], env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        time.sleep(0.5)
        if alive():
            return True
    return False
