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

# Ollama is shared infrastructure: prefer a workspace-wide install in
# <projects-dir>\shared (reused across sibling projects), fall back to a
# project-local copy (what setup.ps1 creates on a fresh clone).
SHARED_DIR = PROJECT_ROOT.parent / "shared"
_OLLAMA_CANDIDATES = [
    SHARED_DIR / "ollama" / "ollama.exe",
    PROJECT_ROOT / "libs" / "ollama" / "ollama.exe",
]
_MODELS_CANDIDATES = [
    SHARED_DIR / "ollama-models",
    PROJECT_ROOT / "models" / "ollama",
]


def find_ollama() -> tuple[Path | None, Path | None]:
    """(ollama.exe, models dir) — shared install first, project-local second."""
    exe = next((p for p in _OLLAMA_CANDIDATES if p.exists()), None)
    models = next((p for p in _MODELS_CANDIDATES if p.exists()), None)
    return exe, models


# ---- llama.cpp (Confucius4-R2T2 ASR runs through llama-server) ---------------

MODEL_STORE = PROJECT_ROOT.parent / "models"   # machine-wide store, registry.yaml inside


def find_llama_server() -> Path | None:
    """Newest shared llama.cpp build (shared/llama.cpp-b<N>), else project-local."""
    builds = sorted(SHARED_DIR.glob("llama.cpp-b*"), key=lambda p: p.name, reverse=True)
    candidates = [b / "llama-server.exe" for b in builds]
    candidates += [SHARED_DIR / "llama.cpp" / "llama-server.exe",
                   PROJECT_ROOT / "libs" / "llama.cpp" / "llama-server.exe"]
    return next((p for p in candidates if p.exists()), None)


def model_path(key: str) -> Path:
    """Resolve a registry key (work\\models\\registry.yaml) to an absolute path.

    Falls back to <project>/models/<key> (a directory, or a single-file model
    saved as <key>.gguf) so a fresh clone without the machine-wide store, or
    with a store that lacks the key, still works after setup.ps1.
    """
    registry = MODEL_STORE / "model_registry.py"
    reason = f"no registry at {registry}"
    if registry.exists():
        if str(MODEL_STORE) not in sys.path:
            sys.path.insert(0, str(MODEL_STORE))
        from model_registry import path_of  # noqa: PLC0415 - optional dependency

        try:
            return path_of(key)
        except (KeyError, FileNotFoundError, ValueError) as e:
            reason = str(e)
    local = PROJECT_ROOT / "models" / key
    for candidate in (local, local.with_name(f"{key}.gguf")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"model {key!r}: {reason}; nothing at {local}[.gguf] either")


def ensure_llama_server(
    model: Path, mmproj: Path | None, port: int, *, ctx: int = 4096, parallel: int = 2,
) -> bool:
    """Start llama-server for `model` (+ audio projector) unless one is already
    healthy on `port`. A server started here is terminated when this process
    exits; a pre-existing one is left alone."""
    import atexit
    import requests

    base = f"http://127.0.0.1:{port}"

    def alive() -> bool:
        try:
            return requests.get(f"{base}/health", timeout=2).ok
        except requests.RequestException:
            return False

    if alive():
        return True
    exe = find_llama_server()
    if exe is None:
        return False
    cmd = [str(exe), "-m", str(model), "-ngl", "99", "-c", str(ctx),
           "--port", str(port), "-np", str(parallel)]
    if mmproj is not None:
        cmd += ["--mmproj", str(mmproj)]
    log = Path(os.environ.get("TEMP", ".")) / f"live-interpreter-llama-server-{port}.log"
    with open(log, "w") as err:
        proc = subprocess.Popen(
            cmd, creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL, stderr=err,
        )
    atexit.register(proc.terminate)
    for _ in range(240):  # a 2 GB GGUF + projector takes ~10-20 s to load
        time.sleep(0.5)
        if alive():
            return True
    return False


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
    exe, models = find_ollama()
    if exe is None:
        return False
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(models or PROJECT_ROOT / "models" / "ollama")
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
