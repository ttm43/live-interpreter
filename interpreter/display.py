"""Console output: live partial line + finalized bilingual segments."""
import sys
import threading


class ConsoleDisplay:
    """Thread-safe console renderer with an in-place updating partial line."""

    def __init__(self):
        self._lock = threading.Lock()
        self._partial_shown = False

    def _clear_partial(self) -> None:
        if self._partial_shown:
            sys.stdout.write("\r" + " " * 100 + "\r")
            self._partial_shown = False

    def partial(self, text: str) -> None:
        with self._lock:
            line = f"  … {text}"
            sys.stdout.write("\r" + " " * 100 + "\r" + line[:98])
            sys.stdout.flush()
            self._partial_shown = True

    def final_source(self, text: str, lang: str) -> None:
        with self._lock:
            self._clear_partial()
            print(f"[{lang}] {text}")
            sys.stdout.flush()

    def translation(self, text: str, lang: str, latency_s: float) -> None:
        with self._lock:
            self._clear_partial()
            print(f"  -> [{lang}] {text}   ({latency_s:.1f}s)")
            sys.stdout.flush()

    def info(self, msg: str) -> None:
        with self._lock:
            self._clear_partial()
            print(msg)
            sys.stdout.flush()
