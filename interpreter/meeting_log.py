"""Per-session meeting log for post-meeting review.

The GUI runs under pythonw and otherwise leaves nothing behind, so every
final of both lanes, every translation (with the corrected source when the
dialogue interpreter changed it), the assistant's verdicts and the status
lines are appended to logs/meeting-<start>.log, one event per line:

    # live-interpreter 2026-09-26 22:00:51  finals=confucius mic=confucius translator=qwen3:4b-instruct
    [22:01:05 +00:14] 对方<TAB>To help Ukraine win the war, ...
    [22:01:06 +00:15] 对方译<TAB>为了帮助乌克兰...  (1.0s)
    [22:01:08 +00:17] 助手<TAB>【意图】... 【无需回应】...
    [22:01:20 +00:29] 我<TAB>I think we should ...
    [22:01:21 +00:30] 我译<TAB>我认为...  (0.8s | 纠错: I think we should ...)

Partials and provisional translations are not logged.
`AppConfig.meeting_log_dir = ""` disables the log.
"""
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .bootstrap import PROJECT_ROOT

log = logging.getLogger(__name__)

Entry = tuple[str, str] | None  # (tag, text), or None = nothing to log


class MeetingLog:
    """Thread-safe append-only session log; a no-op until open() succeeds."""

    def __init__(self, directory: str = "logs"):
        # An absolute `directory` wins over PROJECT_ROOT (pathlib semantics).
        self._dir = PROJECT_ROOT / directory if directory else None
        self._file = None
        self._t0 = 0.0
        self._lock = threading.Lock()
        self.path: Path | None = None
        self.error: str | None = None

    def open(self, header: str) -> Path | None:
        """Start a new file for this session. Returns its path, or None when
        disabled or unwritable (then `error` says why)."""
        self.close()
        if self._dir is None:
            return None
        now = datetime.now()
        path = self._dir / f"meeting-{now:%Y%m%d-%H%M%S}.log"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            f = open(path, "a", encoding="utf-8")
            f.write(f"# live-interpreter {now:%Y-%m-%d %H:%M:%S}  {header}\n")
            f.flush()
        except OSError as e:
            self.error = str(e)
            log.warning("meeting log disabled: %s", e)
            return None
        with self._lock:
            self._file, self._t0, self.path, self.error = f, time.monotonic(), path, None
        return path

    def write(self, tag: str, text: str) -> None:
        with self._lock:
            if self._file is None:
                return
            elapsed = int(time.monotonic() - self._t0)
            line = (f"[{datetime.now():%H:%M:%S} +{elapsed // 60:02d}:{elapsed % 60:02d}] "
                    f"{tag}\t{' '.join(text.split())}\n")
            try:
                self._file.write(line)
                self._file.flush()
            except OSError as e:
                log.warning("meeting log write failed: %s", e)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def tee(self, callback: Callable, fmt: Callable[..., Entry]) -> Callable:
        """Wrap a pipeline callback: log fmt(*args) when it returns an entry,
        then forward the call unchanged."""

        def wrapped(*args):
            entry = fmt(*args)
            if entry is not None:
                self.write(*entry)
            return callback(*args)

        return wrapped


# -- formatters, one per pipeline callback signature ------------------------

def fmt_final(text: str, lang: str) -> Entry:
    return ("对方", text)


def fmt_mic_final(text: str, lang: str) -> Entry:
    return ("我", text)


def _translation(tag: str, translation: str, latency: float, corrected: str, raw: str) -> Entry:
    note = f"{latency:.1f}s"
    if corrected and corrected != raw:
        note += f" | 纠错: {corrected}"
    return (tag, f"{translation}  ({note})")


def fmt_translation(translation: str, lang: str, latency: float, corrected: str, raw: str) -> Entry:
    return _translation("对方译", translation, latency, corrected, raw)


def fmt_mic_translation(translation: str, lang: str, latency: float, corrected: str, raw: str) -> Entry:
    return _translation("我译", translation, latency, corrected, raw)


def fmt_assist(analysis: str, latency: float, is_final: bool) -> Entry:
    return ("助手", analysis) if is_final else None


def fmt_status(message: str) -> Entry:
    return ("状态", message)
