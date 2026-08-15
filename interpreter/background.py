"""User-maintained meeting background: free-form context for the assistant.

Write anything that helps the assistant read the room into background.txt —
meeting agenda, participants and their roles, project state, your own goals
for this meeting. Lines starting with '#' are comments.

Edits take effect on the next analyzed segment — the file is reloaded when
its modification time changes, no restart needed (same pattern as glossary).
"""
from pathlib import Path

BACKGROUND_PATH = Path(__file__).resolve().parent.parent / "background.txt"

MAX_CHARS = 6000  # keep the prompt bounded even if the user pastes a novel


class Background:
    def __init__(self, path: Path = BACKGROUND_PATH, max_chars: int = MAX_CHARS):
        self._path = path
        self._max_chars = max_chars
        self._mtime = -1.0
        self._text = ""

    def _reload_if_changed(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._text = ""
            return
        if mtime == self._mtime:
            return
        lines = [
            line.rstrip()
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        ]
        text = "\n".join(lines).strip()
        if len(text) > self._max_chars:
            text = text[: self._max_chars]
        self._text = text
        self._mtime = mtime

    def text(self) -> str:
        """Current background text ('' when the file is absent or empty)."""
        self._reload_if_changed()
        return self._text
