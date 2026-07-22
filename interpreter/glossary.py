"""User-maintained glossary: forces terminology in translations.

File format (glossary.txt, one entry per line, '#' for comments):
    Claude = Claude
    attention head = 注意力头
Edits take effect on the next segment — the file is reloaded when its
modification time changes, no restart needed.
"""
from pathlib import Path

GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "glossary.txt"


class Glossary:
    def __init__(self, path: Path = GLOSSARY_PATH):
        self._path = path
        self._mtime = -1.0
        self._entries: dict[str, str] = {}

    def _reload_if_changed(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._entries = {}
            return
        if mtime == self._mtime:
            return
        entries: dict[str, str] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            src, dst = line.split("=", 1)
            if src.strip() and dst.strip():
                entries[src.strip()] = dst.strip()
        self._entries = entries
        self._mtime = mtime

    def matches(self, text: str) -> dict[str, str]:
        """Entries whose source term appears in the text (case-insensitive)."""
        self._reload_if_changed()
        lowered = text.lower()
        return {s: d for s, d in self._entries.items() if s.lower() in lowered}
