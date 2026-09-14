"""User-maintained Q&A bank: prepared answers matched against live questions.

File format (qa_bank.md in the repo root — personal content, gitignored):

    ### 5. When AI got it wrong
    **问法：** Tell me about a time AI got something wrong. / Give me an example...

    <prepared answer, any number of paragraphs>

    **↳ Another example? / Tell me about a difficult bug.**
    <prepared answer for that follow-up>

Each `### N. Title` block becomes a matchable entry (id "5"), and each
`**↳ ...**` follow-up inside it becomes its own entry (id "5.1", "5.2", ...).
Anything outside `###` blocks is ignored, so headers and notes are fine.

Hot-reloaded on mtime change, same pattern as the glossary and background.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

QA_BANK_PATH = Path(__file__).resolve().parent.parent / "qa_bank.md"

_GROUP_RE = re.compile(r"^###\s*(\d+)\W*\s*(.+?)\s*$")
_ASK_RE = re.compile(r"^\*\*问法[：:]\*\*\s*(.*)$")
_FOLLOWUP_RE = re.compile(r"^\*\*↳\s*(.+?)\*\*\s*$")
_INDEX_LINE_MAX = 160  # keep the matcher prompt bounded


@dataclass(frozen=True)
class QaEntry:
    id: str            # "5" for a group, "5.1" for its first follow-up
    group_id: str
    title: str         # group title; follow-ups carry the group title too
    questions: tuple[str, ...]
    answer: str
    is_followup: bool = False


@dataclass
class _Block:
    """Mutable accumulator while parsing one entry."""
    questions: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


def _split_questions(raw: str) -> list[str]:
    return [q.strip() for q in raw.split(" / ") if q.strip()]


def _parse(text: str) -> list[QaEntry]:
    entries: list[QaEntry] = []
    group_id = ""
    title = ""
    followup_n = 0
    current: _Block | None = None
    current_id = ""
    current_is_followup = False

    def flush() -> None:
        nonlocal current
        if current is None or not current.questions:
            current = None
            return
        answer = "\n".join(current.lines).strip()
        if answer:
            entries.append(QaEntry(
                id=current_id,
                group_id=group_id,
                title=title,
                questions=tuple(current.questions),
                answer=answer,
                is_followup=current_is_followup,
            ))
        current = None

    for line in text.splitlines():
        stripped = line.strip()
        m = _GROUP_RE.match(stripped)
        if m:
            flush()
            group_id, title = m.group(1), m.group(2)
            followup_n = 0
            current = _Block()
            current_id = group_id
            current_is_followup = False
            continue
        if not group_id:
            continue  # preamble before the first ### block
        if stripped == "---":
            flush()
            continue
        m = _ASK_RE.match(stripped)
        if m and current is not None and not current.questions:
            current.questions = _split_questions(m.group(1))
            continue
        m = _FOLLOWUP_RE.match(stripped)
        if m:
            flush()
            followup_n += 1
            current = _Block(questions=_split_questions(m.group(1)))
            current_id = f"{group_id}.{followup_n}"
            current_is_followup = True
            continue
        if current is not None:
            current.lines.append(line.rstrip())
    flush()
    return entries


class QaBank:
    def __init__(self, path: Path = QA_BANK_PATH):
        self._path = path
        self._mtime = -1.0
        self._entries: dict[str, QaEntry] = {}
        self._index_text = ""

    def _reload_if_changed(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._entries = {}
            self._index_text = ""
            return
        if mtime == self._mtime:
            return
        entries = _parse(self._path.read_text(encoding="utf-8"))
        self._entries = {e.id: e for e in entries}
        lines = []
        for e in entries:
            head = e.title if not e.is_followup else f"{e.title} ↳追问"
            body = " / ".join(e.questions)
            lines.append(f"{e.id} | {head} | {body}"[:_INDEX_LINE_MAX])
        self._index_text = "\n".join(lines)
        self._mtime = mtime

    def index_text(self) -> str:
        """Compact numbered index for the matcher prompt ('' = no bank)."""
        self._reload_if_changed()
        return self._index_text

    def get(self, entry_id: str) -> QaEntry | None:
        self._reload_if_changed()
        return self._entries.get(entry_id)
