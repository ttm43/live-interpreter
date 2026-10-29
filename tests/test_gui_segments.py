"""TranscriptPane attaches translations by segment_id: a translation that
arrives after the NEXT final (same text or not) still lands under its own
source line, and a corrected source replaces the right line."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

tk = pytest.importorskip("tkinter")


@pytest.fixture
def pane():
    try:
        root = tk.Tk()
    except tk.TclError as e:  # no display
        pytest.skip(f"Tk unavailable: {e}")
    root.withdraw()
    from gui import TranscriptPane

    p = TranscriptPane(root, "test")
    yield p
    root.destroy()


def body(pane) -> str:
    return pane.text.get("1.0", "end-1c")


def test_late_translation_lands_under_its_own_line(pane):
    pane.final("hello", "en", 1)
    pane.final("hello", "en", 2)           # identical text, next segment
    pane.partial("and now")                # live block below both
    pane.translation("你好二", "zh", 1.0, "hello", "hello", seg_id=2)
    pane.translation("你好一", "zh", 2.5, "hello", "hello", seg_id=1)
    assert body(pane) == (
        "[en] hello\n      → 你好一 (2.5s)\n\n"
        "[en] hello\n      → 你好二 (1.0s)\n\n"
        "… and now"
    )


def test_correction_promotes_the_right_source_line(pane):
    pane.final("for the swims I would", "en", 1)
    pane.final("next question", "en", 2)
    pane.translation("对于 SWMS 我会", "zh", 1.0, "for the SWMS I would", "for the swims I would", seg_id=1)
    assert body(pane).startswith(
        "[en] for the SWMS I would\n      原: for the swims I would\n      → 对于 SWMS 我会 (1.0s)\n\n"
        "[en] next question\n"
    )


def test_correction_after_previous_late_translation(pane):
    # Segment 2's start mark sits where segment 1's late translation is
    # inserted; the correction of 2 must still rewrite 2's own line.
    pane.final("one", "en", 1)
    pane.final("twoo", "en", 2)
    pane.translation("一", "zh", 1.0, "one", "one", seg_id=1)
    pane.translation("二", "zh", 1.0, "two", "twoo", seg_id=2)
    assert body(pane) == (
        "[en] one\n      → 一 (1.0s)\n\n"
        "[en] two\n      原: twoo\n      → 二 (1.0s)\n\n"
    )


def test_unknown_segment_appends_at_end(pane):
    pane.final("a", "en", 1)
    pane.translation("乙", "zh", 1.0, "b", "b", seg_id=99)
    assert body(pane) == "[en] a\n      → 乙 (1.0s)\n\n"


def test_clear_forgets_segments(pane):
    pane.final("a", "en", 1)
    pane.clear()
    pane.translation("甲", "zh", 1.0, "a", "a", seg_id=1)
    assert body(pane) == "      → 甲 (1.0s)\n\n"
