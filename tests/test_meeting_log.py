"""Unit tests for the per-session meeting log: file format, disabled and
unwritable cases, the callback tee and the per-callback formatters."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interpreter import meeting_log as ml  # noqa: E402

LINE = re.compile(r"^\[\d\d:\d\d:\d\d \+\d\d:\d\d\] (\S+)\t(.*)$")


def test_open_write_close(tmp_path):
    mlog = ml.MeetingLog(str(tmp_path))
    path = mlog.open("finals=confucius")
    assert path is not None and path.parent == tmp_path and path.name.startswith("meeting-")
    mlog.write("对方", "  hello\n  world ")
    mlog.write("助手", "【意图】a\n\n【无需回应】b")
    mlog.close()
    mlog.write("对方", "after close")  # no-op, no error
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# live-interpreter ") and lines[0].endswith("finals=confucius")
    assert [LINE.match(ln).groups() for ln in lines[1:]] == [
        ("对方", "hello world"),
        ("助手", "【意图】a 【无需回应】b"),
    ]


def test_disabled_and_unwritable(tmp_path):
    off = ml.MeetingLog("")
    assert off.open("x") is None and off.error is None
    off.write("对方", "x")  # no-op
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    bad = ml.MeetingLog(str(blocked))
    assert bad.open("x") is None
    assert bad.error


def test_tee_forwards_and_logs(tmp_path):
    mlog = ml.MeetingLog(str(tmp_path))
    path = mlog.open("h")
    calls = []
    cb = mlog.tee(lambda *a: calls.append(a) or "ret", ml.fmt_assist)
    assert cb("speculative", 0.1, False) == "ret"  # forwarded, not logged
    assert cb("final analysis", 1.2, True) == "ret"
    mlog.close()
    assert calls == [("speculative", 0.1, False), ("final analysis", 1.2, True)]
    body = path.read_text(encoding="utf-8")
    assert "speculative" not in body and "助手\tfinal analysis" in body


@pytest.mark.parametrize("fmt,args,expected", [
    (ml.fmt_final, ("Hello there.", "en"), ("对方", "Hello there.")),
    (ml.fmt_mic_final, ("I agree.", "en"), ("我", "I agree.")),
    (ml.fmt_translation, ("你好。", "en", 1.04, "Hello there.", "Hello there."), ("对方译", "你好。  (1.0s)")),
    (ml.fmt_mic_translation, ("我同意。", "en", 0.5, "I agree.", "I agreed"),
     ("我译", "我同意。  (0.5s | 纠错: I agree.)")),
    (ml.fmt_assist, ("x", 0.0, False), None),
    (ml.fmt_status, ("已停止",), ("状态", "已停止")),
])
def test_formatters(fmt, args, expected):
    assert fmt(*args) == expected
