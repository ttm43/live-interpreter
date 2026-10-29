"""Regression tests for the real-time policies of the pipeline: capture queue
overflow and seq gaps, ASR reset across a gap, TTS newest-wins + expiry, mic
mute timing, segment ids, dialogue order counting and session isolation.
Everything runs on fakes: no audio device, no model, no Ollama."""
import dataclasses
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interpreter import pipeline as pl  # noqa: E402
from interpreter.asr import AsrEvent  # noqa: E402
from interpreter.audio_capture import AudioChunk, ChunkQueue  # noqa: E402
from interpreter.config import AppConfig, TranslatorConfig  # noqa: E402
from interpreter.dialogue import ME, OTHER, DialogueInterpreter  # noqa: E402

CFG = AppConfig(meeting_log_dir="", enable_assistant=False)


# -- capture queue ------------------------------------------------------------

def test_chunk_queue_drops_oldest_and_numbers_chunks():
    q = ChunkQueue(chunk_s=0.1, queue_s=0.3)  # holds 3 chunks
    for i in range(5):
        q.push(np.full(4, i, dtype=np.float32))
    got = [q.pop(0.01) for _ in range(4)]
    assert got[-1] is None
    assert [c.seq for c in got[:3]] == [2, 3, 4]          # newest survive
    assert [int(c.samples[0]) for c in got[:3]] == [2, 3, 4]
    assert q.dropped_chunks == 2 and q.dropped_s == pytest.approx(0.2)


def test_seq_tracker_reports_hole_in_seconds():
    t = pl.SeqTracker(chunk_s=0.1)
    mk = lambda s: AudioChunk(s, 0.0, np.zeros(1, dtype=np.float32))  # noqa: E731
    assert t.gap_s(mk(7)) == 0.0        # first chunk: no history
    assert t.gap_s(mk(8)) == 0.0
    assert t.gap_s(mk(11)) == pytest.approx(0.2)  # 9 and 10 are missing
    assert t.gap_s(mk(12)) == 0.0


def test_put_newest_drops_oldest():
    q, dropped = queue.Queue(maxsize=2), []
    for item in "abc":
        pl.put_newest(q, item, lambda: dropped.append(1))
    assert [q.get_nowait(), q.get_nowait()] == ["b", "c"] and dropped == [1]


# -- ASR lane: a seq gap resets both engines, finals carry segment ids --------

class FakeCapture:
    """Hands out prepared chunks, then stops the session."""

    sample_rate = 16000
    chunk_s = 0.1

    def __init__(self, chunks, sess):
        self._chunks, self._sess = list(chunks), sess
        self.suppress = threading.Event()

    def read(self, timeout=0.2):
        if self._chunks:
            return self._chunks.pop(0)
        self._sess.stop.set()
        return None


class FakeAsr:
    """Final for every chunk of 160 samples; records the call sequence."""

    def __init__(self):
        self.calls = []
        self.n = 0

    def accept(self, samples, rate):
        self.calls.append("accept")
        if len(samples) != 160:
            return []
        self.n += 1
        return [AsrEvent(text=f"utterance {self.n}", is_final=True)]

    def reset(self):
        self.calls.append("reset")


def _chunk(seq, n=160):
    return AudioChunk(seq, 0.0, np.full(n, 0.3, dtype=np.float32))


def _lane_pipeline(**events):
    p = pl.InterpreterPipeline(CFG, **events)
    sess = pl._Session(1, threading.Event())
    p._session = sess
    return p, sess


def test_asr_lane_resets_on_gap_and_numbers_segments():
    finals, status, partials = [], [], []
    p, sess = _lane_pipeline(
        on_final=lambda t, lang, sid: finals.append((t, sid)),
        on_status=status.append,
        on_partial=partials.append,
    )
    p._asr, p._fast_asr, p._aec = FakeAsr(), FakeAsr(), None
    p._segments, p._assist_q = queue.Queue(maxsize=16), None
    p._spec_text = "half a sent"
    p._capture = FakeCapture([_chunk(0), _chunk(1), _chunk(5), _chunk(6)], sess)
    p._run(sess, p._asr_loop)

    # chunks 2-4 are missing: both engines are reset exactly once, between
    # chunk 1 and chunk 5, so nothing is spliced across the hole (the last
    # accept is the synthetic silence fed when the capture times out)
    assert p._asr.calls == ["accept", "accept", "reset", "accept", "accept", "accept"]
    assert p._fast_asr.calls == p._asr.calls
    assert p.audio_gaps == 1 and p._spec_text == ""
    assert any("断流 0.3s" in s for s in status)
    assert partials == [""]  # the stale partial on screen is cleared
    assert [sid for _, sid in finals] == [1, 2, 3, 4]
    queued = [p._segments.get_nowait() for _ in range(4)]
    assert [(sid, t) for sid, t, _ in queued] == [(s, f"utterance {s}") for s in range(1, 5)]


def test_stale_asr_worker_cannot_leave_a_partial_for_the_new_session():
    p, old = _lane_pipeline()
    p._session = pl._Session(2, threading.Event())  # a new session has started
    p._set_spec(old, "ghost of session 1")
    assert p._spec_text == ""
    p._set_spec(p._session, "live")
    assert p._spec_text == "live"


# -- translation lane: segment id travels with the translation ---------------

class FakeInterp:
    out_of_order = 0

    def interpret(self, text, speaker, t_final=None):
        return _Res(text)


@dataclasses.dataclass
class _Res:
    corrected: str
    translation: str = "译"
    target_lang: str = "zh"


def test_translation_carries_segment_id_and_feeds_tts_newest_first():
    out = []
    p, sess = _lane_pipeline(on_translation=lambda *a: out.append(a))
    p._interp = FakeInterp()
    p._segments, p._speech = queue.Queue(maxsize=16), queue.Queue(maxsize=1)
    t0 = time.monotonic()
    p._segments.put((7, "seven", t0))
    p._segments.put((8, "eight", t0))

    def worker(s):
        p._translation_worker(s)

    th = threading.Thread(target=p._run, args=(sess, worker))
    th.start()
    deadline = time.monotonic() + 3
    while len(out) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    sess.stop.set()
    th.join(2)
    assert [(a[4], a[5]) for a in out] == [("seven", 7), ("eight", 8)]
    # TTS queue of 1: the newest translation is what is left, not the oldest
    assert p._speech.get_nowait()[2] == t0 and p._speech.empty()


# -- TTS lane: expiry, mute timing ---------------------------------------------

class FakeTts:
    synth_s = 0.0
    log = []

    def __init__(self, cfg, device):
        pass

    def synthesize(self, text, lang):
        FakeTts.log.append(("synth", text, _mic.suppress.is_set(), _cap.suppress.is_set()))
        time.sleep(FakeTts.synth_s)
        return np.ones(8, dtype=np.float32), 24000

    def play(self, samples, sr):
        FakeTts.log.append(("play", _mic.suppress.is_set(), _cap.suppress.is_set()))
        return True

    def close(self):
        pass


class _Gate:
    def __init__(self):
        self.suppress = threading.Event()


_mic, _cap = _Gate(), _Gate()


def _run_tts(monkeypatch, cfg, items, synth_s=0.0):
    import interpreter.tts_engine as te

    monkeypatch.setattr(te, "BilingualTts", FakeTts)
    monkeypatch.setattr(pl, "TTS_TAIL_GUARD_S", 0.0)
    FakeTts.log, FakeTts.synth_s = [], synth_s
    _mic.suppress.clear(), _cap.suppress.clear()
    status = []
    p = pl.InterpreterPipeline(cfg, on_status=status.append)
    sess = pl._Session(1, threading.Event())
    p._session = sess
    p._speech = queue.Queue(maxsize=8)
    p._capture, p._mic, p._aec = _cap, _mic, None
    for it in items:
        p._speech.put(it)
    th = threading.Thread(target=p._run, args=(sess, p._tts_worker))
    th.start()
    deadline = time.monotonic() + 3
    while not p._speech.empty() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.1 + synth_s)
    sess.stop.set()
    th.join(2)
    return p, status


def test_tts_skips_stale_translations_and_mutes_only_while_playing(monkeypatch):
    now = time.monotonic()
    p, status = _run_tts(monkeypatch, CFG, [("old", "zh", now - 60), ("fresh", "zh", now)])
    assert p.tts_skipped == 1 and any("跳过过期朗读" in s for s in status)
    # "old" was never synthesized; "fresh" was synthesized with both gates
    # open and played with both closed, and the gates are open again after
    assert FakeTts.log == [("synth", "fresh", False, False), ("play", True, True)]
    assert not _mic.suppress.is_set() and not _cap.suppress.is_set()
    assert 0 <= p.tts_lag_s < 2


def test_tts_rechecks_expiry_after_synthesis(monkeypatch):
    cfg = dataclasses.replace(CFG, tts_max_age_s=0.15)
    p, _ = _run_tts(monkeypatch, cfg, [("slow", "zh", time.monotonic())], synth_s=0.3)
    assert p.tts_skipped == 1
    assert [e[0] for e in FakeTts.log] == ["synth"]  # synthesized, never played


def test_headset_keeps_mic_open_during_tts(monkeypatch):
    cfg = dataclasses.replace(CFG, mute_mic_during_tts=False, mute_capture_during_tts=False)
    _run_tts(monkeypatch, cfg, [("x", "zh", time.monotonic())])
    assert FakeTts.log == [("synth", "x", False, False), ("play", False, False)]


# -- sessions: a worker from a stopped session cannot reach the front-end -----

def test_stale_session_callbacks_are_dropped(monkeypatch):
    monkeypatch.setattr(pl, "STOP_JOIN_S", 0.05)
    got = []
    p = pl.InterpreterPipeline(CFG, on_status=got.append)
    sess_a = pl._Session(1, threading.Event())
    p._session, release = sess_a, threading.Event()

    def stuck_in_model_request(s):
        p._on_status("A: before")
        release.wait(5)           # the 60 s llama-server timeout, in miniature
        p._on_status("A: late")   # must not reach the front-end

    p._threads = [threading.Thread(target=p._run, args=(sess_a, stuck_in_model_request))]
    p._threads[0].start()
    time.sleep(0.05)
    p.stop()                      # join times out; the worker is abandoned
    assert "A: before" in got and any("仍在等待模型响应" in s for s in got)
    # a new session starts, then the old worker finally returns
    sess_b = pl._Session(2, threading.Event())
    p._session = sess_b
    release.set()
    time.sleep(0.1)
    assert "A: late" not in got
    p._run(sess_b, lambda s: p._on_status("B: hello"))
    assert got[-1] == "B: hello"   # the live session still gets through


# -- dialogue: out-of-order turns are counted, not reordered --------------------

def test_dialogue_counts_out_of_order_turns():
    interp = DialogueInterpreter(TranslatorConfig())
    interp._post = lambda model, messages, json_mode: '{"corrected": "fine", "translation": "好"}'
    interp.interpret("fine one", OTHER, t_final=10.0)
    interp.interpret("fine two", ME, t_final=12.0)
    assert interp.out_of_order == 0
    interp.interpret("fine zero", OTHER, t_final=9.0)  # finalized late, said first
    assert interp.out_of_order == 1
    interp.interpret("fine three", ME, t_final=13.0)
    assert interp.out_of_order == 1
    interp.interpret("no time", ME)  # callers without timing never count
    assert interp.out_of_order == 1
