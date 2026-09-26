"""Unit tests for the Confucius4-R2T2 engine: prefill parsing, stable-prefix
commit logic and the append-only streaming behaviour, with the llama-server
transport replaced by a fake. Run: .venv\\Scripts\\python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interpreter import asr_confucius as ac  # noqa: E402
from interpreter.config import AsrConfig  # noqa: E402

PREFILL = "language English<asr_text>"


def test_strip_prefill_echoed_and_not_echoed():
    assert ac.strip_prefill(PREFILL + "Hello there.", PREFILL) == "Hello there."
    assert ac.strip_prefill("language English<asr_text>Hello there.", PREFILL + "Hello") == " there."
    # server did not echo the prefill: only the continuation came back
    assert ac.strip_prefill(" there.", PREFILL + "Hello") == " there."


@pytest.mark.parametrize("text,expected", [
    ("God as a direct consequence", "God as a direct"),
    (" as Defence Minister.", " as Defence"),   # edge word + spurious period rolled back
    ("ed the sin", "ed the"),                   # finishes the committed word: no leading space
    ("Hello there.", "Hello"),
    ("single", ""),
    ("   ", ""),
])
def test_commit_stable_rolls_back_the_edge_word(text, expected):
    assert ac.commit_stable(text) == expected


def test_join_normalises_whitespace_and_mid_word_continuations():
    assert ac._join("after", " early nightfall") == "after early nightfall"
    assert ac._join("punish", "ed the sin") == "punished the sin"
    assert ac._join("", "  Hello  ") == "Hello"


def test_to_model_rate_resamples_48k_to_16k():
    x = np.zeros(48000, dtype=np.float32)
    assert len(ac._to_model_rate(x, 48000)) == 16000
    assert ac._to_model_rate(x, 16000) is x


class FakeServer:
    """Returns the transcript progressively: each hop reveals a bit more audio."""

    WORDS = "after early nightfall the yellow lamps would light up".split()

    def __init__(self):
        self.calls: list[tuple[str, int, float]] = []

    def __call__(self, audio, rate, prefill, max_tokens):
        secs = len(audio) / rate
        self.calls.append((prefill, max_tokens, secs))
        committed = prefill[len(PREFILL):]
        heard = " ".join(self.WORDS[: min(len(self.WORDS), int(secs * 2))])  # 2 words / second
        assert heard.startswith(committed.strip()), "prefill must be a prefix of what is heard"
        return prefill + heard[len(committed.strip()):]


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(ac.ConfuciusAsr, "_load", lambda self: None)
    cfg = AsrConfig(kind="confucius", language="English", rule2_min_trailing_silence=0.9)
    eng = ac.ConfuciusAsr(cfg)
    eng._prefill = PREFILL
    eng._committed = ""
    fake = FakeServer()
    monkeypatch.setattr(eng, "_request", fake)
    return eng, fake


def feed(eng, seconds: float, loud: bool, rate: int = 16000):
    chunk = int(rate * 0.1)
    events = []
    amp = 0.3 if loud else 0.0
    for _ in range(int(seconds / 0.1)):
        samples = (np.random.default_rng(0).standard_normal(chunk) * amp).astype(np.float32)
        events += eng.accept(samples, rate)
    return events


def test_partials_are_append_only_and_final_completes(engine):
    eng, fake = engine
    events = feed(eng, 4.0, loud=True)
    partials = [e.text for e in events if not e.is_final]
    assert partials, "streaming hops should have produced partials"
    for a, b in zip(partials, partials[1:]):
        assert b.startswith(a), f"partial revised instead of appended: {a!r} -> {b!r}"
    events += feed(eng, 1.5, loud=False)  # trailing silence -> endpoint
    finals = [e for e in events if e.is_final]
    assert len(finals) == 1
    assert finals[0].text.startswith(partials[-1])
    assert eng._committed == ""  # state cleared for the next utterance
    # streaming hops prefilled with the committed text and a small budget;
    # the final is a fresh decode: bare prefill, large budget
    hops = [c for c in fake.calls if c[1] == ac.HOP_MAX_TOKENS]
    assert hops and all(c[0].startswith(PREFILL) for c in hops)
    final_calls = [c for c in fake.calls if c[1] == ac.FINAL_MAX_TOKENS]
    assert final_calls == [(PREFILL, ac.FINAL_MAX_TOKENS, final_calls[0][2])]
