"""Benchmark candidate English streaming ASR models.

For each model x test wav: transcript, RTF, and partial-update cadence
(how often the live caption refreshes — the "word-by-word feel" metric).
Usage: .venv\\Scripts\\python bench_asr.py [extra.wav ...]
"""
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from interpreter.asr import create_asr
from interpreter.config import ASR_EN_MODEL_DIR, EN_ASR_MODELS

CLIPS_DIR = Path(__file__).resolve().parent / "testclips"

REFERENCES = {
    "0.wav": "after early nightfall the yellow lamps would light up here and "
             "there the squalid quarter of the brothels",
    "1.wav": "god as a direct consequence of the sin which man thus punished "
             "had given her a lovely child whose place was on that same "
             "dishonoured bosom to connect her parent for ever with the race "
             "and descent of mortals and to be finally a blessed soul in heaven",
}
_refs_file = CLIPS_DIR / "refs.json"
if _refs_file.exists():
    REFERENCES.update(json.loads(_refs_file.read_text(encoding="utf-8")))


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so styles don't count as errors."""
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0, f.getframerate()


def wer(hyp: str, ref: str) -> float:
    h, r = normalize(hyp).split(), normalize(ref).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(r), len(h)]) / max(1, len(r))


def bench(model_name: str, wav_paths: list[str]) -> None:
    t0 = time.monotonic()
    try:
        asr = create_asr(EN_ASR_MODELS[model_name])
    except Exception as e:  # noqa: BLE001
        print(f"== {model_name}: LOAD FAILED: {e}")
        return
    print(f"== {model_name} (loaded in {time.monotonic() - t0:.1f}s)")

    for path in wav_paths:
        samples, sr = read_wav(path)
        chunk = int(sr * 0.1)
        finals, updates = [], []
        fed_s = 0.0
        t0 = time.monotonic()
        for i in range(0, len(samples), chunk):
            fed_s += len(samples[i:i + chunk]) / sr
            for ev in asr.accept(samples[i:i + chunk], sr):
                if ev.is_final:
                    finals.append(ev.text)
                else:
                    updates.append(fed_s)
        for _ in range(25):
            for ev in asr.accept(np.zeros(chunk, dtype=np.float32), sr):
                if ev.is_final:
                    finals.append(ev.text)
        elapsed = time.monotonic() - t0
        dur = len(samples) / sr
        hyp = " ".join(finals)
        name = path.split("\\")[-1].split("/")[-1]
        gaps = np.diff(updates) if len(updates) > 1 else [0]
        line = (f"  {name}: RTF {elapsed / dur:.2f} | partial updates "
                f"{len(updates)} (avg gap {np.mean(gaps) * 1000:.0f}ms)")
        if name in REFERENCES:
            line += f" | WER {wer(hyp, REFERENCES[name]) * 100:.1f}%"
        print(line)
        print(f"    -> {hyp}")


def main() -> None:
    wavs = [str(ASR_EN_MODEL_DIR / "test_wavs" / n) for n in ("0.wav", "1.wav")]
    wavs += [str(p) for p in sorted(CLIPS_DIR.glob("*.wav"))]
    wavs += sys.argv[1:]
    for model_name in EN_ASR_MODELS:
        bench(model_name, wavs)


if __name__ == "__main__":
    main()
