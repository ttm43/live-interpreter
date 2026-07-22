"""Offline self-test: ASR on bundled test wavs, TTS synthesis for zh/en."""
import sys
import time
import wave

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from interpreter.asr import StreamingAsr
from interpreter.config import AppConfig
from interpreter.tts_engine import BilingualTts

cfg = AppConfig()


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0, f.getframerate()


def test_asr() -> None:
    print("== ASR test ==")
    t0 = time.monotonic()
    asr = StreamingAsr(cfg.asr)
    print(f"ASR model loaded in {time.monotonic() - t0:.1f}s")
    from interpreter.config import ASR_MODEL_DIR
    for wav in sorted((ASR_MODEL_DIR / "test_wavs").glob("*.wav"))[:3]:
        samples, sr = read_wav(str(wav))
        t0 = time.monotonic()
        finals = []
        chunk = int(sr * 0.1)
        for i in range(0, len(samples), chunk):
            for ev in asr.accept(samples[i:i + chunk], sr):
                if ev.is_final:
                    finals.append(ev.text)
        # flush with 1s of trailing silence to trigger endpoint
        for _ in range(15):
            for ev in asr.accept(np.zeros(chunk, dtype=np.float32), sr):
                if ev.is_final:
                    finals.append(ev.text)
        dur = len(samples) / sr
        elapsed = time.monotonic() - t0
        print(f"{wav.name} ({dur:.1f}s audio, decoded in {elapsed:.1f}s, RTF {elapsed/dur:.2f})")
        for t in finals:
            print(f"   -> {t}")


def test_tts() -> None:
    print("\n== TTS test ==")
    t0 = time.monotonic()
    tts = BilingualTts(cfg.tts)
    print(f"TTS model loaded in {time.monotonic() - t0:.1f}s")
    for text, lang in [
        ("今天的会议主要讨论第三季度的产品发布计划。", "zh"),
        ("The quarterly revenue grew by fifteen percent year over year.", "en"),
    ]:
        t0 = time.monotonic()
        samples, sr = tts.synthesize(text, lang)
        elapsed = time.monotonic() - t0
        dur = len(samples) / sr
        print(f"[{lang}] {dur:.1f}s audio in {elapsed:.1f}s (RTF {elapsed/dur:.2f}): {text}")
        out = f"selftest_tts_{lang}.wav"
        with wave.open(out, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes((samples * 32767).astype(np.int16).tobytes())
        print(f"   saved {out}")
    tts.close()


if __name__ == "__main__":
    test_asr()
    test_tts()
