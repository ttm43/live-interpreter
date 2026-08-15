"""Omni audio-LLM experiments: audio slices -> multimodal LLM, via llama-server.

Modes per slice of each test clip:
  direct   — one-step speech translation ("translate this audio to Chinese")
  refine   — audio + cascade draft translation -> corrected translation
  transcribe — plain ASR, scored against references (hearing check)

Usage: .venv\\Scripts\\python bench_omni.py [clip names...]
Requires llama-server + GGUF under <workspace>\\shared (see OMNI_* constants).
"""
import base64
import io
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_asr import REFERENCES, wer
from interpreter.config import TranslatorConfig
from interpreter.translator import OllamaTranslator

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / "shared"
SERVER = SHARED / "llama.cpp" / "llama-server.exe"
MODEL = SHARED / "gguf" / "Qwen2.5-Omni-7B-Q4_K_M.gguf"
MMPROJ = SHARED / "gguf" / "mmproj-Qwen2.5-Omni-7B-f16.gguf"
PORT = 8090
SLICE_S = 15.0

CLIPS = [ROOT / "testclips" / n for n in
         ("news.wav", "accent.wav", "keynote.wav", "podcast.wav")]


def wav_b64(samples: np.ndarray, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def ask(audio_b64: str, prompt: str, max_tokens: int = 400) -> tuple[str, float]:
    t0 = time.monotonic()
    r = requests.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        timeout=300,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"].strip(),
            time.monotonic() - t0)


def ensure_server() -> subprocess.Popen | None:
    try:
        if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
            return None
    except requests.RequestException:
        pass
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--mmproj", str(MMPROJ),
         "-ngl", "99", "-c", "8192", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        time.sleep(1)
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                return proc
        except requests.RequestException:
            continue
    raise SystemExit("llama-server did not become healthy")


def read_slices(path: Path) -> tuple[list[np.ndarray], int]:
    with wave.open(str(path)) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        samples = data.astype(np.float32) / 32768.0
        rate = f.getframerate()
    n = int(SLICE_S * rate)
    return [samples[i:i + n] for i in range(0, len(samples), n)], rate


def main() -> None:
    names = sys.argv[1:] or None
    clips = [c for c in CLIPS if names is None or c.name in names]
    proc = ensure_server()
    translator = OllamaTranslator(TranslatorConfig())
    try:
        for clip in clips:
            slices, rate = read_slices(clip)
            print(f"\n==== {clip.name} ({len(slices)} x {SLICE_S:.0f}s slices) ====")
            transcripts = []
            for i, sl in enumerate(slices):
                b64 = wav_b64(sl, rate)

                asr_text, t_asr = ask(b64, "Transcribe this English audio exactly. Output only the transcript.")
                transcripts.append(asr_text)

                direct, t_direct = ask(b64, "把这段英语语音的内容翻译成中文。只输出中文译文，不要任何解释。")

                draft = translator.translate(asr_text, target_lang="zh")
                refine, t_refine = ask(
                    b64,
                    "下面是这段英语语音的机器翻译草稿：\n"
                    f"「{draft}」\n"
                    "请对照语音内容修正草稿中的错误（尤其是人名、数字、术语），"
                    "只输出修正后的中文译文，不要解释。",
                )
                print(f"  [slice {i}] ASR {t_asr:.1f}s | direct {t_direct:.1f}s | refine {t_refine:.1f}s")
                print(f"    transcribe: {asr_text[:150]}")
                print(f"    direct    : {direct[:150]}")
                print(f"    draft     : {draft[:150]}")
                print(f"    refine    : {refine[:150]}")
            joined = " ".join(transcripts)
            if clip.name in REFERENCES:
                print(f"  transcribe WER vs ref: {wer(joined, REFERENCES[clip.name]) * 100:.1f}%")
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
