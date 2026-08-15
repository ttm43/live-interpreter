"""Omni audio-LLM quality probe via transformers (reference implementation).

Same three modes as bench_omni.py (direct / refine / transcribe) but running
Qwen2.5-Omni through PyTorch — the path where its audio ability is intact,
unlike the degraded llama.cpp GGUF port. Text output only (talker disabled).

Usage: .venv\\Scripts\\python bench_omni_torch.py [clip names...]
"""
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", r"C:\Users\46025\work\shared\hf-cache")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_asr import REFERENCES, wer
from interpreter.config import TranslatorConfig
from interpreter.translator import OllamaTranslator

MODEL_ID = os.environ.get("OMNI_MODEL", "Qwen/Qwen2.5-Omni-3B")
ROOT = Path(__file__).resolve().parent
CLIPS = [ROOT / "testclips" / n for n in
         ("news.wav", "accent.wav", "keynote.wav", "podcast.wav")]
SLICE_S = 15.0
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
          "Group, capable of perceiving auditory and visual inputs, as well as "
          "generating text and speech.")


def read_slices(path: Path) -> tuple[list[np.ndarray], int]:
    with wave.open(str(path)) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        samples = data.astype(np.float32) / 32768.0
        rate = f.getframerate()
    n = int(SLICE_S * rate)
    return [samples[i:i + n] for i in range(0, len(samples), n)], rate


def main() -> None:
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    t0 = time.monotonic()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    print(f"{MODEL_ID} loaded in {time.monotonic() - t0:.0f}s "
          f"(VRAM {torch.cuda.memory_allocated() / 1e9:.1f}GB)")

    def ask(audio: np.ndarray, rate: int, prompt: str) -> tuple[str, float]:
        t0 = time.monotonic()
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text, audio=[audio], sampling_rate=rate,
            return_tensors="pt", padding=True,
        ).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=400, do_sample=False,
            return_audio=False,
        )
        reply = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0].strip()
        return reply, time.monotonic() - t0

    names = sys.argv[1:] or None
    translator = OllamaTranslator(TranslatorConfig())
    for clip in CLIPS:
        if names is not None and clip.name not in names:
            continue
        slices, rate = read_slices(clip)
        print(f"\n==== {clip.name} ({len(slices)} x {SLICE_S:.0f}s slices) ====")
        transcripts = []
        for i, sl in enumerate(slices):
            asr_text, t_asr = ask(sl, rate, "Transcribe this English audio exactly. Output only the transcript.")
            transcripts.append(asr_text)
            direct, t_direct = ask(sl, rate, "把这段英语语音的内容翻译成中文。只输出中文译文，不要任何解释。")
            draft = translator.translate(asr_text, target_lang="zh")
            refine, t_refine = ask(
                sl, rate,
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


if __name__ == "__main__":
    main()
