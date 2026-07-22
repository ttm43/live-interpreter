"""Compare translation models (en->zh) side by side via Ollama.

Usage: .venv\\Scripts\\python bench_translate.py [model1] [model2] ...
Defaults to comparing qwen3:8b against any installed hunyuan/HY-MT model.
"""
import dataclasses
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from interpreter.config import TranslatorConfig
from interpreter.translator import OllamaTranslator

# Mix of casual speech, tech jargon / new terms, idioms, and ASR-style
# unpunctuated run-on text — the real diet of this pipeline.
SENTENCES = [
    "the new model uses speculative decoding and kv cache quantization to "
    "cut inference latency by forty percent",
    "they raised a two hundred million series b at a four billion valuation "
    "which honestly feels a bit frothy",
    "if you look at the attention heads in the transformer you can see they "
    "specialize in different syntactic patterns",
    "we will ship the mvp first then iterate based on user feedback instead "
    "of boiling the ocean",
    # New-term stress tests: product names the base models may not know.
    "anthropic just shipped claude with a bigger context window and it "
    "absolutely crushes the benchmarks",
    "i asked claude to refactor the code and copilot to review it and "
    "honestly claude nailed it",
    "openai's sora can generate a full minute of video from a single prompt",
    # Noisy ASR-style input (verbatim from our streaming ASR on real clips):
    # unpunctuated, misrecognized words, truncations — the pipeline's real diet.
    "well this is him michailo federoff who has been sacked as defense "
    "minister by president selenski in his latest reshuffle",
    "there's a lot of elements about that interface that makes it increly "
    "pleasant that's fundamental to the experience of using a sparphone",
    "anthropic just shipped clod with a bigger context window and it "
    "absolutely crushes the benchmarks",
]


def installed_models() -> list[str]:
    r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
    return [m["name"] for m in r.json().get("models", [])]


def main() -> None:
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        available = installed_models()
        models = sorted(m for m in available if "qwen3" in m)
        models += sorted(m for m in available if "hunyuan" in m.lower() or "hy-mt" in m.lower())
    if not models:
        print("no models to compare; pass names as arguments")
        return

    # Model-major order: avoids swapping models in and out of VRAM per sentence.
    results: dict[str, list[tuple[str, float]]] = {}
    for name in models:
        tr = OllamaTranslator(
            dataclasses.replace(TranslatorConfig(), model=name, timeout_s=180)
        )
        results[name] = []
        for sentence in SENTENCES:
            t0 = time.monotonic()
            try:
                out = tr.translate(sentence, target_lang="zh")
            except Exception as e:  # noqa: BLE001
                out = f"<failed: {e}>"
            results[name].append((out, time.monotonic() - t0))

    for i, sentence in enumerate(SENTENCES):
        print(f"\n[en] {sentence}")
        for name in models:
            out, dt = results[name][i]
            print(f"  {name:<44} ({dt:.1f}s) {out}")


if __name__ == "__main__":
    main()
