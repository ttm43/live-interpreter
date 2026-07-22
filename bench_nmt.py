"""Test old-generation dedicated NMT models (opus-mt, NLLB) on the same
sentences used in bench_translate.py, for a fair three-way comparison."""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_translate import SENTENCES


def bench_opus_mt() -> None:
    from transformers import MarianMTModel, MarianTokenizer

    t0 = time.monotonic()
    name = "Helsinki-NLP/opus-mt-en-zh"
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    print(f"== opus-mt-en-zh (~310MB, loaded in {time.monotonic() - t0:.0f}s)")
    for s in SENTENCES:
        t0 = time.monotonic()
        batch = tok([s], return_tensors="pt")
        out = model.generate(**batch, max_new_tokens=256)
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"  ({time.monotonic() - t0:.1f}s) {text}")


def bench_nllb() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    t0 = time.monotonic()
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    print(f"\n== nllb-200-distilled-600M (~2.4GB, loaded in {time.monotonic() - t0:.0f}s)")
    zho = tok.convert_tokens_to_ids("zho_Hans")
    for s in SENTENCES:
        t0 = time.monotonic()
        batch = tok(s, return_tensors="pt")
        out = model.generate(**batch, forced_bos_token_id=zho, max_new_tokens=256)
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"  ({time.monotonic() - t0:.1f}s) {text}")


if __name__ == "__main__":
    for s in SENTENCES:
        print(f"[en] {s}")
    print()
    bench_opus_mt()
    bench_nllb()
