"""Confucius4-R2T2 (standalone ASR, llama.cpp GGUF) vs the production lane
(Parakeet + qwen3:4b-instruct correction) on human-transcribed speech
(testclips/esb, see esb_sample.py).

Lanes (all fed the SAME AGC-conditioned audio the live pipeline would produce;
Parakeet int8 returns empty strings on some raw low-level clips, AGC masks it):
  confucius     Confucius4-R2T2 GGUF via llama-server, whole-utterance decode
                (its vLLM-only streaming mode cannot run on Windows)
  parakeet      Parakeet-TDT-0.6B-v3 int8 whole-utterance offline decode
                (model upper bound; same conditions as confucius)
  parakeet+llm  parakeet output through DialogueInterpreter (fused
                correction+translation; fresh context per utterance)
  production    the real lane: SemiStreamingAsr windowing/endpointing ->
                each final through a shared DialogueInterpreter (history kept
                within the utterance, reset between utterances)

Scores: verbatim WER (Whisper EnglishTextNormalizer on both sides) and a
"fluent" WER that additionally collapses adjacent repeats and strips standalone
fillers on both sides (verbatim references keep "or or", "you know" that a
caption engine is right to drop). Paired bootstrap 95% CIs vs confucius.

Usage: .venv\\Scripts\\python bench_confucius.py [--quant f16|Q8_0] [--limit N] [--skip-llm]
"""
import argparse
import base64
import io
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests
from whisper_normalizer.english import EnglishTextNormalizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from interpreter.asr import create_asr
from interpreter.audio_capture import AutoGain
from interpreter.config import EN_ASR_MODELS, TranslatorConfig

sys.path.insert(0, r"C:\Users\46025\work\models")
from model_registry import path_of  # noqa: E402

ROOT = Path(__file__).resolve().parent
ESB = ROOT / "testclips" / "esb"
PORT = 8091
LLAMA_SERVER = Path(r"C:\Users\46025\work\shared\llama.cpp-b11193\llama-server.exe")
NORM = EnglishTextNormalizer()
FILLERS = {"ah", "oh", "yeah", "hmm", "mm", "mhm", "um", "uh", "er", "erm"}
FILLER_PHRASES = [("you", "know"), ("i", "mean")]
LANES = ["confucius", "parakeet", "parakeet_llm", "production"]
PARAKEET_KEY = "parakeet-semi"  # overridden by --parakeet-engine


# ---- audio ---------------------------------------------------------------------------------

def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0, f.getframerate()


def agc_condition(x: np.ndarray, sr: int) -> np.ndarray:
    """Run the pipeline's AutoGain over 100 ms chunks, as live capture does."""
    agc, chunk = AutoGain(), int(sr * 0.1)
    return np.concatenate([agc.apply(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# ---- scoring -------------------------------------------------------------------------------

def edits(hyp_tokens: list[str], ref_tokens: list[str]) -> int:
    d = np.zeros((len(ref_tokens) + 1, len(hyp_tokens) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref_tokens) + 1)
    d[0, :] = np.arange(len(hyp_tokens) + 1)
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return int(d[len(ref_tokens), len(hyp_tokens)])


def tokens(text: str) -> list[str]:
    return NORM(text).split()


def fluent(toks: list[str]) -> list[str]:
    """Drop standalone fillers / filler phrases, then collapse immediate
    unigram ("the the") and bigram ("when you when you") repeats."""
    kept: list[str] = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and (toks[i], toks[i + 1]) in FILLER_PHRASES:
            i += 2
            continue
        if toks[i] in FILLERS:
            i += 1
            continue
        kept.append(toks[i])
        i += 1
    res: list[str] = []
    i = 0
    while i < len(kept):
        if res and res[-1] == kept[i]:
            i += 1
            continue
        if len(res) >= 2 and i + 1 < len(kept) and res[-2:] == kept[i:i + 2]:
            i += 2
            continue
        res.append(kept[i])
        i += 1
    return res


def score(hyp: str, ref: str) -> dict:
    h, r = tokens(hyp), tokens(ref)
    hf, rf = fluent(h), fluent(r)
    return {"edits": edits(h, r), "n_ref": len(r), "edits_fluent": edits(hf, rf), "n_ref_fluent": len(rf)}


# ---- Confucius via llama-server -------------------------------------------------------

def start_server(quant: str) -> subprocess.Popen | None:
    try:
        if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
            return None
    except requests.RequestException:
        pass
    proc = subprocess.Popen(
        [str(LLAMA_SERVER), "-m", str(path_of(f"confucius4-r2t2-{quant.lower()}-gguf")),
         "--mmproj", str(path_of("confucius4-r2t2-mmproj-gguf")),
         "-ngl", "99", "-c", "4096", "--port", str(PORT), "-np", "1"],
        stdout=subprocess.DEVNULL, stderr=open(ESB / "llama-server.log", "w"),
    )
    for _ in range(180):
        time.sleep(1)
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                return proc
        except requests.RequestException:
            continue
    proc.kill()
    raise SystemExit("llama-server did not become healthy")


def parse_asr_output(text: str) -> str:
    """Qwen3-ASR emits 'language English<asr_text>...'; keep the transcript only."""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = re.sub(r"^\s*language\s+\w+\s*", "", text)
    return re.sub(r"<[^>]+>", "", text).strip()


def confucius_transcribe(session: requests.Session, wav: bytes) -> tuple[str, str, float]:
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
        ]}],
        "max_tokens": 512, "temperature": 0.0,
    }
    t0 = time.monotonic()
    r = session.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=300)
    dt = time.monotonic() - t0
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return parse_asr_output(raw), raw, dt


# ---- Parakeet lanes -----------------------------------------------------------------------

def load_parakeet_offline():
    import sherpa_onnx

    c = EN_ASR_MODELS[PARAKEET_KEY]
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=c.encoder, decoder=c.decoder, joiner=c.joiner, tokens=c.tokens,
        num_threads=8, model_type="nemo_transducer",
    )


def parakeet_offline(rec, x: np.ndarray, sr: int) -> tuple[str, float]:
    t0 = time.monotonic()
    s = rec.create_stream(); s.accept_waveform(sr, x); rec.decode_stream(s)
    return s.result.text.strip(), time.monotonic() - t0


_SEMI = None


def production_finals(x: np.ndarray, sr: int) -> tuple[list[str], float]:
    """Exactly the live lane: one long-lived SemiStreamingAsr (as in production),
    100 ms chunks, silence flush; state reset between utterances."""
    global _SEMI
    if _SEMI is None:
        _SEMI = create_asr(EN_ASR_MODELS[PARAKEET_KEY])
    asr = _SEMI
    asr._reset()
    chunk, finals = int(sr * 0.1), []
    t0 = time.monotonic()
    for i in range(0, len(x), chunk):
        finals += [ev.text for ev in asr.accept(x[i:i + chunk], sr) if ev.is_final]
    for _ in range(25):
        finals += [ev.text for ev in asr.accept(np.zeros(chunk, dtype=np.float32), sr) if ev.is_final]
    return finals, time.monotonic() - t0


class Corrector:
    """One warm DialogueInterpreter; history cleared between utterances."""

    def __init__(self):
        from interpreter.dialogue import OTHER, DialogueInterpreter

        self._it = DialogueInterpreter(TranslatorConfig())
        self._it.warm_up()
        self._other = OTHER

    def reset(self) -> None:
        self._it._history = []

    def correct(self, raw: str) -> tuple[str, float, bool]:
        t0 = time.monotonic()
        res = self._it.interpret(raw, self._other)
        return res.corrected, time.monotonic() - t0, res.changed


# ---- driver -------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quant", default="f16")
    p.add_argument("--limit", type=int, default=0, help="max utterances per subset (0 = all)")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--skip-confucius", action="store_true", help="Parakeet lanes only (no llama-server)")
    p.add_argument("--parakeet-engine", default="parakeet-semi",
                   help="EN_ASR_MODELS key for the Parakeet lanes (e.g. parakeet-semi-fp32)")
    p.add_argument("--tag", default="", help="suffix for the results file name")
    args = p.parse_args()
    global PARAKEET_KEY
    PARAKEET_KEY = args.parakeet_engine

    refs = json.loads((ESB / "refs.json").read_text(encoding="utf-8"))
    refs.pop("_meta", None)
    items, per_sub = [], {}
    for k, v in refs.items():
        sub = k.split("/")[0]
        if args.limit and per_sub.get(sub, 0) >= args.limit:
            continue
        per_sub[sub] = per_sub.get(sub, 0) + 1
        items.append((k, str(ESB / k), v["text"], v.get("source", "")))

    corrector = None
    if not args.skip_llm:
        from interpreter.bootstrap import ensure_ollama
        if not ensure_ollama():
            raise SystemExit("Ollama not reachable")
        corrector = Corrector()

    proc = None if args.skip_confucius else start_server(args.quant)
    session = requests.Session()
    rec = load_parakeet_offline()
    out = ESB / f"results_{args.quant}{('_' + args.tag) if args.tag else ''}.jsonl"
    rows = []
    try:
        with out.open("w", encoding="utf-8") as fh:
            for i, (key, path, ref, source) in enumerate(items, 1):
                x, sr = read_wav(path)
                xa = agc_condition(x, sr)
                row = {"key": key, "subset": key.split("/")[0], "source": source,
                       "dur": len(x) / sr, "ref": ref, "errors": {}}
                if args.skip_confucius:
                    row.update(confucius="", confucius_raw="", t_confucius=0.0)
                else:
                    try:
                        txt, raw, dt = confucius_transcribe(session, wav_bytes(xa, sr))
                        row.update(confucius=txt, confucius_raw=raw, t_confucius=dt)
                    except Exception as e:  # noqa: BLE001
                        row["errors"]["confucius"] = str(e); row.update(confucius="", t_confucius=0.0)
                try:
                    txt, dt = parakeet_offline(rec, xa, sr)
                    row.update(parakeet=txt, t_parakeet=dt)
                except Exception as e:  # noqa: BLE001
                    row["errors"]["parakeet"] = str(e); row.update(parakeet="", t_parakeet=0.0)
                try:
                    finals, dt = production_finals(xa, sr)
                    row.update(production_finals=finals, t_production_asr=dt)
                except Exception as e:  # noqa: BLE001
                    row["errors"]["production"] = str(e); row.update(production_finals=[], t_production_asr=0.0)
                if corrector is None:
                    row.update(parakeet_llm=row["parakeet"], t_llm=0.0,
                               production=" ".join(row["production_finals"]), t_production_llm=0.0)
                else:
                    try:
                        corrector.reset()
                        txt, dt, changed = corrector.correct(row["parakeet"]) if row["parakeet"] else ("", 0.0, False)
                        row.update(parakeet_llm=txt, t_llm=dt, llm_changed=changed)
                    except Exception as e:  # noqa: BLE001
                        row["errors"]["parakeet_llm"] = str(e); row.update(parakeet_llm=row["parakeet"], t_llm=0.0)
                    try:
                        corrector.reset()
                        parts, t_sum = [], 0.0
                        for f in row["production_finals"]:
                            c, dt, _ = corrector.correct(f); parts.append(c); t_sum += dt
                        row.update(production=" ".join(parts), t_production_llm=t_sum)
                    except Exception as e:  # noqa: BLE001
                        row["errors"]["production"] = str(e)
                        row.update(production=" ".join(row["production_finals"]), t_production_llm=0.0)
                for lane in LANES:
                    row[f"s_{lane}"] = score(row.get(lane, ""), ref)
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                w = {l: row[f"s_{l}"]["edits"] / max(1, row[f"s_{l}"]["n_ref"]) * 100 for l in LANES}
                print(f"[{i}/{len(items)}] {key} ({row['dur']:.1f}s) conf {w['confucius']:5.1f} | "
                      f"pk {w['parakeet']:5.1f} | pk+llm {w['parakeet_llm']:5.1f} | prod {w['production']:5.1f}"
                      + (f"  !! {row['errors']}" if row["errors"] else ""))
    finally:
        if proc is not None:
            proc.terminate()
    report(rows)
    print(f"\nper-utterance results -> {out}")


# ---- report ---------------------------------------------------------------------------------

def corpus_wer(rows: list[dict], lane: str, fl: bool = False) -> float:
    e = "edits_fluent" if fl else "edits"; n = "n_ref_fluent" if fl else "n_ref"
    return 100 * sum(r[f"s_{lane}"][e] for r in rows) / max(1, sum(r[f"s_{lane}"][n] for r in rows))


def bootstrap_diff(rows: list[dict], a: str, b: str, fl: bool = False, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of corpus_wer(a) - corpus_wer(b), resampling recordings (clusters) when available."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("source") or r["key"], []).append(r)
    keys = list(groups)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for k in pick for r in groups[keys[k]]]
        diffs.append(corpus_wer(sample, a, fl) - corpus_wer(sample, b, fl))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def report(rows: list[dict]) -> None:
    subsets = sorted({r["subset"] for r in rows})
    for fl, title in ((False, "VERBATIM WER %"), (True, "FLUENT WER % (repeats + fillers collapsed on both sides)")):
        print("\n" + "=" * 100 + f"\n{title}")
        print(f"{'subset':<18}{'n':>4}{'min':>6} | {'confucius':>9} | {'parakeet':>9} | {'pk+llm':>9} | {'production':>10} |  conf-prod 95% CI")
        print("-" * 100)
        for sub in subsets + ["ALL"]:
            sr = rows if sub == "ALL" else [r for r in rows if r["subset"] == sub]
            w = {l: corpus_wer(sr, l, fl) for l in LANES}
            lo, hi = bootstrap_diff(sr, "confucius", "production", fl)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"{sub:<18}{len(sr):>4}{sum(r['dur'] for r in sr)/60:>6.1f} | {w['confucius']:9.2f} | {w['parakeet']:9.2f} | "
                  f"{w['parakeet_llm']:9.2f} | {w['production']:10.2f} |  [{lo:+.2f}, {hi:+.2f}]{sig}")
    audio = sum(r["dur"] for r in rows)
    print("\nlatency (seconds of compute per second of audio): "
          f"confucius GPU {sum(r['t_confucius'] for r in rows)/audio:.3f} | "
          f"parakeet offline CPU {sum(r['t_parakeet'] for r in rows)/audio:.3f} | "
          f"production semi CPU {sum(r['t_production_asr'] for r in rows)/audio:.3f} | "
          f"LLM correction(+translation) {sum(r.get('t_llm', 0) for r in rows)/audio:.3f}")
    empties = {l: sum(1 for r in rows if not r.get(l)) for l in LANES}
    errs = sum(1 for r in rows if r["errors"])
    print(f"empty outputs per lane: {empties} | rows with errors: {errs}")
    print("CI = cluster bootstrap over recordings, 1000 draws; * = interval excludes 0 (negative favours confucius).")


if __name__ == "__main__":
    main()
