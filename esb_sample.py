"""Sample human-transcribed English test utterances (Open ASR Leaderboard test
sets) into testclips/esb/ so ASR comparisons have real ground truth instead of
the Parakeet pseudo-references in testclips/refs.json.

Every parquet shard of hf-audio/open-asr-leaderboard is sorted by duration
(descending) and the shards partition the split into equal-count length bands,
so we draw equally from several shards spread over the index range — this
approximates a uniform sample of the split's duration distribution instead of
the longest band only. Draws are capped per source (meeting / call / podcast /
speaker-chapter) so one bad recording cannot dominate a subset.

Usage: .venv\\Scripts\\python esb_sample.py [--per-subset 80] [--seed 42]
Writes 16 kHz mono wavs + refs.json (with sampling metadata).
"""
import argparse
import io
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, r"C:\Users\46025\work\models")
from model_registry import cache_env  # noqa: E402

os.environ.update(cache_env())
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "testclips" / "esb"
REPO = "hf-audio/open-asr-leaderboard"
# Shards are equal-count duration bands (descending). Bands entirely below
# MIN_S (AMI 12-14, GigaSpeech 16-18: sub-second backchannels) are skipped.
SHARDS = {
    "ami": [f"ami/test-{i:05d}-of-00015.parquet" for i in (0, 2, 4, 6, 8)],
    "earnings22": [f"earnings22/test-{i:05d}-of-00005.parquet" for i in (0, 2, 4)],
    "gigaspeech": [f"gigaspeech/test-{i:05d}-of-00019.parquet" for i in (0, 3, 6, 9, 12, 15)],
    "librispeech_clean": ["librispeech/test.clean-00000-of-00001.parquet"],
}
MIN_S, MAX_S = 0.5, 30.0
# AMI test has only 16 meetings, Earnings22 ~a dozen calls: caps sized so the
# subset can still reach --per-subset while no recording dominates.
MAX_PER_SOURCE = {"ami": 6, "earnings22": 14, "gigaspeech": 4, "librispeech_clean": 4}


def decode_audio(cell) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x
    return librosa.resample(x, orig_sr=sr, target_sr=16000, res_type="soxr_hq")


def source_of(subset: str, uid: str) -> str:
    """Recording-level id from the dataset's id formats:
    AMI 'AMI_ES2004b_H02_MEE014_...' -> meeting ES2004b; Earnings22 '4483338/281.wav'
    -> call 4483338; GigaSpeech 'POD1000000005_S0000339' -> podcast/video;
    LibriSpeech '1089-134686-0002' -> speaker-chapter."""
    if subset == "ami":
        return uid.split("_")[1]
    if subset == "earnings22":
        return uid.split("/")[0]
    if subset == "gigaspeech":
        return uid.split("_")[0]
    return "-".join(uid.split("-")[:2])


def draw_from_shard(subset: str, shard: str, n: int, rng: random.Random,
                    per_source: Counter, out_dir: Path, start_idx: int) -> dict:
    path = hf_hub_download(REPO, shard, repo_type="dataset")
    table = pq.read_table(path, columns=["audio", "text", "id", "audio_length_s"])
    order = list(range(table.num_rows))
    rng.shuffle(order)
    refs: dict[str, dict] = {}
    for i in order:
        if len(refs) >= n:
            break
        row = table.slice(i, 1).to_pylist()[0]
        text = (row["text"] or "").strip()
        dur = float(row["audio_length_s"] or 0)
        src = source_of(subset, str(row["id"]))
        if not text or not (MIN_S <= dur <= MAX_S) or per_source[src] >= MAX_PER_SOURCE[subset]:
            continue
        x, sr = decode_audio(row["audio"])
        x16 = to_16k(x, sr)
        fname = f"{start_idx + len(refs):03d}.wav"
        sf.write(out_dir / fname, (np.clip(x16, -1, 1) * 32767).astype(np.int16), 16000)
        per_source[src] += 1
        refs[f"{subset}/{fname}"] = {
            "text": text, "duration_s": round(len(x16) / 16000, 2), "orig_sr": sr,
            "source_id": str(row["id"]), "source": src, "shard": shard,
        }
    return refs


def sample_subset(subset: str, n_total: int, seed: int) -> dict:
    rng = random.Random(f"{seed}:{subset}")
    shards = SHARDS[subset]
    per_shard = [n_total // len(shards) + (1 if k < n_total % len(shards) else 0) for k in range(len(shards))]
    out_dir = OUT / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.wav"):
        old.unlink()
    per_source: Counter = Counter()
    refs: dict[str, dict] = {}
    for shard, n in zip(shards, per_shard):
        refs.update(draw_from_shard(subset, shard, n, rng, per_source, out_dir, len(refs)))
    durs = sorted(v["duration_s"] for v in refs.values())
    print(f"{subset}: {len(refs)} utts from {len(shards)} shards, {len(per_source)} sources "
          f"(max {max(per_source.values())}/source), dur min/med/max "
          f"{durs[0]:.1f}/{durs[len(durs)//2]:.1f}/{durs[-1]:.1f}s")
    return refs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-subset", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    refs: dict[str, dict] = {}
    for subset in SHARDS:
        refs.update(sample_subset(subset, args.per_subset, args.seed))
    meta = {"_meta": {"seed": args.seed, "per_subset": args.per_subset, "shards": SHARDS,
                      "duration_bounds_s": [MIN_S, MAX_S], "max_per_source": MAX_PER_SOURCE,
                      "source_of": "AMI meeting / earnings call / GigaSpeech recording / LibriSpeech speaker-chapter",
                      "resample": "librosa soxr_hq to 16 kHz", "text": "verbatim 'text' column"}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "refs.json").write_text(json.dumps({**meta, **refs}, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(v["duration_s"] for v in refs.values())
    print(f"wrote {len(refs)} utterances ({total / 60:.1f} min) -> {OUT / 'refs.json'}")


if __name__ == "__main__":
    main()
