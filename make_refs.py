"""Generate pseudo-reference transcripts for testclips/ using Parakeet-TDT v3.

Parakeet-TDT 0.6B v3 (offline, non-streaming) is far stronger than any of the
streaming candidates, so its output serves as the reference to score them.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
PK = BASE / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
CLIPS = BASE / "testclips"


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0, f.getframerate()


def main() -> None:
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(PK / "encoder.int8.onnx"),
        decoder=str(PK / "decoder.int8.onnx"),
        joiner=str(PK / "joiner.int8.onnx"),
        tokens=str(PK / "tokens.txt"),
        num_threads=8,
        model_type="nemo_transducer",
    )
    refs: dict[str, str] = {}
    for wav in sorted(CLIPS.glob("*.wav")):
        samples, sr = read_wav(wav)
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        recognizer.decode_stream(stream)
        refs[wav.name] = stream.result.text.strip()
        print(f"{wav.name}: {refs[wav.name]}")
    (CLIPS / "refs.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {CLIPS / 'refs.json'}")


if __name__ == "__main__":
    main()
