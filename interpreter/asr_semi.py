"""Semi-streaming ASR: an offline model re-decoding a rolling utterance buffer.

Offline models (Parakeet-TDT) are far more accurate than any streaming model,
but only decode complete audio. We fake streaming by re-decoding the growing
utterance every few hundred ms — partials arrive as whole-hypothesis
revisions (the GUI's live line already revises in place), and an energy-based
endpoint triggers the final decode. The re-decode interval self-throttles to
the decode speed, so long utterances can't pile up.
"""
import time

import numpy as np
import sherpa_onnx

from .asr import AsrEvent
from .config import AsrConfig

SILENCE_RMS = 0.004     # below this a chunk counts as silence
PRE_ROLL_S = 1.0        # audio kept before detected speech onset
MIN_DECODE_GAP_S = 0.6  # audio-seconds between re-decodes (adaptive floor)


class SemiStreamingAsr:
    """Same accept() interface as StreamingAsr, backed by an offline model."""

    def __init__(self, cfg: AsrConfig):
        self._cfg = cfg
        self._load()
        self._buf: list[np.ndarray] = []
        self._buf_s = 0.0
        self._rate = 0
        self._speech = False
        self._trailing_silence_s = 0.0
        self._since_decode_s = 0.0
        self._decode_gap_s = MIN_DECODE_GAP_S
        self._last_partial = ""

    def _load(self) -> None:
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self._cfg.encoder,
            decoder=self._cfg.decoder,
            joiner=self._cfg.joiner,
            tokens=self._cfg.tokens,
            num_threads=max(self._cfg.num_threads, 8),
            model_type="nemo_transducer",
        )

    def _decode_samples(self, samples: np.ndarray, rate: int) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def _decode(self) -> str:
        t0 = time.monotonic()
        text = self._decode_samples(np.concatenate(self._buf), self._rate)
        # Re-decoding a whole window is expensive; space decodes so we never
        # spend more than ~half our time decoding.
        self._decode_gap_s = max(MIN_DECODE_GAP_S, (time.monotonic() - t0) * 1.5)
        return text

    def _reset(self) -> None:
        self._buf = []
        self._buf_s = 0.0
        self._speech = False
        self._trailing_silence_s = 0.0
        self._since_decode_s = 0.0
        self._last_partial = ""

    def accept(self, samples: np.ndarray, sample_rate: int) -> list[AsrEvent]:
        self._rate = sample_rate
        chunk_s = len(samples) / sample_rate
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0

        if rms < SILENCE_RMS:
            self._trailing_silence_s += chunk_s
        else:
            self._trailing_silence_s = 0.0
            self._speech = True

        if not self._speech:
            # Nothing but silence so far: keep only a short pre-roll so the
            # utterance onset isn't clipped, and don't decode anything.
            self._buf.append(samples)
            self._buf_s += chunk_s
            while self._buf and self._buf_s > PRE_ROLL_S:
                self._buf_s -= len(self._buf[0]) / sample_rate
                self._buf.pop(0)
            return []

        self._buf.append(samples)
        self._buf_s += chunk_s
        self._since_decode_s += chunk_s

        # Natural endpoint: enough trailing silence — decode all and reset.
        if self._trailing_silence_s >= self._cfg.rule2_min_trailing_silence:
            text = self._decode()
            self._reset()
            return [AsrEvent(text=text, is_final=True)] if text else []

        # Hard cap on continuous speech: never cut mid-word — split at the
        # quietest point in the window's tail and carry the remainder over
        # into the next window.
        if self._buf_s >= self._cfg.rule3_min_utterance_length:
            flat = np.concatenate(self._buf)
            tail_n = int(3.0 * sample_rate)
            win_n = max(1, int(0.15 * sample_rate))
            tail = flat[-tail_n:]
            energies = [
                float(np.dot(tail[i:i + win_n], tail[i:i + win_n]))
                for i in range(0, len(tail) - win_n, win_n)
            ]
            split = len(flat) - tail_n + int(np.argmin(energies)) * win_n + win_n // 2
            head, rest = flat[:split], flat[split:]
            self._buf = [head]
            text = self._decode()
            self._buf = [rest]
            self._buf_s = len(rest) / sample_rate
            self._since_decode_s = 0.0
            self._last_partial = ""
            return [AsrEvent(text=text, is_final=True)] if text else []

        if self._since_decode_s >= self._decode_gap_s:
            self._since_decode_s = 0.0
            text = self._decode()
            if text and text != self._last_partial:
                self._last_partial = text
                return [AsrEvent(text=text, is_final=False)]
        return []


class WhisperSemiAsr(SemiStreamingAsr):
    """Semi-streaming with faster-whisper as the offline decoder.

    Same windowing/endpointing as SemiStreamingAsr; whisper brings cleaned-up
    output (names, digits as numerals, disfluencies removed) at ~5x the
    compute of Parakeet. cfg.encoder holds the whisper model name.
    """

    def _load(self) -> None:
        from faster_whisper import WhisperModel  # heavy import, only if used

        self._model = WhisperModel(
            self._cfg.encoder or "large-v3-turbo",
            device="cpu", compute_type="int8",
            cpu_threads=max(self._cfg.num_threads, 8),
        )

    def _decode_samples(self, samples: np.ndarray, rate: int) -> str:
        if rate != 16000:  # faster-whisper expects 16 kHz input arrays
            n = int(len(samples) * 16000 / rate)
            samples = np.interp(
                np.linspace(0.0, len(samples), n, endpoint=False),
                np.arange(len(samples)), samples,
            ).astype(np.float32)
        segments, _ = self._model.transcribe(samples, language="en", beam_size=2)
        return " ".join(s.text.strip() for s in segments).strip()
