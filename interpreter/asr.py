"""Streaming bilingual (zh/en) ASR built on sherpa-onnx online paraformer."""
from dataclasses import dataclass

import numpy as np
import sherpa_onnx

from .config import AsrConfig


@dataclass(frozen=True)
class AsrEvent:
    text: str
    is_final: bool


def create_asr(cfg: AsrConfig):
    """Factory: streaming transducer/paraformer, or semi-streaming offline."""
    if cfg.kind == "offline_transducer":
        from .asr_semi import SemiStreamingAsr  # avoid import cycle

        return SemiStreamingAsr(cfg)
    if cfg.kind == "whisper":
        from .asr_semi import WhisperSemiAsr

        return WhisperSemiAsr(cfg)
    return StreamingAsr(cfg)


class StreamingAsr:
    """Feed audio chunks in, get partial/final transcript events out."""

    def __init__(self, cfg: AsrConfig):
        self._cfg = cfg
        common = dict(
            tokens=cfg.tokens,
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=cfg.rule1_min_trailing_silence,
            rule2_min_trailing_silence=cfg.rule2_min_trailing_silence,
            rule3_min_utterance_length=cfg.rule3_min_utterance_length,
        )
        if cfg.kind in ("zipformer", "nemo_transducer"):
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=cfg.encoder, decoder=cfg.decoder, joiner=cfg.joiner,
                model_type="nemo_transducer" if cfg.kind == "nemo_transducer" else "",
                **common,
            )
        else:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                encoder=cfg.encoder, decoder=cfg.decoder, **common
            )
        self._stream = self._recognizer.create_stream()
        self._last_partial = ""

    def accept(self, samples: np.ndarray, sample_rate: int) -> list[AsrEvent]:
        """Feed one chunk; returns zero or more transcript events."""
        events: list[AsrEvent] = []
        self._stream.accept_waveform(sample_rate, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        text = self._recognizer.get_result(self._stream).strip()
        if text.isupper():  # en zipformer emits ALL CAPS; normalize it
            text = text.lower()
        if self._recognizer.is_endpoint(self._stream):
            # Only reset when something was decoded. Resetting an empty stream
            # (rule1 fires every 2.4s during silence) discards buffered but
            # not-yet-decoded leading speech — the first seconds after silence
            # would be lost.
            if text:
                events.append(AsrEvent(text=text, is_final=True))
                self._recognizer.reset(self._stream)
                self._last_partial = ""
        elif text and text != self._last_partial:
            events.append(AsrEvent(text=text, is_final=False))
            self._last_partial = text
        return events
