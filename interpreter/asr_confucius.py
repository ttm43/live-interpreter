"""Confucius4-R2T2 (Youdao, Qwen3-ASR-1.7B fine-tune) through llama-server.

The model's native streaming ("longest stable prefix": output is only ever
appended to, never revised) exists only in its vLLM backend, so on Windows we
emulate it over llama.cpp: every hop re-sends the utterance audio so far plus
the already-committed transcript as an assistant prefill and asks for a few
new tokens; all but the last word of the continuation are committed. Finals
are a fresh whole-window decode (the benchmarked path), so the committed
preview is replaced exactly once. Endpointing and windowing come from
SemiStreamingAsr (energy VAD, 18 s hard cap).

Language is forced through the prefill (`language English<asr_text>`): on
very short clips the model's own language ID occasionally picks Chinese or
Dutch (3/320 in the 2026-09-26 bench).
"""
import base64
import io
import logging
import wave

import numpy as np
import requests

from .asr_semi import SemiStreamingAsr
from .bootstrap import ensure_llama_server, model_path
from .config import AsrConfig

log = logging.getLogger(__name__)

MODEL_RATE = 16000
HOP_MAX_TOKENS = 12      # new tokens per streaming hop
FINAL_MAX_TOKENS = 400
ASR_TAG = "<asr_text>"


def _to_model_rate(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == MODEL_RATE:
        return samples
    try:
        import librosa  # noqa: PLC0415 - optional, soxr-quality resampling

        return librosa.resample(samples, orig_sr=rate, target_sr=MODEL_RATE, res_type="soxr_hq")
    except ImportError:
        n = int(len(samples) * MODEL_RATE / rate)
        return np.interp(np.linspace(0.0, len(samples), n, endpoint=False),
                         np.arange(len(samples)), samples).astype(np.float32)


def _wav_b64(samples: np.ndarray, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def strip_prefill(content: str, prefill: str) -> str:
    """Text the model added after the prefill (llama-server echoes the prefill)."""
    if content.startswith(prefill):
        return content[len(prefill):]
    if ASR_TAG in content:  # prefill not echoed: drop the language header
        content = content.split(ASR_TAG, 1)[1]
    committed = prefill.split(ASR_TAG, 1)[1] if ASR_TAG in prefill else ""
    return content[len(committed):] if committed and content.startswith(committed) else content


def _join(committed: str, tail: str) -> str:
    """Concatenate committed text and a continuation, normalising whitespace
    (a continuation may start with a space, or none when it finishes a word)."""
    return " ".join((committed + tail).split())


def commit_stable(text: str) -> str:
    """Everything except the trailing word — the R2T2 'unfixed token' rollback.

    The last word sits at the audio edge: it may be cut mid-word, and the
    model tends to close it with a period because the clip ends there. Keeping
    a period-terminated word would cascade ("As. Defence. Minister. By.").
    Leading whitespace is kept: it tells _join whether the continuation
    starts a new word or finishes the committed one."""
    head, _, _ = text.rstrip().rpartition(" ")
    return head


class ConfuciusAsr(SemiStreamingAsr):
    """Same accept() interface as the other engines; append-only partials."""

    def _load(self) -> None:
        cfg = self._cfg
        self._port = cfg.server_port
        self._prefill = f"language {cfg.language or 'English'}{ASR_TAG}"
        self._model = model_path(cfg.model_key)
        self._mmproj = model_path(cfg.mmproj_key) if cfg.mmproj_key else None
        if not ensure_llama_server(self._model, self._mmproj, self._port):
            raise RuntimeError(
                f"llama-server for {cfg.model_key} could not be started on port {self._port} "
                "(no llama.cpp build found under <workspace>\\shared or the model failed to load)"
            )
        self._session = requests.Session()
        self._committed = ""

    # -- transport ----------------------------------------------------------

    def _request(self, audio: np.ndarray, rate: int, prefill: str, max_tokens: int) -> str:
        payload = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": _wav_b64(audio, rate), "format": "wav"}},
                ]},
                {"role": "assistant", "content": prefill},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        for attempt in range(2):
            try:
                r = self._session.post(
                    f"http://127.0.0.1:{self._port}/v1/chat/completions", json=payload, timeout=60,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except requests.RequestException as e:
                log.warning("confucius request failed (%s), attempt %d", e, attempt + 1)
                ensure_llama_server(self._model, self._mmproj, self._port)
        return prefill  # nothing new; caller keeps what it had

    def _decode_samples(self, samples: np.ndarray, rate: int) -> str:
        """Whole-window decode (used by the rescue path and by callers that
        want a fresh transcript): no committed prefix."""
        content = self._request(_to_model_rate(samples, rate), MODEL_RATE, self._prefill, FINAL_MAX_TOKENS)
        return strip_prefill(content, self._prefill).strip()

    # -- streaming ------------------------------------------------------------

    def _hop(self, audio: np.ndarray, rate: int, max_tokens: int) -> str:
        prefill = self._prefill + self._committed
        content = self._request(_to_model_rate(audio, rate), MODEL_RATE, prefill, max_tokens)
        return strip_prefill(content, prefill)

    def _decode(self, final: bool = False) -> str:
        audio = np.concatenate(self._buf)
        if final:
            # Fresh whole-window decode, not a continuation: this is the
            # benchmarked path (12.1% WER on human truth) and it cannot inherit
            # a name mis-spelled from a hop that ended mid-word. The GUI swaps
            # the append-only preview for it once.
            self._committed = ""
            text = self._decode_samples(audio, self._rate)
            if not text:
                text = self._rescue(audio)
            return text
        stable = commit_stable(self._hop(audio, self._rate, HOP_MAX_TOKENS))
        if stable:
            # No trailing space: the model's continuation starts with " word",
            # and a space-terminated prefill would split that token badly.
            self._committed = _join(self._committed, stable)
        return self._committed

    def _reset(self) -> None:
        super()._reset()
        self._committed = ""
