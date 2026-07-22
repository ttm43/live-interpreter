"""Bilingual TTS (Kokoro multi-lang via sherpa-onnx) with speaker playback."""
import threading

import numpy as np
import pyaudiowpatch as pyaudio
import sherpa_onnx

from .config import TtsConfig


class BilingualTts:
    """Synthesizes zh/en text and plays it on an output device.

    While audio is playing, `playing` is set so the capture side can gate its
    input and avoid re-interpreting our own voice (feedback loop).
    """

    def __init__(self, cfg: TtsConfig, output_device_index: int | None = None):
        self._cfg = cfg
        self._tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=cfg.model,
                        voices=cfg.voices,
                        tokens=cfg.tokens,
                        data_dir=cfg.data_dir,
                        dict_dir=cfg.dict_dir,
                        lexicon=cfg.lexicon,
                    ),
                    num_threads=cfg.num_threads,
                    provider="cpu",
                ),
                max_num_sentences=1,
            )
        )
        self._pa = pyaudio.PyAudio()
        self._output_device_index = output_device_index
        self._lock = threading.Lock()
        self.playing = threading.Event()

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        sid = self._cfg.zh_speaker_id if lang == "zh" else self._cfg.en_speaker_id
        audio = self._tts.generate(text, sid=sid, speed=self._cfg.speed)
        return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate

    def speak(self, text: str, lang: str) -> bool:
        """Synthesize and play; returns True only if audio was actually played."""
        samples, sample_rate = self.synthesize(text, lang)
        if samples.size == 0:
            return False
        with self._lock:
            self.playing.set()
            try:
                stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    output_device_index=self._output_device_index,
                )
                stream.write(samples.tobytes())
                stream.stop_stream()
                stream.close()
            finally:
                self.playing.clear()
        return True

    def close(self) -> None:
        self._pa.terminate()
