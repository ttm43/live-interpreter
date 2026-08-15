"""System audio (loopback) capture via WASAPI using PyAudioWPatch."""
import queue
import threading

import numpy as np
import pyaudiowpatch as pyaudio


class LoopbackCapture:
    """Captures whatever is playing on the speakers as mono float32 chunks."""

    def __init__(self, device_index: int | None = None, chunk_ms: int = 100):
        self._pa = pyaudio.PyAudio()
        self._device = self._resolve_device(device_index)
        self.sample_rate = int(self._device["defaultSampleRate"])
        self.channels = max(1, int(self._device["maxInputChannels"]))
        self._frames_per_buffer = int(self.sample_rate * chunk_ms / 1000)
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream = None
        # When set, incoming frames are discarded (anti-feedback while TTS plays).
        self.suppress = threading.Event()
        self.callback_errors = 0

    def _resolve_device(self, device_index: int | None) -> dict:
        if device_index is not None:
            return self._pa.get_device_info_by_index(device_index)
        wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            return speakers
        for loopback in self._pa.get_loopback_device_info_generator():
            if speakers["name"] in loopback["name"]:
                return loopback
        raise RuntimeError(
            "No WASAPI loopback device found for default speakers: "
            f"{speakers['name']!r}. Run with --list-devices to pick one manually."
        )

    @property
    def device_name(self) -> str:
        return self._device["name"]

    def _callback(self, in_data, frame_count, time_info, status):
        # Never let an exception escape into the PortAudio callback thread —
        # that would silently kill the stream with no diagnostics.
        try:
            if not self.suppress.is_set():
                samples = np.frombuffer(in_data, dtype=np.float32)
                if self.channels > 1 and samples.size % self.channels == 0:
                    # Only average the front L/R pair: on 5.1/7.1 endpoints the
                    # other channels are usually empty and averaging them all
                    # attenuates the signal (8ch stereo content -> 1/4 level).
                    samples = samples.reshape(-1, self.channels)[:, :2].mean(axis=1)
                try:
                    self._chunks.put_nowait(samples)
                except queue.Full:
                    pass  # drop oldest behaviour is fine for live captioning
        except Exception:  # noqa: BLE001
            self.callback_errors += 1
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self._device["index"],
            frames_per_buffer=self._frames_per_buffer,
            stream_callback=self._callback,
        )

    def read(self, timeout: float = 0.2) -> np.ndarray | None:
        try:
            return self._chunks.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._pa.terminate()


class AutoGain:
    """Slow-tracking automatic gain so quiet system audio still drives ASR.

    Low playback volume plus leading silence makes the streaming decoder miss
    the first seconds of speech entirely; normalizing peaks to ~0.5 fixes it.
    """

    def __init__(self, target_peak: float = 0.5, max_gain: float = 20.0):
        self._target = target_peak
        self._max_gain = max_gain
        self._peak = 0.0
        self._gain = 1.0

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        self._peak = max(self._peak * 0.995, peak)
        if self._peak < 1e-4:
            return chunk
        wanted = min(self._target / self._peak, self._max_gain)
        # Slew-limit gain changes (<=10% per chunk): a jumpy gain modulates
        # loudness WITHIN an utterance, which wrecks re-decoding ASR engines.
        self._gain = float(np.clip(wanted, self._gain / 1.1, self._gain * 1.1))
        # Hard-limit output so a stale (tiny) tracked peak right after silence
        # can't blast the utterance onset past ±1.
        return np.clip(chunk * self._gain, -1.0, 1.0)


class KeepAliveOutput:
    """Continuously renders silence so the WASAPI loopback stream never stalls.

    Without an active render stream, loopback capture delivers no frames and
    the first ~1-4s of any new audio is lost (device wake-up / session start).
    A permanent silent output stream keeps the render path hot.
    """

    def __init__(self, match_name: str | None = None):
        self._pa = pyaudio.PyAudio()
        device_index = self._find_render_device(match_name)
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=48000,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=4800,
            stream_callback=lambda *_: (b"\x00" * 4800 * 4, pyaudio.paContinue),
        )

    def _find_render_device(self, match_name: str | None) -> int | None:
        """Find the render device whose loopback twin is being captured."""
        if not match_name:
            return None  # default output device
        base = match_name.replace(" [Loopback]", "").strip()
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if (
                dev.get("maxOutputChannels", 0) > 0
                and not dev.get("isLoopbackDevice")
                and dev["name"].strip() == base
            ):
                return i
        return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._pa.terminate()


def get_loopback_devices() -> list[tuple[int, str]]:
    """(index, name) pairs of WASAPI loopback capture devices."""
    pa = pyaudio.PyAudio()
    devices = [
        (dev["index"], dev["name"])
        for dev in pa.get_loopback_device_info_generator()
    ]
    pa.terminate()
    return devices


def list_devices() -> str:
    """Human-readable listing of WASAPI loopback and output devices."""
    pa = pyaudio.PyAudio()
    lines = ["-- WASAPI loopback (capture) devices --"]
    for dev in pa.get_loopback_device_info_generator():
        lines.append(
            f"  [{dev['index']}] {dev['name']} "
            f"({int(dev['defaultSampleRate'])} Hz, {dev['maxInputChannels']} ch)"
        )
    lines.append("-- Output (playback) devices --")
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        if dev.get("maxOutputChannels", 0) > 0 and not dev.get("isLoopbackDevice"):
            lines.append(f"  [{dev['index']}] {dev['name']}")
    pa.terminate()
    return "\n".join(lines)
