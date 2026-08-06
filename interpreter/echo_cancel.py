"""Digital-domain echo cancellation for loopback capture.

WASAPI loopback taps the endpoint mix, so our own TTS playback reappears in
the capture almost verbatim — same clock, constant delay, constant gain, no
room acoustics. Since we know exactly what we played, we can locate it by
cross-correlation and subtract it, instead of gating (and losing) the
original programme audio while the interpreter speaks.

States: idle -> aligning (output muted, <=1.5s) -> cancelling (subtract) ->
idle. If alignment fails, we stay muted for the reference duration, which
degrades gracefully to the old gate behaviour.
"""
import numpy as np

PROBE_S = 0.5        # seconds of reference used for alignment
SEARCH_S = 1.5       # capture window searched for the reference onset
MIN_GAIN, MAX_GAIN = 0.02, 4.0
PEAK_RATIO = 3.0     # correlation peak must stand out this much


class EchoCanceller:
    def __init__(self, capture_rate: int):
        self._rate = capture_rate
        self._ref: np.ndarray | None = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._gain = 1.0
        self.state = "idle"

    @property
    def active(self) -> bool:
        return self.state != "idle"

    def set_reference(self, samples: np.ndarray, sample_rate: int) -> None:
        """Register the audio that is about to be played."""
        if sample_rate != self._rate:
            n = int(len(samples) * self._rate / sample_rate)
            samples = np.interp(
                np.linspace(0.0, len(samples), n, endpoint=False),
                np.arange(len(samples)), samples,
            )
        self._ref = samples.astype(np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0
        # Pick the most energetic PROBE_S window as the alignment probe —
        # synthesized speech often starts with near-silence, which makes a
        # leading-edge probe correlate with anything.
        probe_n = int(self._rate * PROBE_S)
        if len(self._ref) > probe_n:
            hop = probe_n // 4
            energies = [
                float(np.dot(self._ref[i:i + probe_n], self._ref[i:i + probe_n]))
                for i in range(0, len(self._ref) - probe_n, hop)
            ]
            self._probe_at = int(np.argmax(energies)) * hop
        else:
            self._probe_at = 0
        self.state = "aligning"

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Feed one capture chunk; returns the echo-cleaned chunk."""
        if self.state == "idle" or self._ref is None:
            return chunk
        if self.state == "aligning":
            self._buf = np.concatenate([self._buf, chunk])
            probe_n = int(self._rate * PROBE_S)
            search_n = int(self._rate * SEARCH_S) + self._probe_at
            probe = self._ref[self._probe_at:self._probe_at + probe_n]
            # The probe sits _probe_at samples into the reference, so it can
            # only appear in the capture after that much audio has played.
            if len(self._buf) >= self._probe_at + probe_n + int(self._rate * 0.2):
                offset = self._locate(self._buf, probe)
                if offset is not None and offset >= self._probe_at:
                    echo_start = offset - self._probe_at
                    self._pos = len(self._buf) - echo_start
                    self.state = "cancelling"
                elif len(self._buf) > search_n:
                    # Alignment failed — behave like the old gate: stay muted
                    # for the rest of the reference, then resume.
                    self.state = "muting"
            return np.zeros_like(chunk)
        if self.state == "muting":
            self._pos += len(chunk)
            if self._pos >= len(self._ref):
                self.state = "idle"
            return np.zeros_like(chunk)
        # cancelling: subtract, then adapt the single-tap gain on the residual
        seg = self._ref[self._pos:self._pos + len(chunk)]
        cleaned = chunk.copy()
        cleaned[: len(seg)] -= self._gain * seg
        seg_energy = float(np.dot(seg, seg))
        if seg_energy > 1e-4:
            err = float(np.dot(cleaned[: len(seg)], seg)) / seg_energy
            self._gain = float(np.clip(self._gain + 0.5 * err, MIN_GAIN, MAX_GAIN))
        self._pos += len(chunk)
        if self._pos >= len(self._ref):
            self.state = "idle"
        # If subtraction removed nearly everything, the capture was only our
        # own echo (no programme audio underneath) — gate the residual so the
        # ASR never hears a faint copy of our own voice in silence.
        in_energy = float(np.dot(chunk, chunk))
        if in_energy > 1e-6 and float(np.dot(cleaned, cleaned)) < 0.05 * in_energy:
            return np.zeros_like(chunk)
        return cleaned

    def _locate(self, buf: np.ndarray, probe: np.ndarray) -> int | None:
        """Find probe's offset in buf via FFT cross-correlation."""
        n = len(buf) + len(probe)
        fft_n = 1 << (n - 1).bit_length()
        corr = np.fft.irfft(
            np.fft.rfft(buf, fft_n) * np.conj(np.fft.rfft(probe, fft_n)), fft_n
        )[: len(buf) - len(probe) + 1]
        if corr.size == 0:
            return None
        peak = int(np.argmax(corr))
        baseline = np.median(np.abs(corr)) + 1e-9
        if corr[peak] / baseline < PEAK_RATIO:
            return None
        seg = buf[peak:peak + len(probe)]
        denom = float(np.dot(probe[: len(seg)], probe[: len(seg)])) + 1e-9
        gain = float(np.dot(seg, probe[: len(seg)])) / denom
        if not (MIN_GAIN <= gain <= MAX_GAIN):
            return None
        self._gain = gain
        return peak
