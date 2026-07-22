"""Threaded interpreter pipeline, reusable by console and GUI front-ends.

Events are delivered via callbacks which may be invoked from worker threads;
front-ends must marshal them onto their own UI thread if needed.
"""
import queue
import threading
import time
from typing import Callable

import numpy as np

from .asr import StreamingAsr
from .audio_capture import AutoGain, KeepAliveOutput, LoopbackCapture
from .config import ASR_ENGLISH, EN_ASR_MODELS, AppConfig
from .translator import OllamaTranslator, detect_lang

TTS_TAIL_GUARD_S = 0.4  # keep capture muted briefly after TTS stops

Noop = lambda *a, **k: None  # noqa: E731


class InterpreterPipeline:
    """Capture -> ASR -> translation -> optional TTS, all on worker threads."""

    def __init__(
        self,
        cfg: AppConfig,
        on_partial: Callable[[str], None] = Noop,
        on_final: Callable[[str, str], None] = Noop,          # (text, lang)
        on_translation: Callable[[str, str, float], None] = Noop,  # (text, lang, latency_s)
        on_status: Callable[[str], None] = Noop,
    ):
        self._cfg = cfg
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_translation = on_translation
        self._on_status = on_status
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._capture: LoopbackCapture | None = None
        self.running = False

    def check_backend(self) -> str | None:
        """Returns an error message if Ollama/model are unavailable, else None."""
        translator = OllamaTranslator(self._cfg.translator)
        if not translator.ping():
            return (
                f"Ollama 未运行（{self._cfg.translator.base_url}）。"
                "请先通过 run.bat / run_gui.bat 启动。"
            )
        if not translator.has_model():
            return (
                f"Ollama 中没有模型 {self._cfg.translator.model!r}，"
                f"请先执行: ollama pull {self._cfg.translator.model}"
            )
        return None

    def start(self) -> None:
        """Loads models and starts all worker threads. Raises on fatal errors."""
        cfg = self._cfg
        self._stop.clear()

        self._translator = OllamaTranslator(cfg.translator)
        self._on_status("正在加载 ASR 模型 ...")
        # English source uses a dedicated en model: far better accuracy and
        # endpointing than the Chinese-dominant bilingual model.
        if cfg.lang_mode == "en":
            asr_cfg = EN_ASR_MODELS.get(cfg.en_asr_model, ASR_ENGLISH)
        else:
            asr_cfg = cfg.asr
        self._asr = StreamingAsr(asr_cfg)

        # Optional fast preview engine (dual-ASR): live partials only.
        self._fast_asr = None
        fast_name = cfg.en_asr_fast_model
        if (
            cfg.lang_mode == "en"
            and fast_name
            and fast_name != cfg.en_asr_model
            and fast_name in EN_ASR_MODELS
        ):
            self._fast_asr = StreamingAsr(EN_ASR_MODELS[fast_name])
            self._on_status(f"双引擎: 预览 {fast_name} + 定稿 {cfg.en_asr_model}")

        self._capture = LoopbackCapture(cfg.capture_device_index)
        self._on_status(f"采集设备: {self._capture.device_name}")

        # Keep the render path active so loopback never drops leading audio.
        try:
            self._keepalive = KeepAliveOutput(self._capture.device_name)
        except Exception as e:  # noqa: BLE001 - degraded but functional without it
            self._keepalive = None
            self._on_status(f"keep-alive 输出流启动失败（开头可能丢音）: {e}")

        self._segments: "queue.Queue[tuple[str, float]]" = queue.Queue(maxsize=16)
        self._speech: "queue.Queue[tuple[str, str]] | None" = (
            queue.Queue(maxsize=8) if cfg.enable_tts else None
        )

        self._threads = [
            threading.Thread(target=self._asr_loop, daemon=True),
            threading.Thread(target=self._translation_worker, daemon=True),
        ]
        if self._speech is not None:
            self._threads.append(threading.Thread(target=self._tts_worker, daemon=True))

        self._capture.start()
        for t in self._threads:
            t.start()
        self.running = True
        self._on_status("运行中 — 正在监听系统音频")

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception as e:  # noqa: BLE001
                self._on_status(f"采集关闭异常: {e}")
            self._capture = None
        if getattr(self, "_keepalive", None) is not None:
            try:
                self._keepalive.stop()
            except Exception:  # noqa: BLE001
                pass
            self._keepalive = None
        self.running = False
        self._on_status("已停止")

    # -- worker loops --------------------------------------------------------

    def _asr_loop(self) -> None:
        capture = self._capture
        agc = AutoGain()
        # WASAPI loopback delivers no frames while nothing is playing, so feed
        # synthetic silence on timeout to keep endpoint detection moving.
        silence = np.zeros(int(capture.sample_rate * 0.2), dtype=np.float32)
        fast = self._fast_asr
        while not self._stop.is_set():
            chunk = capture.read()
            chunk = silence if chunk is None else agc.apply(chunk)
            if fast is not None:
                # Preview engine drives the live partial line only; its finals
                # are discarded (accept() still resets its stream internally).
                for event in fast.accept(chunk, capture.sample_rate):
                    if not event.is_final:
                        self._on_partial(event.text)
            for event in self._asr.accept(chunk, capture.sample_rate):
                if event.is_final:
                    if len(event.text) < self._cfg.min_chars_to_translate:
                        continue
                    mode = self._cfg.lang_mode
                    src_lang = mode if mode in ("zh", "en") else detect_lang(event.text)
                    self._on_final(event.text, src_lang)
                    try:
                        self._segments.put_nowait((event.text, time.monotonic()))
                    except queue.Full:
                        self._segments.get_nowait()  # drop oldest, keep newest
                        self._segments.put_nowait((event.text, time.monotonic()))
                        self._on_status("翻译积压，丢弃最旧片段")
                elif fast is None:
                    self._on_partial(event.text)

    def _translation_worker(self) -> None:
        while not self._stop.is_set():
            try:
                text, t_final = self._segments.get(timeout=0.2)
            except queue.Empty:
                continue
            mode = self._cfg.lang_mode
            if mode == "zh":
                dst_lang = "en"
            elif mode == "en":
                dst_lang = "zh"
            else:
                dst_lang = "en" if detect_lang(text) == "zh" else "zh"
            try:
                translation = self._translator.translate(text, target_lang=dst_lang)
            except Exception as e:  # noqa: BLE001 - keep the pipeline alive
                self._on_status(f"翻译失败: {e}")
                continue
            self._on_translation(translation, dst_lang, time.monotonic() - t_final)
            if self._speech is not None and translation:
                try:
                    self._speech.put_nowait((translation, dst_lang))
                except queue.Full:
                    self._on_status("TTS 积压，跳过本段朗读")

    def _tts_worker(self) -> None:
        from .tts_engine import BilingualTts  # heavy import, load in worker

        try:
            tts = BilingualTts(self._cfg.tts, self._cfg.tts_output_device_index)
        except Exception as e:  # noqa: BLE001
            self._on_status(f"TTS 初始化失败，仅字幕模式: {e}")
            while not self._stop.is_set():  # keep draining so queue never blocks
                try:
                    self._speech.get(timeout=0.5)
                except queue.Empty:
                    pass
            return
        self._on_status("TTS 就绪")
        try:
            while not self._stop.is_set():
                try:
                    text, lang = self._speech.get(timeout=0.2)
                except queue.Empty:
                    continue
                gated = self._cfg.mute_capture_during_tts and self._capture is not None
                if gated:
                    self._capture.suppress.set()
                played = False
                try:
                    played = tts.speak(text, lang)
                except Exception as e:  # noqa: BLE001
                    self._on_status(f"TTS 播放失败: {e}")
                finally:
                    if gated:
                        if played:
                            time.sleep(TTS_TAIL_GUARD_S)
                        if self._capture is not None:
                            self._capture.suppress.clear()
        finally:
            tts.close()
