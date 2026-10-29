"""Threaded interpreter pipeline, reusable by console and GUI front-ends.

Events are delivered via callbacks which may be invoked from worker threads;
front-ends must marshal them onto their own UI thread if needed.

Real-time policy (live captions, not a recording):
  * audio: each lane buffers a few seconds; when the ASR falls further
    behind the OLDEST audio is dropped and the ASR's utterance state reset,
    so a sentence is never spliced across the hole (seq gap, see
    audio_capture.AudioChunk);
  * finals carry a `segment_id` so front-ends attach a late translation to
    the right source line even when the next final arrived first;
  * TTS: queue overflow drops the oldest unplayed translation, and anything
    older than `tts_max_age_s` at dequeue or after synthesis is subtitles
    only — the listener needs what is being said now;
  * sessions: every start() is a session; a worker from a previous session
    that was stuck in a model request cannot emit into the new one.
"""
import itertools
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .asr import StreamingAsr, create_asr
from .assistant import MeetingAssistant
from .audio_capture import AudioChunk, AutoGain, KeepAliveOutput, LoopbackCapture, MicCapture
from .config import ASR_ENGLISH, EN_ASR_MODELS, AppConfig
from .echo_cancel import EchoCanceller
from .dialogue import ME, OTHER, DialogueInterpreter
from .meeting_log import (MeetingLog, fmt_assist, fmt_final, fmt_mic_final, fmt_mic_translation,
                          fmt_status, fmt_translation)
from .translator import detect_lang

TTS_TAIL_GUARD_S = 0.4  # keep capture muted briefly after TTS stops
STOP_JOIN_S = 3.0       # stop() waits this long per worker; a worker stuck in a
                        # model request is abandoned (its session is over)

Noop = lambda *a, **k: None  # noqa: E731

_tls = threading.local()  # .session = the _Session a worker thread belongs to


@dataclass(frozen=True)
class _Session:
    """One start()..stop() lifetime. Workers hold theirs; callbacks from a
    worker whose session is no longer the pipeline's current one are dropped."""

    id: int
    stop: threading.Event


class SeqTracker:
    """Detects holes in a lane's capture timeline from AudioChunk.seq."""

    def __init__(self, chunk_s: float):
        self._chunk_s = chunk_s
        self._last: int | None = None

    def gap_s(self, chunk: AudioChunk) -> float:
        """Seconds of audio missing between the previous chunk and this one
        (0.0 when they are consecutive, or for the first chunk)."""
        missing = 0 if self._last is None else chunk.seq - self._last - 1
        self._last = chunk.seq
        return max(0, missing) * self._chunk_s


def put_newest(q: "queue.Queue", item, on_drop: Callable[[], None] = Noop) -> None:
    """Enqueue for a real-time consumer: when full, drop the OLDEST item."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)
        on_drop()


class InterpreterPipeline:
    """Capture -> ASR -> translation -> optional TTS, all on worker threads."""

    def __init__(
        self,
        cfg: AppConfig,
        on_partial: Callable[[str], None] = Noop,
        on_final: Callable[[str, str, int], None] = Noop,          # (text, lang, segment_id)
        # (translation, lang, latency_s, corrected_source, raw_source, segment_id)
        on_translation: Callable[[str, str, float, str, str, int], None] = Noop,
        on_provisional: Callable[[str, str], None] = Noop,    # (text, lang)
        on_assist: Callable[[str, float, bool], None] = Noop,  # (analysis, latency_s, is_final)
        on_mic_partial: Callable[[str], None] = Noop,
        on_mic_final: Callable[[str, str, int], None] = Noop,       # (text, lang, segment_id)
        on_mic_translation: Callable[[str, str, float, str, str, int], None] = Noop,
        on_status: Callable[[str], None] = Noop,
    ):
        self._cfg = cfg
        self._spec_text = ""  # newest partial from the accurate engine
        self._threads: list[threading.Thread] = []
        self._capture: LoopbackCapture | None = None
        self._mic: MicCapture | None = None
        self._session: _Session | None = None
        self._session_ids = itertools.count(1)
        self._segment_ids = itertools.count(1)  # shared by both lanes: unique per pipeline
        self.running = False
        # Real-time health counters (read by front-ends / tests; status lines
        # report them as they change).
        self.audio_gaps = 0
        self.tts_skipped = 0
        self.tts_lag_s = 0.0  # playback start - utterance finalized, last spoken segment
        # Per-session transcript (logs/meeting-*.log) for post-meeting review:
        # tee the lane callbacks into it. Opened in start(), closed in stop().
        # The session guard wraps outermost so a stale worker neither logs nor
        # reaches the front-end.
        self._mlog = MeetingLog(cfg.meeting_log_dir)
        self._on_partial = self._guard(on_partial)
        self._on_final = self._guard(self._mlog.tee(on_final, fmt_final))
        self._on_translation = self._guard(self._mlog.tee(on_translation, fmt_translation))
        self._on_provisional = self._guard(on_provisional)
        self._on_assist = self._guard(self._mlog.tee(on_assist, fmt_assist))
        self._on_mic_partial = self._guard(on_mic_partial)
        self._on_mic_final = self._guard(self._mlog.tee(on_mic_final, fmt_mic_final))
        self._on_mic_translation = self._guard(
            self._mlog.tee(on_mic_translation, fmt_mic_translation)
        )
        self._on_status = self._guard(self._mlog.tee(on_status, fmt_status))

    # -- session plumbing ---------------------------------------------------

    def _guard(self, callback: Callable) -> Callable:
        """Drop events from a worker whose session is over. Calls from
        control threads (start/stop/check_backend) carry no session and pass."""

        def wrapped(*args):
            sess = getattr(_tls, "session", None)
            if sess is not None and not self._live(sess):
                return None
            return callback(*args)

        return wrapped

    def _live(self, sess: _Session) -> bool:
        return sess is self._session and not sess.stop.is_set()

    def _run(self, sess: _Session, loop: Callable[[_Session], None]) -> None:
        previous = getattr(_tls, "session", None)
        _tls.session = sess
        try:
            loop(sess)
        finally:
            _tls.session = previous

    def _spawn(self, sess: _Session, loop: Callable[[_Session], None]) -> threading.Thread:
        return threading.Thread(target=self._run, args=(sess, loop), daemon=True)

    def check_backend(self) -> str | None:
        """Returns an error message if Ollama/model are unavailable, else None."""
        translator = DialogueInterpreter(self._cfg.translator)
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
        sess = _Session(next(self._session_ids), threading.Event())
        self._session = sess
        self._spec_text = ""
        path = self._mlog.open(
            f"finals={cfg.en_asr_model} preview={cfg.en_asr_fast_model} "
            f"mic={cfg.mic_asr_model} translator={cfg.translator.model}"
        )
        if path is not None:
            self._on_status(f"会议日志: {path}")
        elif self._mlog.error:
            self._on_status(f"会议日志无法创建: {self._mlog.error}")

        # One dialogue-level interpreter shared by both lanes: the other
        # side's turns give the context that fixes MY ASR errors (and vice
        # versa). Pull the LLM into memory now so the first segment is fast.
        self._interp = DialogueInterpreter(cfg.translator)
        threading.Thread(target=self._interp.warm_up, daemon=True).start()
        self._on_status("正在加载 ASR 模型 ...")
        # English source uses a dedicated en model: far better accuracy and
        # endpointing than the Chinese-dominant bilingual model.
        if cfg.lang_mode == "en":
            asr_cfg = EN_ASR_MODELS.get(cfg.en_asr_model, ASR_ENGLISH)
        else:
            asr_cfg = cfg.asr
        self._asr = create_asr(asr_cfg)

        # Optional fast preview engine (dual-ASR): live partials only.
        self._fast_asr = None
        fast_name = cfg.en_asr_fast_model
        if (
            cfg.lang_mode == "en"
            and fast_name
            and fast_name != cfg.en_asr_model
            and fast_name in EN_ASR_MODELS
        ):
            self._fast_asr = create_asr(EN_ASR_MODELS[fast_name])
            self._on_status(f"双引擎: 预览 {fast_name} + 定稿 {cfg.en_asr_model}")

        self._capture = LoopbackCapture(cfg.capture_device_index, queue_s=cfg.audio_queue_s)
        # AEC only makes sense when our TTS plays on the captured endpoint.
        use_aec = cfg.echo_cancel and cfg.tts_output_device_index is None
        self._aec = EchoCanceller(self._capture.sample_rate) if use_aec else None
        self._on_status(f"采集设备: {self._capture.device_name}")

        # Keep the render path active so loopback never drops leading audio.
        try:
            self._keepalive = KeepAliveOutput(self._capture.device_name)
        except Exception as e:  # noqa: BLE001 - degraded but functional without it
            self._keepalive = None
            self._on_status(f"keep-alive 输出流启动失败（开头可能丢音）: {e}")

        # (segment_id, text, t_final)
        self._segments: "queue.Queue[tuple[int, str, float]]" = queue.Queue(maxsize=16)
        # (translation, lang, t_final)
        self._speech: "queue.Queue[tuple[str, str, float]] | None" = (
            queue.Queue(maxsize=8) if cfg.enable_tts else None
        )

        # Meeting-assistant mode: intent + reply hints per finalized segment.
        self._assistant: MeetingAssistant | None = None
        self._assist_q: "queue.Queue[tuple[str, float]] | None" = None
        if cfg.enable_assistant:
            self._assistant = MeetingAssistant(cfg.assistant, cfg.translator)
            self._assist_q = queue.Queue(maxsize=16)
            if cfg.assistant.model and cfg.assistant.model != cfg.translator.model:
                threading.Thread(target=self._assistant.warm_up, daemon=True).start()

        # Microphone lane: the user's own speech -> own ASR -> own translator.
        # Separate translator instance so its rolling history never mixes
        # with the other side's; optional and non-fatal if no mic exists.
        self._mic = None
        self._mic_segments: "queue.Queue[tuple[int, str, float]]" = queue.Queue(maxsize=16)
        if cfg.enable_mic:
            try:
                self._mic_asr = create_asr(EN_ASR_MODELS.get(cfg.mic_asr_model, asr_cfg))
                self._mic = MicCapture(cfg.mic_device_index, queue_s=cfg.audio_queue_s)
                self._on_status(f"麦克风: {self._mic.device_name}")
            except Exception as e:  # noqa: BLE001 - degrade to loopback-only
                self._mic = None
                self._on_status(f"麦克风启动失败，仅监听系统音频: {e}")

        loops = [self._asr_loop, self._translation_worker, self._speculative_worker]
        if self._mic is not None:
            loops += [self._mic_loop, self._mic_translation_worker]
        if self._assist_q is not None:
            loops += [self._assist_worker, self._assist_spec_worker]
        if self._speech is not None:
            loops.append(self._tts_worker)
        self._threads = [self._spawn(sess, loop) for loop in loops]

        self._capture.start()
        if self._mic is not None:
            self._mic.start()
        for t in self._threads:
            t.start()
        self.running = True
        self._on_status(
            "运行中 — 正在监听系统音频 + 麦克风" if self._mic is not None
            else "运行中 — 正在监听系统音频"
        )

    def stop(self) -> None:
        sess = self._session
        if sess is not None:
            sess.stop.set()
        for t in self._threads:
            t.join(timeout=STOP_JOIN_S)
        stuck = [t for t in self._threads if t.is_alive()]
        self._threads = []
        # From here on a worker still blocked in a model request belongs to a
        # dead session: when it returns, _live() is false, so it exits its
        # loop and every callback it makes is dropped by _guard().
        self._session = None
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception as e:  # noqa: BLE001
                self._on_status(f"采集关闭异常: {e}")
            self._capture = None
        if self._mic is not None:
            try:
                self._mic.stop()
            except Exception as e:  # noqa: BLE001
                self._on_status(f"麦克风关闭异常: {e}")
            self._mic = None
        if getattr(self, "_keepalive", None) is not None:
            try:
                self._keepalive.stop()
            except Exception:  # noqa: BLE001
                pass
            self._keepalive = None
        self.running = False
        if stuck:
            self._on_status(f"{len(stuck)} 个工作线程仍在等待模型响应，其结果将被丢弃")
        self._on_status("已停止")
        self._mlog.close()

    # -- worker loops --------------------------------------------------------

    def _set_spec(self, sess: _Session, text: str) -> None:
        """The speculative workers read `_spec_text`, which is shared state:
        a worker from a stopped session that was stuck in a model request
        must not leave its last partial for the new session to translate."""
        if self._live(sess):
            self._spec_text = text

    def _audio_gap(
        self, sess: _Session, lane: str, gap_s: float, asr, fast, aec: EchoCanceller | None
    ) -> None:
        """The lane's capture timeline has a hole: nothing decoded from before
        it may be joined with what comes after."""
        self.audio_gaps += 1
        asr.reset()
        if fast is not None:
            fast.reset()
        if aec is not None:
            aec.reset()
        if lane == OTHER:
            self._set_spec(sess, "")
            self._on_partial("")  # the partial on screen died with the reset
        else:
            self._on_mic_partial("")
        who = "对方" if lane == OTHER else "我方"
        self._on_status(
            f"{who}音频断流 {gap_s:.1f}s（识别落后，已丢弃最旧音频并重置当前句，累计 {self.audio_gaps} 次）"
        )

    def _asr_loop(self, sess: _Session) -> None:
        capture, asr, fast, aec = self._capture, self._asr, self._fast_asr, self._aec
        segments, assist_q = self._segments, self._assist_q
        cfg = self._cfg
        agc = AutoGain()
        seq = SeqTracker(capture.chunk_s)
        # WASAPI loopback delivers no frames while nothing is playing, so feed
        # synthetic silence on timeout to keep endpoint detection moving.
        silence = np.zeros(int(capture.sample_rate * 0.2), dtype=np.float32)
        while self._live(sess):
            item = capture.read()
            if item is None:
                chunk = silence
            else:
                gap = seq.gap_s(item)
                if gap > 0:
                    self._audio_gap(sess, OTHER, gap, asr, fast, aec)
                chunk = item.samples
                if aec is not None and aec.active:
                    chunk = aec.process(chunk)
                chunk = agc.apply(chunk)
            if fast is not None:
                # Preview engine drives the live partial line only; its finals
                # are discarded (accept() still resets its stream internally).
                for event in fast.accept(chunk, capture.sample_rate):
                    if not event.is_final:
                        self._on_partial(event.text)
            for event in asr.accept(chunk, capture.sample_rate):
                if event.is_final:
                    self._set_spec(sess, "")  # utterance closed; stop speculating
                    if len(event.text) < cfg.min_chars_to_translate:
                        continue
                    mode = cfg.lang_mode
                    if mode == "en" and detect_lang(event.text) == "zh":
                        # A mostly-Chinese segment in en mode can only be our
                        # own TTS echo leaking through — never programme audio.
                        continue
                    src_lang = mode if mode in ("zh", "en") else detect_lang(event.text)
                    seg_id = next(self._segment_ids)
                    t_final = time.monotonic()
                    self._on_final(event.text, src_lang, seg_id)
                    put_newest(
                        segments, (seg_id, event.text, t_final),
                        lambda: self._on_status("翻译积压，丢弃最旧片段"),
                    )
                    if assist_q is not None:
                        # The worker coalesces pending segments per LLM call,
                        # so a full queue means the LLM is far behind — drop
                        # oldest, the context window in MeetingAssistant still
                        # records recent history.
                        put_newest(assist_q, (event.text, t_final))
                else:
                    self._set_spec(sess, event.text)
                    if fast is None:
                        self._on_partial(event.text)

    def _mic_loop(self, sess: _Session) -> None:
        """Own-speech lane: mic -> AGC -> mic ASR -> pane + assistant context."""
        capture, asr, segments = self._mic, self._mic_asr, self._mic_segments
        cfg = self._cfg
        agc = AutoGain()
        seq = SeqTracker(capture.chunk_s)
        # Frames are dropped while TTS plays (suppress); feed silence so the
        # endpoint detector still closes an utterance cut off by playback.
        silence = np.zeros(int(capture.sample_rate * 0.2), dtype=np.float32)
        while self._live(sess):
            item = capture.read()
            if item is None:
                chunk = silence
            else:
                gap = seq.gap_s(item)
                if gap > 0:
                    self._audio_gap(sess, ME, gap, asr, None, None)
                chunk = agc.apply(item.samples)
            for event in asr.accept(chunk, capture.sample_rate):
                if not event.is_final:
                    self._on_mic_partial(event.text)
                    continue
                if len(event.text) < cfg.min_chars_to_translate:
                    continue
                seg_id = next(self._segment_ids)
                t_final = time.monotonic()
                self._on_mic_final(event.text, detect_lang(event.text), seg_id)
                # Assistant context is fed from the translation worker with
                # the CORRECTED text ("SWMS", not "swims").
                put_newest(
                    segments, (seg_id, event.text, t_final),
                    lambda: self._on_status("我方翻译积压，丢弃最旧片段"),
                )

    def _report_order(self, interp: DialogueInterpreter, before: int) -> None:
        if interp.out_of_order != before:
            self._on_status(f"对话上下文乱序（两路定稿先后与发言顺序不符，累计 {interp.out_of_order} 次）")

    def _mic_translation_worker(self, sess: _Session) -> None:
        """Own speech through the shared interpreter (corrected + translated)."""
        segments, interp = self._mic_segments, self._interp
        assistant = self._assistant
        while self._live(sess):
            try:
                seg_id, text, t_final = segments.get(timeout=0.2)
            except queue.Empty:
                continue
            before = interp.out_of_order
            try:
                res = interp.interpret(text, ME, t_final)
            except Exception as e:  # noqa: BLE001 - keep the pipeline alive
                self._on_status(f"我方翻译失败: {e}")
                if assistant is not None:
                    assistant.record_own(text)  # raw beats nothing
                continue
            self._report_order(interp, before)
            if assistant is not None:
                assistant.record_own(res.corrected)
            self._on_mic_translation(
                res.translation, res.target_lang, time.monotonic() - t_final,
                res.corrected, text, seg_id,
            )

    def _speculative_worker(self, sess: _Session) -> None:
        """Provisional translation of the growing partial (speculative view).

        Only ever translates the newest snapshot; superseded partials are
        skipped, so this self-throttles to the LLM's own speed.
        """
        interp = self._interp
        last = ""
        while self._live(sess):
            text = self._spec_text
            if text == last or len(text) < 8:
                time.sleep(0.1)
                continue
            dst_lang = "en" if detect_lang(text) == "zh" else "zh"
            try:
                provisional = interp.translate_partial(text, OTHER)
            except Exception:  # noqa: BLE001 - provisional output is best-effort
                time.sleep(0.5)
                continue
            last = text
            # Only show it if this partial is still the current one.
            if self._spec_text == text and provisional:
                self._on_provisional(provisional, dst_lang)

    def _translation_worker(self, sess: _Session) -> None:
        segments, speech, interp = self._segments, self._speech, self._interp
        while self._live(sess):
            try:
                seg_id, text, t_final = segments.get(timeout=0.2)
            except queue.Empty:
                continue
            before = interp.out_of_order
            try:
                res = interp.interpret(text, OTHER, t_final)
            except Exception as e:  # noqa: BLE001 - keep the pipeline alive
                self._on_status(f"翻译失败: {e}")
                continue
            self._report_order(interp, before)
            self._on_translation(
                res.translation, res.target_lang, time.monotonic() - t_final,
                res.corrected, text, seg_id,
            )
            if speech is not None and res.translation:
                # Newest wins: a listener who is behind needs the current
                # sentence, not the one from four sentences ago.
                put_newest(
                    speech, (res.translation, res.target_lang, t_final),
                    lambda: self._on_status("TTS 积压，丢弃最旧未播放片段"),
                )

    def _assist_worker(self, sess: _Session) -> None:
        """Analyze finalized segments: speaker intent + reply-direction hints.

        Segments that queued up while the previous analysis ran are coalesced
        into one utterance, so the assistant self-throttles to the LLM's speed
        instead of falling ever further behind the meeting.
        """
        q, assistant = self._assist_q, self._assistant
        while self._live(sess):
            try:
                text, t0 = q.get(timeout=0.2)
            except queue.Empty:
                continue
            batch = [text]
            try:
                while True:
                    more, t0 = q.get_nowait()
                    batch.append(more)
            except queue.Empty:
                pass
            try:
                analysis = assistant.analyze(batch)
            except Exception as e:  # noqa: BLE001 - keep the pipeline alive
                self._on_status(f"会议助手分析失败: {e}")
                continue
            if analysis:
                self._on_assist(analysis, time.monotonic() - t0, True)

    def _assist_spec_worker(self, sess: _Session) -> None:
        """Speculative assist: analyze the growing partial before it finalizes.

        Same pattern as _speculative_worker — only ever the newest snapshot,
        so it self-throttles to the LLM's speed. By the time the speaker
        finishes the question, a provisional hint is usually already visible;
        the authoritative analysis replaces it a moment later.
        """
        assistant = self._assistant
        last = ""
        while self._live(sess):
            text = self._spec_text
            if text == last or len(text) < 15:
                time.sleep(0.15)
                continue
            try:
                hint = assistant.analyze_partial(text)
            except Exception:  # noqa: BLE001 - provisional output is best-effort
                time.sleep(0.5)
                continue
            last = text
            # Only show it if this partial is still the current utterance.
            if self._spec_text == text and hint:
                self._on_assist(hint, 0.0, False)

    def _tts_expired(self, t_final: float, stage: str) -> bool:
        """True (and counted) when a translation is too old to be worth
        speaking: the subtitle already said it, the listener has moved on."""
        age = time.monotonic() - t_final
        if age <= self._cfg.tts_max_age_s:
            return False
        self.tts_skipped += 1
        self._on_status(f"跳过过期朗读（{stage}，落后 {age:.1f}s，累计 {self.tts_skipped} 段）")
        return True

    def _tts_worker(self, sess: _Session) -> None:
        from .tts_engine import BilingualTts  # heavy import, load in worker

        speech, capture, aec, cfg = self._speech, self._capture, self._aec, self._cfg
        # The mic hears our TTS through the speakers — unless the user wears
        # a headset, in which case muting it only loses their own words.
        mic = self._mic if cfg.mute_mic_during_tts else None
        try:
            tts = BilingualTts(cfg.tts, cfg.tts_output_device_index)
        except Exception as e:  # noqa: BLE001
            self._on_status(f"TTS 初始化失败，仅字幕模式: {e}")
            while self._live(sess):  # keep draining so queue never blocks
                try:
                    speech.get(timeout=0.5)
                except queue.Empty:
                    pass
            return
        self._on_status("TTS 就绪")
        gated = aec is None and cfg.mute_capture_during_tts and capture is not None
        try:
            while self._live(sess):
                try:
                    text, lang, t_final = speech.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._tts_expired(t_final, "出队"):
                    continue
                played = muted = False
                try:
                    samples, sr = tts.synthesize(text, lang)
                    # Synthesis takes a few hundred ms during which nothing is
                    # audible, so capture stays live for it; also the last
                    # chance to drop a translation the meeting has outrun.
                    if self._tts_expired(t_final, "合成后"):
                        continue
                    if aec is not None and samples.size:
                        aec.set_reference(samples, sr)
                    if gated:
                        capture.suppress.set()
                    if mic is not None:
                        mic.suppress.set()
                    muted = True
                    self.tts_lag_s = time.monotonic() - t_final
                    played = tts.play(samples, sr)
                except Exception as e:  # noqa: BLE001
                    self._on_status(f"TTS 播放失败: {e}")
                finally:
                    if played and (gated or mic is not None):
                        time.sleep(TTS_TAIL_GUARD_S)  # playback buffer tail
                    if muted:
                        if gated:
                            capture.suppress.clear()
                        if mic is not None:
                            mic.suppress.clear()
        finally:
            tts.close()
