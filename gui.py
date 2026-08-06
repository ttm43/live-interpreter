"""Tkinter GUI front-end for the live interpreter (en->zh, system audio)."""
from interpreter.bootstrap import ensure_deps

ensure_deps()  # allow running with bare system python (deps live in .venv)

import dataclasses
import queue
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import requests

from interpreter.audio_capture import get_loopback_devices
from interpreter.config import EN_ASR_MODELS, AppConfig, TranslatorConfig
from interpreter.pipeline import InterpreterPipeline

# -- theme -------------------------------------------------------------------
BG = "#1f2126"
BG_PANEL = "#2a2d34"
FG = "#e8eaed"
FG_DIM = "#9aa0a6"
ACCENT = "#7cc4ff"
GREEN = "#7ee2a8"
RED = "#ff8a80"
FONT = ("Microsoft YaHei UI", 11)
FONT_SMALL = ("Microsoft YaHei UI", 9)

DEFAULT_DEVICE_LABEL = "（默认扬声器）"


class InterpreterGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.pipeline: InterpreterPipeline | None = None
        self.events: "queue.Queue[tuple[str, tuple]]" = queue.Queue()
        self._busy = False
        self._build_ui()
        self.root.after(50, self._drain_events)

    # -- UI construction ------------------------------------------------------

    def _installed_ollama_models(self) -> list[str]:
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
            names = sorted(m["name"] for m in r.json().get("models", []))
            return names or [TranslatorConfig().model]
        except requests.RequestException:
            return [TranslatorConfig().model]

    def _build_ui(self) -> None:
        self.root.title("英中同传 · Live Interpreter")
        self.root.geometry("1240x560")
        self.root.minsize(560, 380)
        self.root.configure(bg=BG)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG_DIM, font=FONT_SMALL)
        style.configure(
            "TCheckbutton", background=BG, foreground=FG, font=FONT_SMALL,
            indicatorcolor=BG_PANEL,
        )
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure(
            "TCombobox", fieldbackground=BG_PANEL, background=BG_PANEL,
            foreground=FG, arrowcolor=FG, borderwidth=0,
        )

        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")

        self.start_btn = tk.Button(
            bar, text="▶  开始", command=self._toggle, width=10,
            bg=GREEN, fg="#1b1b1b", activebackground="#a5f0c5",
            font=("Microsoft YaHei UI", 10, "bold"), relief="flat", cursor="hand2",
        )
        self.start_btn.pack(side="left")

        self.tts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="朗读译文", variable=self.tts_var).pack(
            side="left", padx=(12, 0)
        )

        # 语向暂时固定为 英文 -> 中文（先把这一个方向打磨好）。
        # 恢复三模式时把下面这段解开，并还原 _do_start / _ev_* 里的 mode_box。
        # self.mode_map = {"自动（中英混合）": "auto", "中文 → 英文": "zh", "英文 → 中文": "en"}
        # self.mode_var = tk.StringVar(value="自动（中英混合）")
        # self.mode_box = ttk.Combobox(...)
        ttk.Label(bar, text="EN → 中").pack(side="left", padx=(12, 0))

        ttk.Label(bar, text="识别:").pack(side="left", padx=(16, 4))
        self.asr_var = tk.StringVar(value=AppConfig().en_asr_model)
        self.asr_box = ttk.Combobox(
            bar, textvariable=self.asr_var, state="readonly", width=16,
            values=list(EN_ASR_MODELS),
        )
        self.asr_box.pack(side="left")

        ttk.Label(bar, text="预览:").pack(side="left", padx=(16, 4))
        self.fast_var = tk.StringVar(value=AppConfig().en_asr_fast_model or "关闭")
        self.fast_box = ttk.Combobox(
            bar, textvariable=self.fast_var, state="readonly", width=14,
            values=["关闭"] + list(EN_ASR_MODELS),
        )
        self.fast_box.pack(side="left")

        ttk.Label(bar, text="翻译:").pack(side="left", padx=(16, 4))
        self.llm_var = tk.StringVar(value=TranslatorConfig().model)
        self.llm_box = ttk.Combobox(
            bar, textvariable=self.llm_var, state="readonly", width=24,
            values=self._installed_ollama_models(),
        )
        self.llm_box.pack(side="left")

        ttk.Label(bar, text="采集:").pack(side="left", padx=(16, 4))
        self.device_map: dict[str, int | None] = {DEFAULT_DEVICE_LABEL: None}
        try:
            for idx, name in get_loopback_devices():
                self.device_map[f"[{idx}] {name}"] = idx
        except Exception:  # noqa: BLE001 - device list is a convenience only
            pass
        self.device_var = tk.StringVar(value=DEFAULT_DEVICE_LABEL)
        self.device_box = ttk.Combobox(
            bar, textvariable=self.device_var, state="readonly", width=22,
            values=list(self.device_map),
        )
        self.device_box.pack(side="left")

        tk.Button(
            bar, text="清空", command=self._clear_transcript, width=6,
            bg=BG_PANEL, fg=FG, activebackground="#3a3e47", activeforeground=FG,
            font=FONT_SMALL, relief="flat", cursor="hand2",
        ).pack(side="right")
        tk.Button(
            bar, text="词表", command=self._open_glossary, width=6,
            bg=BG_PANEL, fg=FG, activebackground="#3a3e47", activeforeground=FG,
            font=FONT_SMALL, relief="flat", cursor="hand2",
        ).pack(side="right", padx=(0, 6))

        self.text = ScrolledText(
            self.root, bg=BG_PANEL, fg=FG, insertbackground=FG, wrap="word",
            font=FONT, relief="flat", padx=12, pady=10, state="disabled",
        )
        self.text.pack(fill="both", expand=True, padx=10)
        self.text.tag_configure("src", foreground=FG)
        self.text.tag_configure("dst", foreground=ACCENT)
        self.text.tag_configure("meta", foreground=FG_DIM, font=FONT_SMALL)
        self.text.tag_configure("live", foreground=FG_DIM, font=(FONT[0], 11, "italic"))
        self.text.tag_configure("live_dst", foreground="#5b87b0", font=(FONT[0], 11, "italic"))
        self._partial_active = False
        self._live_src = ""   # speculative view: growing source partial
        self._live_dst = ""   # speculative view: provisional translation

        bottom = ttk.Frame(self.root, padding=(12, 6))
        bottom.pack(fill="x")
        self.partial_var = tk.StringVar(value="")
        tk.Label(
            bottom, textvariable=self.partial_var, bg=BG, fg=FG_DIM,
            font=(FONT[0], 10, "italic"), anchor="w",
        ).pack(fill="x")
        self.status_var = tk.StringVar(value="就绪 — 点击「开始」后播放任何声音即可翻译")
        tk.Label(
            bottom, textvariable=self.status_var, bg=BG, fg=FG_DIM,
            font=FONT_SMALL, anchor="w",
        ).pack(fill="x")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- pipeline control (worker threads keep the UI responsive) -------------

    def _toggle(self) -> None:
        if self._busy:
            return
        if self.pipeline and self.pipeline.running:
            self._set_busy("正在停止 ...")
            threading.Thread(target=self._do_stop, daemon=True).start()
        else:
            self._set_busy("正在启动 ...")
            threading.Thread(target=self._do_start, daemon=True).start()

    def _do_start(self) -> None:
        from interpreter.bootstrap import ensure_ollama

        self.events.put(("status", ("正在检查 Ollama ...",)))
        ensure_ollama()  # auto-start the bundled Ollama when needed
        fast = self.fast_var.get()
        cfg = AppConfig(
            lang_mode="en",
            en_asr_model=self.asr_var.get(),
            en_asr_fast_model="" if fast == "关闭" else fast,
            enable_tts=self.tts_var.get(),
            capture_device_index=self.device_map.get(self.device_var.get()),
            translator=dataclasses.replace(TranslatorConfig(), model=self.llm_var.get()),
        )
        pipeline = InterpreterPipeline(
            cfg,
            on_partial=lambda t: self.events.put(("partial", (t,))),
            on_final=lambda t, lang: self.events.put(("final", (t, lang))),
            on_translation=lambda t, lang, s: self.events.put(("translation", (t, lang, s))),
            on_provisional=lambda t, lang: self.events.put(("provisional", (t, lang))),
            on_status=lambda m: self.events.put(("status", (m,))),
        )
        error = pipeline.check_backend()
        if error is None:
            try:
                pipeline.start()
                self.pipeline = pipeline
            except Exception as e:  # noqa: BLE001
                error = f"启动失败: {e}"
        self.events.put(("started", (error,)))

    def _do_stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()
        self.events.put(("stopped", ()))

    def _set_busy(self, msg: str) -> None:
        self._busy = True
        self.start_btn.configure(state="disabled")
        self.status_var.set(msg)

    def _on_close(self) -> None:
        if self.pipeline and self.pipeline.running:
            self.pipeline.stop()
        self.root.destroy()

    # -- event marshalling (worker threads -> Tk main thread) -----------------

    def _drain_events(self) -> None:
        try:
            while True:
                kind, args = self.events.get_nowait()
                getattr(self, f"_ev_{kind}")(*args)
        except queue.Empty:
            pass
        self.root.after(50, self._drain_events)

    def _ev_started(self, error: str | None) -> None:
        self._busy = False
        self.start_btn.configure(state="normal")
        if error:
            self.status_var.set(f"⚠ {error}")
            return
        self.start_btn.configure(text="■  停止", bg=RED, activebackground="#ffb3ab")
        for box in (self.device_box, self.asr_box, self.fast_box, self.llm_box):
            box.configure(state="disabled")

    def _ev_stopped(self) -> None:
        self._busy = False
        self.pipeline = None
        self.start_btn.configure(
            state="normal", text="▶  开始", bg=GREEN, activebackground="#a5f0c5"
        )
        for box in (self.device_box, self.asr_box, self.fast_box, self.llm_box):
            box.configure(state="readonly")
        self.partial_var.set("")
        self.status_var.set("已停止")

    def _ev_partial(self, text: str) -> None:
        self._live_src = text
        self._render_live()

    def _ev_provisional(self, text: str, lang: str) -> None:
        self._live_dst = text
        self._render_live()

    def _render_live(self) -> None:
        """Redraw the speculative two-line block (source + provisional)."""
        self.text.configure(state="normal")
        self._remove_partial()
        if self._live_src or self._live_dst:
            self.text.mark_set("partial_start", "end-1c")
            self.text.mark_gravity("partial_start", "left")
            if self._live_src:
                self.text.insert("end", f"… {self._live_src}", "live")
            if self._live_dst:
                prefix = "\n" if self._live_src else ""
                self.text.insert("end", f"{prefix}⇢ {self._live_dst}", "live_dst")
            self._partial_active = True
        self.text.see("end")
        self.text.configure(state="disabled")

    def _remove_partial(self) -> None:
        if self._partial_active:
            self.text.delete("partial_start", "end")
            self._partial_active = False

    def _ev_final(self, text: str, lang: str) -> None:
        # The live block is this utterance's stale preview — drop it; the
        # authoritative translation follows within a beat.
        self._live_src = ""
        self._live_dst = ""
        self._append(f"[{lang}] {text}\n", "src")

    def _ev_translation(self, text: str, lang: str, latency_s: float) -> None:
        self._append(f"      → {text} ", "dst")
        self._append(f"({latency_s:.1f}s)\n\n", "meta")

    def _ev_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def _open_glossary(self) -> None:
        """Open glossary.txt in the default editor; edits apply on next segment."""
        import os

        from interpreter.glossary import GLOSSARY_PATH

        try:
            os.startfile(str(GLOSSARY_PATH))
            self.status_var.set("词表已打开 — 保存后下一段翻译立即生效")
        except OSError as e:
            self.status_var.set(f"打开词表失败: {e}")

    def _clear_transcript(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self._partial_active = False
        self._live_src = ""
        self._live_dst = ""
        self.text.configure(state="disabled")
        self.partial_var.set("")

    MAX_LINES = 600  # keep long sessions from growing the widget unboundedly

    def _append(self, text: str, tag: str) -> None:
        """Append finalized content, then re-draw the live block below it."""
        self.text.configure(state="normal")
        if int(self.text.index("end-1c").split(".")[0]) > self.MAX_LINES:
            self.text.delete("1.0", f"{self.MAX_LINES // 3}.0")
        self._remove_partial()
        self.text.insert("end", text, tag)
        self.text.configure(state="disabled")
        self._render_live()


def main() -> None:
    root = tk.Tk()
    InterpreterGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
