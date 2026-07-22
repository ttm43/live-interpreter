"""Console front-end for the live interpreter (zh<->en, system audio).

Pipeline: WASAPI loopback capture -> streaming bilingual ASR (sherpa-onnx)
-> local LLM translation (Ollama) -> bilingual TTS playback (Kokoro).
"""
import argparse
import dataclasses
import sys
import threading

from interpreter.bootstrap import ensure_deps

ensure_deps()  # allow running with bare system python (deps live in .venv)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from interpreter.audio_capture import list_devices
from interpreter.config import AppConfig, TranslatorConfig
from interpreter.display import ConsoleDisplay
from interpreter.pipeline import InterpreterPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time zh<->en interpreter for system audio")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--no-tts", action="store_true", help="subtitles only, no spoken output")
    p.add_argument(
        "--lang", choices=["auto", "zh", "en"], default="auto",
        help="source language: zh=Chinese->English, en=English->Chinese (better en ASR)",
    )
    p.add_argument("--capture-device", type=int, default=None, help="loopback device index")
    p.add_argument("--tts-device", type=int, default=None, help="output device index for TTS")
    p.add_argument("--model", type=str, default=None, help="Ollama model name (default qwen3:8b)")
    p.add_argument(
        "--no-mute-during-tts", action="store_true",
        help="keep capturing while TTS speaks (use when TTS plays on another device)",
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig(
        lang_mode=args.lang,
        enable_tts=not args.no_tts,
        mute_capture_during_tts=not args.no_mute_during_tts,
        capture_device_index=args.capture_device,
        tts_output_device_index=args.tts_device,
    )
    if args.model:
        cfg = dataclasses.replace(
            cfg, translator=dataclasses.replace(TranslatorConfig(), model=args.model)
        )
    return cfg


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print(list_devices())
        return

    cfg = build_config(args)
    display = ConsoleDisplay()

    from interpreter.bootstrap import ensure_ollama

    if not ensure_ollama(cfg.translator.base_url):
        display.info("[warn] Ollama 未运行且自动启动失败")

    pipeline = InterpreterPipeline(
        cfg,
        on_partial=display.partial,
        on_final=display.final_source,
        on_translation=display.translation,
        on_status=lambda msg: display.info(f"[info] {msg}"),
    )
    error = pipeline.check_backend()
    if error:
        display.info(f"[error] {error}")
        return

    pipeline.start()
    display.info("[ready] interpreting live audio — Ctrl+C to quit\n")
    try:
        threading.Event().wait()  # workers do everything; just wait for Ctrl+C
    except KeyboardInterrupt:
        display.info("\n[exit] shutting down ...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
