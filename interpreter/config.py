"""Central configuration for the live interpreter pipeline."""
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

ASR_MODEL_DIR = MODELS_DIR / "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
ASR_EN_MODEL_DIR = MODELS_DIR / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
TTS_MODEL_DIR = MODELS_DIR / "kokoro-multi-lang-v1_1"


@dataclass(frozen=True)
class AsrConfig:
    kind: str = "paraformer"  # "paraformer" (zh/bilingual) or "zipformer" (en)
    encoder: str = str(ASR_MODEL_DIR / "encoder.int8.onnx")
    decoder: str = str(ASR_MODEL_DIR / "decoder.int8.onnx")
    joiner: str = ""  # zipformer transducer only
    tokens: str = str(ASR_MODEL_DIR / "tokens.txt")
    num_threads: int = 4
    sample_rate: int = 16000
    # Endpoint rules (seconds of trailing silence that finalize a segment)
    rule1_min_trailing_silence: float = 2.4   # silence with no decoded text
    rule2_min_trailing_silence: float = 0.9   # silence after some decoded text
    rule3_min_utterance_length: float = 18.0  # hard cap on utterance length


# Chinese-dominant bilingual model: use for zh->en and mixed/auto mode.
ASR_BILINGUAL = AsrConfig()

# Dedicated English model: much better English accuracy and endpointing.
ASR_ENGLISH = AsrConfig(
    kind="zipformer",
    encoder=str(ASR_EN_MODEL_DIR / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
    decoder=str(ASR_EN_MODEL_DIR / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
    joiner=str(ASR_EN_MODEL_DIR / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
    tokens=str(ASR_EN_MODEL_DIR / "tokens.txt"),
)


def _nemo_en(dirname: str) -> AsrConfig:
    d = MODELS_DIR / dirname
    return AsrConfig(
        kind="nemo_transducer",
        encoder=str(d / "encoder.onnx"),
        decoder=str(d / "decoder.onnx"),
        joiner=str(d / "joiner.onnx"),
        tokens=str(d / "tokens.txt"),
    )


def _nemotron(dirname: str) -> AsrConfig:
    d = MODELS_DIR / dirname
    return AsrConfig(
        kind="nemo_transducer",
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"),
        tokens=str(d / "tokens.txt"),
    )


# Candidate English streaming models, selectable in the GUI for comparison.
# nemotron-3.5 (2026-06) adds punctuation + capitalization; suffix = chunk
# size (latency vs accuracy). nemo-* are the 2023 FastConformer generation.
EN_ASR_MODELS: dict[str, AsrConfig] = {
    "zipformer-2023": ASR_ENGLISH,
    "nemo-80ms": _nemo_en("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms"),
    "nemo-480ms": _nemo_en("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms"),
    "nemo-1040ms": _nemo_en("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms"),
    # nemotron 80ms/160ms tiers excluded: RTF 0.4-0.9 on CPU with the worst
    # accuracy of all candidates (bench 2026-07-18) — small chunks ruin both.
    "nemotron3.5-320ms": _nemotron("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11"),
    "nemotron3.5-560ms": _nemotron("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"),
    "nemotron3.5-1120ms": _nemotron("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11"),
}


@dataclass(frozen=True)
class TranslatorConfig:
    base_url: str = "http://127.0.0.1:11434"
    # bench_translate.py verdict: best quality/size/speed balance; fixes ASR
    # errors (clod->Claude). qwen3:14b = higher quality; hy-mt2 1.8b = low-RAM.
    model: str = "qwen3:4b-instruct"
    # Optional per-direction overrides; empty string = use `model`.
    model_zh2en: str = ""
    model_en2zh: str = ""
    # Speculative (provisional) translation of unfinished partials. Empty
    # string = use `model`. Point this at a smaller model (e.g.
    # "kaelri/hy-mt2:1.8b-q8_0") for a two-tier draft+final setup.
    spec_model: str = ""
    temperature: float = 0.2
    timeout_s: float = 30.0
    keep_alive: str = "30m"
    history_size: int = 4  # previous segment pairs kept as context


@dataclass(frozen=True)
class TtsConfig:
    model: str = str(TTS_MODEL_DIR / "model.onnx")
    voices: str = str(TTS_MODEL_DIR / "voices.bin")
    tokens: str = str(TTS_MODEL_DIR / "tokens.txt")
    data_dir: str = str(TTS_MODEL_DIR / "espeak-ng-data")
    dict_dir: str = str(TTS_MODEL_DIR / "dict")
    lexicon: str = ",".join([
        str(TTS_MODEL_DIR / "lexicon-us-en.txt"),
        str(TTS_MODEL_DIR / "lexicon-zh.txt"),
    ])
    num_threads: int = 4
    en_speaker_id: int = 0   # af_maple (English female)
    zh_speaker_id: int = 3   # first Chinese female voice
    speed: float = 1.1


@dataclass(frozen=True)
class AppConfig:
    # "auto" = bilingual ASR + per-segment direction detection;
    # "zh" = Chinese source (bilingual ASR), always translate to English;
    # "en" = English source (dedicated en ASR), always translate to Chinese.
    # NOTE: currently focused on en->zh only; auto/zh are parked until the
    # en->zh path is fully tuned (see README).
    lang_mode: str = "en"
    # Real-audio bench 2026-07-18: nemotron3.5-1120ms wins 3/4 clips and adds
    # punctuation + capitalization (better input for translation). Caveat: on
    # heavily-accented speech nemo-1040ms is still stronger (multilingual
    # nemotron can hallucinate foreign scripts there).
    en_asr_model: str = "nemotron3.5-1120ms"
    # Dual-engine: a second fast ASR renders the live partial line (word-by-
    # word feel) while en_asr_model produces the accurate finals that feed
    # translation. Empty string disables the preview engine.
    en_asr_fast_model: str = "nemo-80ms"
    asr: AsrConfig = field(default_factory=AsrConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    enable_tts: bool = True
    mute_capture_during_tts: bool = True
    # Subtract our own TTS from the capture (digital AEC) instead of gating
    # it. Falls back to a mute equivalent when alignment fails. Only applies
    # when TTS plays on the captured (default) device.
    echo_cancel: bool = True
    min_chars_to_translate: int = 2
    capture_device_index: int | None = None  # None = default speakers loopback
    tts_output_device_index: int | None = None  # None = default output
