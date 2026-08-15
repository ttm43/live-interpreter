# Live Interpreter — real-time speech translation for system audio

**English** | [中文](README.zh-CN.md)

Captures whatever your computer is playing (WASAPI loopback) and interprets it
in real time — English speech in, live subtitles + Chinese translation out,
optionally spoken aloud via TTS.

Fully local. No cloud APIs, no WSL, no PyTorch, zero compilation on Windows.

## Architecture

```
System audio (speaker loopback, PyAudioWPatch, auto-gain)
   → Streaming ASR (sherpa-onnx, endpoint detection; dual-engine, see below)
   → Translation (Ollama, local GPU/CPU, glossary-aware, fixes ASR errors)
   → Bilingual TTS (sherpa-onnx kokoro-multi-lang-v1_1) → speakers
```

- **Currently focused on EN → ZH** (auto / ZH → EN modes are parked; the code
  remains in comments in `config.py` / `gui.py`).
- **Switchable ASR models** (GUI "识别" dropdown). Benchmarked on real YouTube
  clips (`bench_asr.py`, pseudo-reference = offline Parakeet-TDT-0.6B-v3):

  | Model | Avg WER on real clips | Caption update interval | Role |
  |---|---|---|---|
  | parakeet-semi | **best accuracy overall** (LibriSpeech 0%/2.1%; heavy accents ~4x better than any streaming engine) | ~850ms whole-sentence revisions | **Default** finals engine (word-by-word feel comes from the preview engine); RTF ≈0.4 |
  | whisper-semi | same tier as parakeet-semi; best names ("Mikhail Fedorov") + digits ("35") + cleans disfluencies | ~5-7s revisions | faster-whisper large-v3-turbo int8. **CPU RTF 0.8-1.3 — can't hold real-time**; also hallucinates ("Thank you.") on music/silence. Kept for comparison |
  | nemotron3.5-1120ms | **≈23% (best streaming on news / keynote / podcast)** | ~1.3s | Streaming alternative; native punctuation + casing, 5x lower CPU |
  | nemo-1040ms | ≈28%; **still best on heavy accents** | ~1.2s | Use for accented speakers |
  | nemotron3.5-320ms | ≈31% | ~500ms | Low-latency compromise |
  | nemo-80ms | ≈37% | **~330ms** | Word-by-word feel |
  | nemo-480ms | ≈39% | ~700ms | Middle ground (poor with music) |
  | zipformer-2023 | ≈41% (worst on real audio) | ~360ms | Only strong on LibriSpeech-style read speech |

  Notes: nemotron3.5 is multilingual — heavy accents can trigger language
  confusion (foreign-script fragments). Its 80ms/160ms tiers were dropped:
  RTF 0.4–0.9 on CPU *and* the worst accuracy (small chunks lose both ways).
  The 560ms tier (≈28%, punctuated) is a decent single-engine compromise.

- **Dual-engine ASR (on by default)**: a fast *preview* engine (nemo-80ms,
  ~240ms word-by-word updates) drives only the gray live-caption line, while
  the *final* engine (nemotron3.5-1120ms) produces the accurate, punctuated
  segments that feed translation. No single streaming model is both fast and
  accurate — the pair gets you both at a combined RTF of ≈0.25. The GUI
  "预览" dropdown switches or disables the preview engine.

- **Switchable translation models** (GUI "翻译" dropdown lists everything
  installed in Ollama; compare with `bench_translate.py`, which includes
  noisy ASR-style inputs). Verdicts from our bench:

  | Model | Size | Role |
  |---|---|---|
  | qwen3:4b-instruct | 2.5G | **Default**: best quality/speed/size balance; fixes ASR errors (clod→Claude) |
  | qwen3:14b | 9.3G | Highest quality (numbers, names, terms all correct) if you have the VRAM |
  | kaelri/hy-mt2:1.8b-q8_0 | 2.0G | Translation-specialized (Tencent Hy-MT2); best pick for low-RAM machines |
  | qwen3:8b | 5.2G | Unreliable numbers ("four billion" mistranslated twice) — not recommended |
  | demonbyron/HY-MT1.5-7B | 4.6G | Previous-gen MT model, superseded by 4b-instruct / Hy-MT2 |

  Hunyuan/HY-MT models automatically get their official translation prompt and
  terminology-intervention format (detected by model name in `translator.py`).
  Warning: the HY-MT1.5-**1.8B** GGUFs produce corrupted output under Ollama
  (template echo / hallucination) — use Hy-MT2 for a small MT model.

- **Speculative translation** (inspired by GPT-Live's speculative/authoritative
  dual view): while a sentence is still being spoken, the growing partial is
  provisionally translated and shown as a dim blue line that revises in place;
  the authoritative translation replaces it once the segment finalizes. Set
  `TranslatorConfig.spec_model` to a smaller model (e.g.
  `kaelri/hy-mt2:1.8b-q8_0`) for a two-tier draft+final setup. The model is
  pre-warmed at session start, so the first segment translates fast.

- **Glossary** (GUI "词表" button / `glossary.txt`): one `source = target`
  entry per line; saving takes effect on the next segment, no restart. Spot a
  mistranslated term → add a line. General LLMs get it via prompt injection,
  HY-MT models via their official terminology format. Only entries matched in
  the current segment are injected, so translation stays fast. Tip: add common
  ASR mishearings (`clod = Claude`) to force corrections even with MT models.

- **Anti-feedback via digital AEC (default)**: instead of muting capture
  while TTS speaks (losing the programme audio underneath), the app
  subtracts its own known TTS waveform from the loopback capture —
  cross-correlation alignment + adaptive single-tap gain, ~-10dB residual,
  plus a pure-echo gate and a language filter as backstops. Only ~1s is lost
  per utterance (the alignment window) versus the whole utterance with
  gating. Verified: programme speech is transcribed *while* the interpreter
  talks over it. Disable with `--no-echo-cancel` (falls back to gating);
  AEC auto-disables when TTS plays on a separate device (`--tts-device`).

- **Audio-LLM one-step translation (experimented, archived)**: feeding audio
  slices straight into an omni model. Two rounds of evidence:
  llama.cpp b10437 (`bench_omni.py`) — audio path broken (Qwen2.5-Omni GGUF
  hears fragments, gemma-4-E4B crashes at init); transformers
  (`bench_omni_torch.py`, Qwen2.5-Omni-3B bf16) — hearing is fine, but 3B
  chat-contaminates transcripts, the refine mode parrots, and latency is a
  hardware-level dealbreaker: 1.5-13s per 15s slice on an RTX 5070 Ti, true
  for any ≤10B omni. Archived for live use; the cascade wins.

## Fresh install (after cloning)

Requirements: Windows 10/11, Python 3.10+ (3.13 verified). Zero compilation,
no PyTorch/WSL required.

```bat
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The script creates `.venv` and installs dependencies, downloads the default
ASR/TTS models (~1.5GB) and the portable Ollama runtime (~1.4GB), then pulls
the default translation model qwen3:4b-instruct (~2.5GB). Other candidate
models from the tables above can be downloaded on demand.

## Usage

GUI — the two commands are equivalent (both auto-attach the `.venv`
dependencies and auto-start Ollama):

```bat
run_gui.bat          :: double-click, no console window
python gui.py        :: bare system python works too
```

Dark-themed window: start/stop button, spoken-translation toggle, capture
device picker. The gray italic line is the live partial; finalized source
text and the blue translation (with latency tag) appear above it.

CLI:

```bat
run.bat                 :: full mode (subtitles + speech)
run.bat --no-tts        :: subtitles only (lowest latency, never interrupts)
run.bat --list-devices  :: list capture/playback devices
run.bat --capture-device 5 --tts-device 8   :: pick devices manually
run.bat --model qwen3:14b                   :: switch translation model
```

The first segment is slower (Ollama cold-loads the model into VRAM);
afterwards expect a translation ~1–2s after each sentence ends.

## Layout

```
gui.py             Tkinter GUI (no extra dependencies)
app.py             CLI entry point
interpreter/       modules: pipeline / capture / ASR / translation / TTS / display
models/            ASR + TTS + Ollama models (all project-local, fully portable)
libs/ollama/       portable Ollama (no registry writes, no system service)
selftest.py        offline self-test (ASR on bundled wavs + TTS synthesis)
bench_asr.py       streaming-ASR shootout (real clips in testclips/ + LibriSpeech)
bench_translate.py translation shootout (installed qwen/hunyuan models auto-enter)
bench_nmt.py       legacy NMT baseline (opus-mt / NLLB — spoiler: unusable)
make_refs.py       pseudo-references for testclips/ via offline Parakeet
testclips/         real YouTube test audio (fetched with yt-dlp, 16k mono)
```

## Tuning

`interpreter/config.py`:

- Endpointing sensitivity: `rule2_min_trailing_silence` (default 0.9s; lower
  = faster segments but choppier sentences)
- TTS voices: `en_speaker_id` / `zh_speaker_id` (kokoro v1.1 has 103 voices)
- TTS speed: `speed` (default 1.1 — slightly faster than source is standard
  practice for interpretation)
- Translation context depth: `history_size`
