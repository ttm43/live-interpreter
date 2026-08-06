# Live Interpreter（系统音频实时同传）

[English](README.md) | **中文**

采集电脑正在播放的声音（WASAPI loopback），实时做同声传译：英文语音进来，
实时字幕 + 中文译文出去，还可以让 TTS 把译文朗读出来。

全部本地运行，无需联网 API、无需 WSL、无需 PyTorch，Windows 上零编译。

## 架构

```
系统音频 (扬声器 loopback, PyAudioWPatch, 自动增益)
   → 流式 ASR (sherpa-onnx, 带断句; 双引擎, 见下)
   → 翻译 (Ollama 本地推理, 词表干预, 能纠 ASR 错词)
   → 双语 TTS (sherpa-onnx kokoro-multi-lang-v1_1) → 扬声器播放
```

- **当前专注英→中**（自动/中→英 已暂时下线，代码保留在 config/gui 注释里）
- **识别模型可切换**（GUI「识别」下拉框），真实 YouTube 音频基准
  （`bench_asr.py`，伪参考 = Parakeet-TDT-0.6B-v3 离线转写）：

  | 模型 | 真实片段平均 WER | 字幕更新间隔 | 定位 |
  |---|---|---|---|
  | nemotron3.5-1120ms | **≈23%（新闻/发布会/播客三项最佳）** | ~1.3s | **默认**；自带标点+大小写 |
  | nemo-1040ms | ≈28%；**重口音场景仍最强** | ~1.2s | 口音重的说话人用这个 |
  | nemotron3.5-320ms | ≈31% | ~500ms | 低延迟折中 |
  | nemo-80ms | ≈37% | **~330ms** | 逐字感优先 |
  | nemo-480ms | ≈39% | ~700ms | 折中（配乐场景异常差） |
  | zipformer-2023 | ≈41%（真实音频最差） | ~360ms | 仅 LibriSpeech 朗读音频强 |

  注：nemotron3.5 是多语模型，重口音会触发语种混淆（吐外文碎片）；其
  80ms/160ms 档在 CPU 上 RTF 0.4~0.9 且精度全场最差（小分块双输），已剔除；
  560ms 档 ≈28% 带标点，是单引擎低延迟场景的可选折中。

- **双引擎（默认开启）**：「预览」引擎（默认 nemo-80ms，~240ms 逐字更新）
  只负责灰色预览行的手感；「识别」引擎（默认 nemotron3.5-1120ms）负责定稿
  和喂翻译。单模型做不到又快又准，双引擎各取所长，合计 RTF ≈0.25。
  GUI「预览」下拉框可换预览引擎或选「关闭」回到单引擎。

- **翻译模型可切换**（GUI「翻译」下拉框，列出本地 Ollama 全部模型；
  `bench_translate.py` 对比，含 ASR 脏输入测试）。实测结论：

  | 模型 | 大小 | 定位 |
  |---|---|---|
  | qwen3:4b-instruct | 2.5G | **默认**：质量/速度/体积最佳平衡，能纠 ASR 错词（clod→Claude） |
  | qwen3:14b | 9.3G | 质量最高（数字、人名、术语全对），显存充裕时用 |
  | kaelri/hy-mt2:1.8b-q8_0 | 2.0G | 翻译特化（腾讯 Hy-MT2），低内存机器首选 |
  | qwen3:8b | 5.2G | 数字不稳定（four billion 翻错过两次），不再推荐 |
  | demonbyron/HY-MT1.5-7B | 4.6G | 上一代翻译特化，被 4b-instruct/Hy-MT2 替代 |

  Hunyuan/HY-MT 系模型自动使用其官方翻译提示词与术语干预格式
  （translator.py 按模型名识别）。注意：HY-MT1.5-**1.8B** 的 GGUF 在
  Ollama 下输出损坏（回显模板/幻觉），已验证不可用——要小模型请用 Hy-MT2。

- **投机翻译**（借鉴 GPT-Live 的"投机视图 + 权威视图"）：一句话还没说完时，
  不断增长的识别文本就被临时翻译成蓝灰色的修订行；断句定稿后由权威译文替换。
  在 `TranslatorConfig.spec_model` 里指定更小的模型（如
  `kaelri/hy-mt2:1.8b-q8_0`）即可组成"小模型草稿 + 大模型定稿"的双档。
  翻译模型在会话开始时预热，首段不再有冷启动延迟。

- **词表**（GUI「词表」按钮 / 根目录 `glossary.txt`）：`原文 = 译文` 一行一条，
  保存后下一段立即生效。发现翻错术语 → 加一行即可。通用 LLM 走 prompt 注入，
  HY-MT 系走官方术语干预格式；只注入当前句子命中的词条，不拖慢翻译。
  技巧：把常见听错形式也加进词表（`clod = Claude`），翻译特化模型也能被拽回来。

- 防回声：TTS 朗读期间自动暂停采集（否则会把自己的译音再翻一遍）。
  如果 TTS 输出到另一台设备（如耳机），可用 `--no-mute-during-tts` 关掉门控。

## 全新安装（克隆后）

要求：Windows 10/11，Python 3.10+（3.13 已验证）。项目零编译，不依赖 PyTorch/WSL。

```bat
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本会创建 `.venv` 装依赖、下载默认 ASR/TTS 模型（约 1.5GB）、下载 Ollama
便携版（约 1.4GB）并拉取默认翻译模型 qwen3:4b-instruct（约 2.5GB）。
表格中的其他候选模型按需另行下载。

## 使用

图形界面，以下两种方式等价（都会自动挂载 .venv 依赖并拉起 Ollama）：

```bat
run_gui.bat          :: 双击即可，无黑窗口
python gui.py        :: 在项目目录下直接跑也行
```

深色窗口界面：「开始/停止」按钮、朗读译文开关、采集设备下拉框；
灰色斜体行实时显示识别中的半句，识别定稿后显示原文 + 蓝色译文（带延迟标注）。

命令行版：

```bat
run.bat                 :: 完整模式（字幕 + 语音）
run.bat --no-tts        :: 纯字幕模式（延迟最低，也不会打断原声）
run.bat --list-devices  :: 列出可选的采集/播放设备
run.bat --capture-device 5 --tts-device 8   :: 手动指定设备
run.bat --model qwen3:14b                   :: 换翻译模型
```

第一段翻译稍慢（Ollama 冷启动加载模型到显存），之后每句结束约 1~2 秒出译文。

## 目录

```
gui.py             图形界面（tkinter，无额外依赖）
app.py             命令行入口
interpreter/       各模块：管线编排 / 采集 / ASR / 翻译 / TTS / 显示
models/            ASR + TTS + Ollama 模型（全部项目内，可整体搬移）
libs/ollama/       Ollama 便携版（不写注册表，不装系统服务）
selftest.py        离线自检（ASR 识别测试音频 + TTS 合成试听 wav）
bench_asr.py       流式 ASR 模型对比（testclips/ 真实音频 + LibriSpeech）
bench_translate.py 翻译模型对比（安装的 qwen/hunyuan 系自动参战）
bench_nmt.py       老一代 NMT 基线（opus-mt / NLLB——剧透：不可用）
make_refs.py       用 Parakeet 离线模型为 testclips/ 生成伪参考文本
testclips/         真实 YouTube 测试音频（yt-dlp 抓取，16k 单声道）
```

## 调参位置

`interpreter/config.py`：

- 断句灵敏度：`rule2_min_trailing_silence`（默认 0.9 秒，调小出字快但句子碎）
- TTS 音色：`en_speaker_id` / `zh_speaker_id`（kokoro v1.1 共 103 个音色）
- TTS 语速：`speed`（默认 1.1，同传建议略快于原速）
- 翻译上下文条数：`history_size`
