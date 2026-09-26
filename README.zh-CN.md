# Live Interpreter（系统音频实时同传）

[English](README.md) | **中文**

采集电脑正在播放的声音（WASAPI loopback），实时做同声传译：英文语音进来，
实时字幕 + 中文译文出去，还可以让 TTS 把译文朗读出来。

全部本地运行，无需联网 API、无需 WSL、无需 PyTorch，Windows 上零编译。

## 架构

```
系统音频 (扬声器 loopback, PyAudioWPatch, 自动增益)
   → ASR: Confucius4-R2T2 定稿 (llama.cpp, GPU) + sherpa-onnx 流式预览 (双引擎, 见下)
   → 翻译 (Ollama 本地推理, 词表干预, 能纠 ASR 错词)
   → 双语 TTS (sherpa-onnx kokoro-multi-lang-v1_1) → 扬声器播放
```

- **当前专注英→中**（自动/中→英 已暂时下线，代码保留在 config/gui 注释里）
- **识别模型可切换**（GUI「识别」下拉框），真实 YouTube 音频基准
  （`bench_asr.py`，伪参考 = Parakeet-TDT-0.6B-v3 离线转写）：

  | 模型 | 真实片段平均 WER | 字幕更新间隔 | 定位 |
  |---|---|---|---|
  | confucius | **所有引擎中真人标注语料精度最高**：320 条 ESB 会议/电话/播客/朗读语句 WER 9.4%，parakeet-semi 生产路径 12.1%（见下文对决）；自带标点、大小写、数字规整 | ~0.5s 只增不改的预览，定稿 ~0.2s | 2026-09-26 起两条通道的**默认**定稿引擎。有道 Confucius4-R2T2（2B，Q8_0 GGUF）经自动拉起的 llama-server 推理；需要 NVIDIA 显卡（约 2.5GB 显存） |
  | parakeet-semi | CPU 引擎中精度最高（LibriSpeech 0%/2.1%；重口音比任何流式引擎好约 4 倍） | ~850ms 整句修订式 | 无显卡时的定稿引擎（CPU）；RTF≈0.4 |
  | whisper-semi | 与 parakeet-semi 同档；人名（Mikhail Fedorov）和数字（35）最准、自动抹平结巴 | ~5-7s 修订 | faster-whisper large-v3-turbo int8。**CPU RTF 0.8~1.3 撑不住实时**；音乐/静音段会幻觉出 "Thank you."。保留作对比 |
  | nemotron3.5-1120ms | **≈23%（流式引擎中新闻/发布会/播客三项最佳）** | ~1.3s | 流式备选；自带标点+大小写，CPU 占用低 5 倍 |
  | nemo-1040ms | ≈28%；**重口音场景仍最强** | ~1.2s | 口音重的说话人用这个 |
  | nemotron3.5-320ms | ≈31% | ~500ms | 低延迟折中 |
  | nemo-80ms | ≈37% | **~330ms** | 逐字感优先 |
  | nemo-480ms | ≈39% | ~700ms | 折中（配乐场景异常差） |
  | zipformer-2023 | ≈41%（真实音频最差） | ~360ms | 仅 LibriSpeech 朗读音频强 |

  注：nemotron3.5 是多语模型，重口音会触发语种混淆（吐外文碎片）；其
  80ms/160ms 档在 CPU 上 RTF 0.4~0.9 且精度全场最差（小分块双输），已剔除；
  560ms 档 ≈28% 带标点，是单引擎低延迟场景的可选折中。

- **双引擎（默认开启）**：「预览」引擎（默认 nemo-80ms，~240ms 逐字更新）
  只负责灰色预览行的手感；「识别」引擎（默认 confucius）负责定稿和喂翻译。
  单模型做不到又快又准，双引擎各取所长。GUI「预览」下拉框可换预览引擎或
  选「关闭」回到单引擎。

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

- **会议助手（默认开启）**：面向英语非母语的参会者，不止翻译字面意思。
  每段定稿识别文本会连同最近的会议上下文一起送给 LLM，在右侧面板输出：
  - **【意图】**：对方这段话到底想表达/询问/推动什么，包括言外之意和情绪立场；
  - **【提示】**：当这段话需要你回应时（提问、点名、征求意见等），给出
    1-3 条大方向互不相同的回复思路（同意推进 / 提出顾虑 / 先澄清再表态），
    每条 = 方向（中文 + 简单英文说法）+ 2-4 个关键词（每个英文关键词
    紧跟中文释义，如 `risk 风险`）——刻意不给完整句子，词汇限制在最常见
    的日常词（读音、含义都不会出错的程度），你照关键词组织自己的话；
  - **【无需回应】**：对方在陈述信息或明确点名别人时，明确告诉你不用接话。

  **题库直答**（GUI「题库」按钮 / 根目录 `qa_bank.md`，已 gitignore）：
  提前把预想的问题和准备好的答案写进题库（`### 编号 + **问法：** + 答案`，
  `**↳ ...**` 为追问），听到问题时先做一次轻量匹配（输出只有一个编号，
  ~0.15 秒）：命中就把你写好的答案原样放上面板；匹配不到绝不硬凑，
  回退到上面的意图+关键词提示。像 "Another example?" 这类几个词的追问
  靠上下文和上一次匹配定位到对应追问条目。实测（90 条目题库）：
  主问题 26/27 精确命中、0 错组，追问链 9/9。热加载，会中可随时补题。

  **抢跑提示（借鉴投机翻译）**：对方问题还没说完时，助手就开始分析不断
  增长的识别文本，蓝灰色斜体的预测提示原地刷新——通常对方话音刚落，
  方向提示已经在屏幕上；定稿分析在 2-4 秒后替换预测版。题库匹配也走
  抢跑：问题说到一半命中后，准备好的答案直接提前上屏。

  工具栏「称呼」填上同事们对你的称呼（如 `Ximing`），助手就能区分
  "在问你"和"在问别人"。

  **背景资料**（GUI「背景」按钮 / 根目录 `background.txt`）：开会前或会中
  随时把议程、与会者角色、项目现状、你此次开会的目标写进去，保存后下一段
  分析立即生效（与词表相同的热加载机制，`#` 开头为注释）。助手会结合背景
  判断意图并让回复方向贴着你的目标走——实测背景里写"避免当场承诺日期"，
  三条回复建议全部改为先评估风险/先要信息，而不是直接答应。

  LLM 追不上语速时自动把积压的几段合并成一次分析，
  始终贴着会议进度。默认复用翻译模型（不占额外显存）；
  `AssistantConfig.model` 可单独指定更强的模型（如 `qwen3:14b`）。
  取消勾选「会议助手」即回到纯同传模式。命令行版用 `--no-assist` 关闭。

- **我方麦克风通道（默认开启）**：界面分成三栏——左「对方」（系统音频）、
  中「我」（麦克风）、右「会议助手」。你自己说的话走独立的 ASR 实例，修正后的
  英文 + 译文显示在中栏，方便回头确认自己有没有说清楚。同时以 `[我]` 标注写进
  助手的会议上下文，这样对方接着问的时候，助手知道那是对你刚才回答的追问，
  而不是一个新问题。第二行工具栏可选麦克风
  设备和识别模型（默认 confucius：和对方通道共用同一个 llama-server，会上一次
  只有一个人说话，所以第二个实例几乎不额外花钱；没显卡的机器选 parakeet-semi）。
  TTS 朗读时麦克风自动静音，
  避免把自己的合成语音转写进来。**建议戴耳机**：开放式麦克风会把扬声器里
  对方的声音也录进「我」这一栏。命令行版 `--no-mic` / `--mic-device N`。

- **对话级解释器（`DialogueInterpreter`）**：两条音频通路不再各自跑一个
  单句翻译器，而是共用一个带说话人标签的对话上下文：

  ```
  [OTHER] How would you handle an out-of-date SWMS?
  [ME]    for the swims I would first check who signed it off   ← 原始 ASR
  ```

  每段定稿一次推理同时输出**修正后的原文**和**译文**（Ollama `format: json`
  在解码层强制严格 JSON，不靠正则拆）。ASR 最容易听错的恰恰是对方刚引入的
  专有名词、缩写和术语——上面这句 `swims → SWMS` 只有看到上一轮才修得出来，
  实测 0.8 秒；`it / that / they` 的指代、省略主语也都受益。"是否修改过"由
  代码比对（忽略大小写和标点）决定而不是让模型自报，改过时原始 ASR 以灰色
  "原: …" 留作审计行；填充词（um/uh）也在代码里去掉。会议背景资料同样喂给
  解释器，所以建筑面试里的 "live site" 会译成工地而不是生产环境。窗口按
  轮次 + 字符数双重上限（默认 8 轮 / 1500 字符），锁只包住推理和历史追加，
  抢跑翻译读快照后在锁外推理。翻译特化模型（HY-MT）不支持纠错，自动回退纯翻译。
  会议助手保持完全独立的上下文和提示词。

- **词表**（GUI「词表」按钮 / 根目录 `glossary.txt`）：`原文 = 译文` 一行一条，
  保存后下一段立即生效。发现翻错术语 → 加一行即可。通用 LLM 走 prompt 注入，
  HY-MT 系走官方术语干预格式；只注入当前句子命中的词条，不拖慢翻译。
  技巧：把常见听错形式也加进词表（`clod = Claude`），翻译特化模型也能被拽回来。

- **防回声改为数字 AEC（默认）**：不再在 TTS 朗读时掐断采集（那会把底下的
  节目声一起丢掉），而是把自己已知的 TTS 波形从 loopback 采集里减掉——
  互相关对齐 + 自适应单抽头增益，残余约 -10dB，另有纯回声门限和语言过滤
  两道兜底。每次朗读只损失约 1 秒对齐窗口（旧门控损失整段）。已实测：
  译音播放的同时节目原声仍被识别翻译。`--no-echo-cancel` 可退回旧门控；
  TTS 输出到独立设备（`--tts-device`）时 AEC 自动停用。

- **音频 LLM 一步直翻（已实验，归档）**：音频切片直接喂 omni 模型。两轮证据：
  llama.cpp b10437（`bench_omni.py`）——音频路径损坏（Qwen2.5-Omni GGUF 只听到
  碎片、gemma-4-E4B 初始化崩溃）；transformers（`bench_omni_torch.py`，
  Qwen2.5-Omni-3B bf16）——听力正常，但 3B 转写会"聊起来"、refine 模式复读草稿，
  且延迟是硬件级死刑：RTX 5070 Ti 上 15 秒切片要 1.5~13 秒，任何 ≤10B omni
  都一样。实时用途正式归档，级联胜出。

## ASR 对决：有道 Confucius4-R2T2 vs 生产管线（2026-09-26）

[Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) 是网易有道的
真流式 ASR（Qwen3-ASR-1.7B 微调，输出"最长稳定前缀"只追加不回改）。它的流式模式只有
vLLM 后端（Linux），但官方 GGUF 可以在 Windows 上用 llama.cpp 离线跑——和 Omni 系不同，
llama.cpp 对 Qwen3-ASR 家族的音频路径是完好的。

**测试集**（`esb_sample.py`）：Open ASR 排行榜测试集的 320 条人工标注语音、35 分钟——
AMI 会议 / Earnings-22 财报会 / GigaSpeech / LibriSpeech-clean 各 80 条，跨时长分布抽样、
按录音限额。所有赛道吃同样的、经过管线 AGC 的音频。**逐字 WER** 对双方用 Whisper 英文
规整器；**容忍口语 WER** 再对双方折叠重复词和语气词（逐字参考里保留着 "or or"、"you know"）。

| 语料 WER % | Confucius 单独 | Parakeet 离线 | Parakeet + 4B 纠错 | **生产路径**（半流式 + AGC + 纠错） |
|---|---|---|---|---|
| AMI 会议 | **12.1** | 13.3 | 15.3 | 20.0 |
| Earnings-22 | **14.2** | 15.3 | 14.8 | 17.0 |
| GigaSpeech | **10.7** | 10.9 | 11.0 | 11.3 |
| LibriSpeech-clean | **1.3** | 3.0 | 3.2 | 3.8 |
| **合计（逐字）** | **9.4** | 10.4 | 10.6 | 12.1 |
| **合计（容忍口语）** | **8.1** | 9.7 | 9.9 | 11.1 |

Confucius 对生产路径：合计的 95% 聚类 bootstrap 置信区间 [−4.3, −1.2]（显著），AMI 和
LibriSpeech 上同样显著。Q8_0 GGUF 与 f16 相差 0.2 点以内。算力：Confucius GPU 整段解码
每秒音频 0.03 秒，生产半流式 CPU 0.31 秒。

结论：
1. **Confucius 单模型在每个子集上都赢过现有的"ASR + LLM 纠错"**，会议语音领先最多，且输出
   自带标点、大小写和数字规整。
2. **4B 纠错对真实错误的修复 ≈ 0**，在会议语音上还倒扣 1~2 点（改写、去重复词被逐字参考
   惩罚）。它的设计价值——跨通道术语上下文——在孤立单句上测不出来。
3. **生产路径缺陷**：半流式 Parakeet 比整段 Parakeet 差约 2 点（AMI 差 6.6 点），并且对 8 条
   清晰的短句*完全没有出定稿*。根因：Parakeet int8 对特定输入波形会返回空串（可复现；
   放大 4 倍增益或换个窗口就正常），半流式的窗口（预滚动 + 补零）更容易撞上；管线里的
   AGC 是让其余部分正常工作的关键。

复现：`esb_sample.py`（生成 `testclips/esb/`），再 `bench_confucius.py [--quant f16|Q8_0]
[--skip-llm]`；逐句结果在 `testclips/esb/results_<quant>.jsonl`。模型是 registry 键
`confucius4-r2t2-{f16,q8_0,mmproj}-gguf`；llama-server 默认在
`<workspace>\shared\llama.cpp-b11193`。

**结果——2026-09-26 起已上生产。** `confucius` 是两条通道的默认定稿引擎
（`interpreter/asr_confucius.py`）。R2T2 原生流式（"最长稳定前缀"：转写只增不改）
只有 vLLM 后端才有，所以在 llama-server 上模拟：每 ~0.5 秒把这句到目前为止的音频
连同已提交的转写（作为 assistant 预填）重新发一次，只要 12 个新 token；回复里除最后
一个词外全部提交（边缘词经常被截在半截、还会被补上一个假句号）。定稿是整句重新
解码（GPU 上约 0.2 秒），也就是上表测的那条路径；语种通过预填强制
（`language English<asr_text>`）。llama-server 按需拉起（端口 8091，`-np 2`：每条
通道一个槽，约 2.5GB 显存），程序退出时一起结束。断句、18 秒窗口上限和空定稿
的抢救级联与 Parakeet 半流式引擎共用。

## 全新安装（克隆后）

要求：Windows 10/11，Python 3.10+（3.13 已验证）。项目零编译，不依赖 PyTorch/WSL。

```bat
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本会创建 `.venv` 装依赖、下载 sherpa-onnx ASR/TTS 模型（约 2GB，含无显卡
时用的 parakeet-semi）、llama.cpp CUDA 预编译版（约 0.5GB；有 `..\shared`
工作区时放 `..\shared\llama.cpp-b11193`，否则放 `libs\llama.cpp`）、
Confucius4-R2T2 Q8_0 GGUF + 音频投影器（约 2.4GB）、Ollama 便携版（约 1.4GB），
并拉取默认翻译模型 qwen3:4b-instruct（约 2.5GB）。表格中的其他候选模型按需
另行下载。没有 NVIDIA 显卡的机器：「识别」下拉框（和麦克风通道）选 parakeet-semi。

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
libs/llama.cpp/    跑 Confucius4-R2T2 的 llama-server（或 ..\shared\llama.cpp-b<N>）
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
