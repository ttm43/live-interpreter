# 一键安装：Python 依赖 + ASR/TTS 模型 + llama.cpp(CUDA) + Confucius4-R2T2 + Ollama 便携版 + 翻译模型
# 用法：在项目目录执行  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "[1/6] Python venv + 依赖"
if (-not (Test-Path "$root\.venv")) { python -m venv "$root\.venv" }
& "$root\.venv\Scripts\python.exe" -m pip install --quiet -r "$root\requirements.txt"

Write-Host "[2/6] sherpa-onnx ASR / TTS 模型 (官方发布; parakeet = 无显卡时的定稿引擎)"
$asrBase = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
$ttsBase = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
$models = @(
    @{ url = "$asrBase/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2";        dir = "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms" },
    @{ url = "$asrBase/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2";                          dir = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" },
    @{ url = "$asrBase/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2"; dir = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11" },
    @{ url = "$ttsBase/kokoro-multi-lang-v1_1.tar.bz2";                                              dir = "kokoro-multi-lang-v1_1" }
)
New-Item -ItemType Directory -Force "$root\models" | Out-Null
foreach ($m in $models) {
    if (Test-Path "$root\models\$($m.dir)\tokens.txt") { continue }
    $tar = "$root\models\_dl.tar.bz2"
    Write-Host "  下载 $($m.dir) ..."
    curl.exe -sL -o $tar $m.url
    tar -xjf $tar -C "$root\models"
    Remove-Item $tar
}

Write-Host "[3/6] llama.cpp CUDA 预编译版（跑 Confucius4-R2T2；优先复用 ..\shared 的共享安装）"
$llamaTag = "b11193"
$llamaDir = "$root\..\shared\llama.cpp-$llamaTag"
if (-not (Test-Path "$root\..\shared")) { $llamaDir = "$root\libs\llama.cpp" }
if (-not (Test-Path "$llamaDir\llama-server.exe")) {
    New-Item -ItemType Directory -Force $llamaDir | Out-Null
    $llamaBase = "https://github.com/ggml-org/llama.cpp/releases/download/$llamaTag"
    foreach ($name in @("llama-$llamaTag-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip")) {
        $zip = "$llamaDir\_dl.zip"
        Write-Host "  下载 $name ..."
        curl.exe -sL -o $zip "$llamaBase/$name"
        Expand-Archive -Path $zip -DestinationPath $llamaDir -Force
        Remove-Item $zip
    }
}

Write-Host "[4/6] Confucius4-R2T2 识别模型 (Q8_0 GGUF + 音频投影器, 约 2.4GB, 有道官方 HF 仓库)"
$hfBase = "https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF/resolve/main"
$registry = "$root\..\models\registry.yaml"
$ggufs = @(
    @{ file = "Confucius4-R2T2-Q8_0.gguf";       key = "confucius4-r2t2-q8_0-gguf" },
    @{ file = "mmproj-Confucius4-R2T2-f16.gguf"; key = "confucius4-r2t2-mmproj-gguf" }
)
foreach ($g in $ggufs) {
    # 机器级模型库已登记该键 -> 放进库里；否则放项目 models\ 下、按 registry 键命名
    # （interpreter/bootstrap.py model_path 的回退路径）
    if ((Test-Path $registry) -and (Select-String -Path $registry -Pattern "^\s*$($g.key):" -Quiet)) {
        $dest = "$root\..\models\asr\confucius4-r2t2-gguf\$($g.file)"
    } else {
        $dest = "$root\models\$($g.key).gguf"
    }
    if (Test-Path $dest) { continue }
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Write-Host "  下载 $($g.file) ..."
    curl.exe -L --progress-bar -o $dest "$hfBase/$($g.file)"
}

Write-Host "[5/6] Ollama 便携版（优先复用 ..\shared 的共享安装）"
$ollamaExe = "$root\..\shared\ollama\ollama.exe"
$modelsDir = "$root\..\shared\ollama-models"
if (-not (Test-Path $ollamaExe)) {
    $ollamaExe = "$root\libs\ollama\ollama.exe"
    $modelsDir = "$root\models\ollama"
    if (-not (Test-Path $ollamaExe)) {
        New-Item -ItemType Directory -Force "$root\libs\ollama" | Out-Null
        $zip = "$root\libs\ollama.zip"
        curl.exe -sL -o $zip "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
        Expand-Archive -Path $zip -DestinationPath "$root\libs\ollama" -Force
        Remove-Item $zip
    }
}

Write-Host "[6/6] 翻译模型 (qwen3:4b-instruct 默认; 低内存机器可换 kaelri/hy-mt2:1.8b-q8_0)"
$env:OLLAMA_MODELS = $modelsDir
New-Item -ItemType Directory -Force $env:OLLAMA_MODELS | Out-Null
$serve = Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 4
& $ollamaExe pull qwen3:4b-instruct

Write-Host "完成。运行 run_gui.bat 或 python gui.py 启动。"
Write-Host "识别默认 confucius（需要 NVIDIA 显卡，约 2.5GB 显存）；没有显卡的机器在「识别」下拉框选 parakeet-semi（CPU）。"
Write-Host "可选模型（更高质量/更多对比项）：ollama pull qwen3:14b ；其余 ASR 档位参见 README。"
