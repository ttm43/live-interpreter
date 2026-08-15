# 一键安装：Python 依赖 + ASR/TTS 模型 + Ollama 便携版 + 翻译模型
# 用法：在项目目录执行  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "[1/4] Python venv + 依赖"
if (-not (Test-Path "$root\.venv")) { python -m venv "$root\.venv" }
& "$root\.venv\Scripts\python.exe" -m pip install --quiet -r "$root\requirements.txt"

Write-Host "[2/4] ASR / TTS 模型 (sherpa-onnx 官方发布)"
$asrBase = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
$ttsBase = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
$models = @(
    @{ url = "$asrBase/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2"; dir = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11" },
    @{ url = "$asrBase/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2";        dir = "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms" },
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

Write-Host "[3/4] Ollama 便携版（优先复用 ..\shared 的共享安装）"
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

Write-Host "[4/4] 翻译模型 (qwen3:4b-instruct 默认; 低内存机器可换 kaelri/hy-mt2:1.8b-q8_0)"
$env:OLLAMA_MODELS = $modelsDir
New-Item -ItemType Directory -Force $env:OLLAMA_MODELS | Out-Null
$serve = Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 4
& $ollamaExe pull qwen3:4b-instruct

Write-Host "完成。运行 run_gui.bat 或 python gui.py 启动。"
Write-Host "可选模型（更高质量/更多对比项）：ollama pull qwen3:14b ；其余 ASR 档位参见 README。"
