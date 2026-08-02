[CmdletBinding()]
param(
    [string]$EnvironmentDir = "$env:USERPROFILE\.venvs\qwen3-vl-cu128",
    [string]$ModelDir = "$env:USERPROFILE\.cache\pastvideo\models\Qwen3-VL-Embedding-2B-modelscope",
    [switch]$SkipUnderstandingModelDownload
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/ first."
}

$python = Join-Path $EnvironmentDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    New-Item -ItemType Directory -Force (Split-Path $EnvironmentDir) | Out-Null
    uv venv $EnvironmentDir --python 3.11
}

$cudaReady = $false
try {
    & $python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    $cudaReady = $LASTEXITCODE -eq 0
} catch {
    $cudaReady = $false
}

if (-not $cudaReady) {
    uv pip install --python $python "torch==2.8.0" "torchvision==0.23.0" `
        --index-url https://download.pytorch.org/whl/cu128
}

uv pip install --python $python `
    "sentence-transformers[video]==5.5.1" `
    "qwen-vl-utils==0.0.14" `
    "decord==0.6.0" `
    "PyNvVideoCodec==2.2.0" `
    "modelscope>=1.39.0"

$understandingRequirements = Join-Path $PSScriptRoot "..\python\requirements-understanding.txt"
uv pip install --python $python -r $understandingRequirements

$modelFile = Join-Path $ModelDir "model.safetensors"
if (-not (Test-Path -LiteralPath $modelFile)) {
    New-Item -ItemType Directory -Force $ModelDir | Out-Null
    $modelscope = Join-Path $EnvironmentDir "Scripts\modelscope.exe"
    & $modelscope download --model qwen/Qwen3-VL-Embedding-2B --local_dir $ModelDir
}

if (-not $SkipUnderstandingModelDownload) {
    Write-Host "Caching the default local Caption and Whisper models..."
    & $python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct'); from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('Caption and Whisper models cached.')"
}

& $python -c "import torch, accelerate, faster_whisper, rapidocr; print(f'PastVideo local AI runtime ready: PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
Write-Host "Model: $ModelDir"
if ($SkipUnderstandingModelDownload) {
    Write-Host "Caption and Whisper model weights will download on first use."
}
