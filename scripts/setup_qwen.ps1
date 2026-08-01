[CmdletBinding()]
param(
    [string]$EnvironmentDir = "$env:USERPROFILE\.venvs\qwen3-vl-cu128",
    [string]$ModelDir = "$env:USERPROFILE\.cache\pastvideo\models\Qwen3-VL-Embedding-2B-modelscope"
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

$modelFile = Join-Path $ModelDir "model.safetensors"
if (-not (Test-Path -LiteralPath $modelFile)) {
    New-Item -ItemType Directory -Force $ModelDir | Out-Null
    $modelscope = Join-Path $EnvironmentDir "Scripts\modelscope.exe"
    & $modelscope download --model qwen/Qwen3-VL-Embedding-2B --local_dir $ModelDir
}

& $python -c "import torch; print(f'Qwen runtime ready: {torch.__version__}, CUDA={torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
Write-Host "Model: $ModelDir"
