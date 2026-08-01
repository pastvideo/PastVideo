[CmdletBinding()]
param(
    [string]$DataDir = "",
    [string]$Video = "",
    [int]$ApiPort = 8787,
    [int]$WebPort = 3001
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VideoWasProvided = -not [string]::IsNullOrWhiteSpace($Video)
if (-not $DataDir) {
    $DataDir = Join-Path $RepoRoot ".tools\web-data"
}
if (-not [System.IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $RepoRoot $DataDir
}

$env:PASTVIDEO_QWEN_PYTHON = if ($env:PASTVIDEO_QWEN_PYTHON) {
    $env:PASTVIDEO_QWEN_PYTHON
} else {
    Join-Path $env:USERPROFILE ".venvs\qwen3-vl-cu128\Scripts\python.exe"
}
$env:PASTVIDEO_QWEN_MODEL = if ($env:PASTVIDEO_QWEN_MODEL) {
    $env:PASTVIDEO_QWEN_MODEL
} else {
    Join-Path $env:USERPROFILE ".cache\pastvideo\models\Qwen3-VL-Embedding-2B-modelscope"
}

if (-not (Test-Path -LiteralPath $env:PASTVIDEO_QWEN_PYTHON) -or
    -not (Test-Path -LiteralPath $env:PASTVIDEO_QWEN_MODEL)) {
    Write-Host "Setting up the shared CUDA/Qwen runtime (first run only)..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "setup_qwen.ps1")
}

$BenchmarkDir = Join-Path $RepoRoot ".tools\benchmark-data\benchmark"
$BenchmarkVideo = Join-Path $BenchmarkDir "benchmark_video.mp4"
if (-not $Video) {
    $Video = $BenchmarkVideo
}
if (-not [System.IO.Path]::IsPathRooted($Video)) {
    $Video = Join-Path $RepoRoot $Video
}

$DatabasePath = Join-Path $DataDir "pastvideo.db"
$ShouldIndex = $VideoWasProvided -or -not (Test-Path -LiteralPath $DatabasePath)
if ($ShouldIndex) {
    if (-not (Test-Path -LiteralPath $Video)) {
        if ($VideoWasProvided -or $Video -ne $BenchmarkVideo) {
            throw "Video file or directory not found: $Video"
        }
        New-Item -ItemType Directory -Force -Path $BenchmarkDir | Out-Null
        Write-Host "Downloading the issue #68 benchmark clip..." -ForegroundColor Cyan
        Invoke-WebRequest `
            -Uri "https://github.com/ssrajadh/sentrysearch/releases/download/benchmark-clip-v1/benchmark_video.mp4" `
            -OutFile $BenchmarkVideo
    }
    Write-Host "Building PastVideo..." -ForegroundColor Cyan
    cargo build --release --manifest-path (Join-Path $RepoRoot "Cargo.toml")
    Write-Host "Indexing $Video with Qwen3-VL..." -ForegroundColor Cyan
    & (Join-Path $RepoRoot "target\release\pastvideo.exe") `
        --data-dir $DataDir --backend qwen index $Video --no-preprocess --no-skip-still
}

if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "target\release\pastvideo.exe"))) {
    cargo build --release --manifest-path (Join-Path $RepoRoot "Cargo.toml")
}
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "web\node_modules"))) {
    npm install --prefix (Join-Path $RepoRoot "web")
}

$LogDir = Join-Path $RepoRoot ".tools"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ApiProcess = $null
$WebProcess = $null
try {
    $ApiProcess = Start-Process `
        -FilePath (Join-Path $RepoRoot "target\release\pastvideo.exe") `
        -ArgumentList @(
            "--data-dir", $DataDir,
            "--backend", "qwen",
            "serve", "--bind", "127.0.0.1:$ApiPort",
            "--clips", (Join-Path $RepoRoot ".tools\web-clips")
        ) `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput (Join-Path $LogDir "web-api.log") `
        -RedirectStandardError (Join-Path $LogDir "web-api.err") `
        -WindowStyle Hidden `
        -PassThru

    $WebProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--port", "$WebPort") `
        -WorkingDirectory (Join-Path $RepoRoot "web") `
        -RedirectStandardOutput (Join-Path $LogDir "web-dev.log") `
        -RedirectStandardError (Join-Path $LogDir "web-dev.err") `
        -WindowStyle Hidden `
        -PassThru

    $Url = "http://localhost:$WebPort/"
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 1 | Out-Null
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Host "PastVideo is ready at $Url" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the local app."
    Start-Process $Url
    Wait-Process -Id $WebProcess.Id
} finally {
    if ($WebProcess -and -not $WebProcess.HasExited) {
        Stop-Process -Id $WebProcess.Id -Force
    }
    if ($ApiProcess -and -not $ApiProcess.HasExited) {
        Stop-Process -Id $ApiProcess.Id -Force
    }
}
