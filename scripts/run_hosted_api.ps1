[CmdletBinding()]
param(
    [int]$ApiPort = 8787
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot ".tools"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

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

$Binary = Join-Path $RepoRoot "target\release\pastvideo.exe"
$DataDir = Join-Path $RepoRoot ".tools\web-data"
$ClipsDir = Join-Path $RepoRoot ".tools\web-clips"
if (-not (Test-Path -LiteralPath $Binary)) {
    throw "PastVideo release binary not found: $Binary"
}
if (-not (Test-Path -LiteralPath (Join-Path $DataDir "pastvideo.db"))) {
    throw "PastVideo web index not found: $DataDir"
}

while ($true) {
    $LogPath = Join-Path $LogDir "hosted-api.log"
    $ErrorPath = Join-Path $LogDir "hosted-api.err"
    $Process = Start-Process `
        -FilePath $Binary `
        -ArgumentList @(
            "--data-dir", $DataDir,
            "--backend", "qwen",
            "serve", "--bind", "127.0.0.1:$ApiPort",
            "--clips", $ClipsDir
        ) `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $LogPath `
        -RedirectStandardError $ErrorPath `
        -WindowStyle Hidden `
        -PassThru
    Wait-Process -Id $Process.Id
    Start-Sleep -Seconds 5
}
