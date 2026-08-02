[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$ModelSource = "$env:USERPROFILE\.cache\pastvideo\models\Qwen3-VL-Embedding-2B-modelscope",
    [switch]$CreateArchive
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot ".tools\release\PastVideo-win-x64"
}
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}

$os = Get-CimInstance Win32_OperatingSystem
$memoryUsed = (($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100
if ($memoryUsed -ge 95) {
    throw "Memory use is above 95%; save work and restart Windows before packaging."
}

& cargo build --release --bin pastvideo-desktop --manifest-path (Join-Path $RepoRoot "Cargo.toml")
if (Test-Path -LiteralPath $OutputDir) {
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
    $releaseRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot ".tools\release"))
    if (-not $resolvedOutput.StartsWith($releaseRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Package output must stay under $releaseRoot before it can be replaced."
    }
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "target\release\pastvideo-desktop.exe") `
    -Destination (Join-Path $OutputDir "PastVideo.exe") -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $OutputDir -Force
if (Test-Path -LiteralPath (Join-Path $RepoRoot "LICENSE")) {
    Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination $OutputDir -Force
}

$RuntimeDir = Join-Path $OutputDir "runtime"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "python\qwen_worker.py") `
    -Destination (Join-Path $RuntimeDir "qwen_worker.py") -Force

if (Test-Path -LiteralPath (Join-Path $ModelSource "config.json")) {
    $ModelTemplate = Join-Path $OutputDir "model-template"
    New-Item -ItemType Directory -Force -Path $ModelTemplate | Out-Null
    Get-ChildItem -LiteralPath $ModelSource -Force -Recurse -File | Where-Object {
        $_.Name -ne "model.safetensors" -and
        $_.Name -notin @("README.md", ".gitattributes") -and
        $_.Extension -notin @(".pyc", ".pyo") -and
        $_.FullName -notmatch "[\\/]__pycache__[\\/]"
    } | ForEach-Object {
        $Relative = $_.FullName.Substring($ModelSource.TrimEnd("\").Length + 1)
        $Destination = Join-Path $ModelTemplate $Relative
        $DestinationParent = Split-Path -Parent $Destination
        New-Item -ItemType Directory -Force -Path $DestinationParent | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Force
    }
} else {
    throw "Qwen model metadata was not found at $ModelSource. The release needs its small model template files."
}

$FfmpegDir = Join-Path $RepoRoot ".tools\ffmpeg\bin"
if (Test-Path -LiteralPath $FfmpegDir) {
    $PackageBin = Join-Path $OutputDir "bin"
    New-Item -ItemType Directory -Force -Path $PackageBin | Out-Null
    foreach ($Tool in @("ffmpeg.exe", "ffprobe.exe", "ffplay.exe")) {
        $Source = Join-Path $FfmpegDir $Tool
        if (Test-Path -LiteralPath $Source) {
            Copy-Item -LiteralPath $Source -Destination $PackageBin -Force
        }
    }
}

Write-Host "PastVideo Windows package: $OutputDir" -ForegroundColor Green

if ($CreateArchive) {
    $Archive = Join-Path (Split-Path -Parent $OutputDir) "PastVideo-v0.2.0-win-x64.zip"
    if (Test-Path -LiteralPath $Archive) {
        Remove-Item -LiteralPath $Archive -Force
    }
    Compress-Archive -Path (Join-Path $OutputDir "*") -DestinationPath $Archive -CompressionLevel Optimal
    $Hash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    $Size = [math]::Round((Get-Item -LiteralPath $Archive).Length / 1MB, 1)
    Write-Host "Release archive: $Archive · $Size MiB · SHA-256 $Hash" -ForegroundColor Green
}
