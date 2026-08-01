[CmdletBinding()]
param(
    [string]$OutputDir = ""
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
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "target\release\pastvideo-desktop.exe") `
    -Destination (Join-Path $OutputDir "PastVideo.exe") -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $OutputDir -Force

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
