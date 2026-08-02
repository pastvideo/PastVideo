[CmdletBinding()]
param(
    [string]$SourceEnvironment = "$env:USERPROFILE\.venvs\qwen3-vl-cu128",
    [string]$OutputDir = "",
    [string]$ArchiveDir = "",
    [switch]$ArchiveOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot ".tools\portable-ai-runtime"
}
if (-not $ArchiveDir) {
    $ArchiveDir = Join-Path $RepoRoot ".tools\release"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$ArchiveDir = [System.IO.Path]::GetFullPath($ArchiveDir)
$ToolsRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot ".tools"))
if (-not $OutputDir.StartsWith($ToolsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Portable runtime output must stay under $ToolsRoot"
}

$os = Get-CimInstance Win32_OperatingSystem
$memoryUsed = (($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100
if ($memoryUsed -ge 95) {
    throw "Memory use is above 95%; save work and restart Windows before building the runtime."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required on the release build machine. End users do not need uv."
}

if (-not $ArchiveOnly) {
$sourceConfig = Join-Path $SourceEnvironment "pyvenv.cfg"
if (-not (Test-Path -LiteralPath $sourceConfig)) {
    throw "Source Python environment was not found: $SourceEnvironment"
}
$homeLine = Get-Content -LiteralPath $sourceConfig | Where-Object { $_ -like "home = *" } | Select-Object -First 1
$pythonBase = $homeLine -replace "^home = ", ""
if (-not (Test-Path -LiteralPath (Join-Path $pythonBase "python.exe"))) {
    throw "The relocatable uv Python base was not found: $pythonBase"
}

if (Test-Path -LiteralPath $OutputDir) {
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}
$portablePythonDir = Join-Path $OutputDir "python"
New-Item -ItemType Directory -Force -Path $portablePythonDir | Out-Null
Get-ChildItem -LiteralPath $pythonBase -Force | Copy-Item -Destination $portablePythonDir -Recurse -Force
$portablePython = Join-Path $portablePythonDir "python.exe"

& uv pip install --python $portablePython `
    "torch==2.11.0" "torchvision==0.26.0" `
    --index-url https://download.pytorch.org/whl/cu128 `
    --system --break-system-packages
if ($LASTEXITCODE -ne 0) { throw "Could not install the portable CUDA PyTorch runtime." }

& uv pip install --python $portablePython `
    "transformers==5.14.1" `
    "accelerate==1.14.0" `
    "qwen-vl-utils==0.0.14" `
    "decord==0.6.0" `
    "PyNvVideoCodec==2.2.0" `
    "pillow==12.3.0" `
    --system --break-system-packages
if ($LASTEXITCODE -ne 0) { throw "Could not install the portable Qwen dependencies." }

Copy-Item -LiteralPath (Join-Path $RepoRoot "python\qwen_worker.py") `
    -Destination (Join-Path $OutputDir "qwen_worker.py") -Force

$sitePackages = Join-Path $portablePythonDir "Lib\site-packages"
$pruneRoots = @(
    (Join-Path $sitePackages "torch\include"),
    (Join-Path $sitePackages "torch\bin")
)
foreach ($path in $pruneRoots) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
Get-ChildItem -LiteralPath (Join-Path $sitePackages "torch\lib") -Filter "*.lib" -File `
    -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -LiteralPath $OutputDir -Directory -Recurse -Filter "__pycache__" `
    -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $OutputDir -File -Recurse -ErrorAction SilentlyContinue | `
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } | Remove-Item -Force

& $portablePython -c "import torch, torchvision, transformers, accelerate, qwen_vl_utils, decord, PyNvVideoCodec, PIL, numpy; assert torch.cuda.is_available(); print('Portable PastVideo AI runtime:', torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "Portable AI runtime smoke test failed." }
} else {
    $portablePythonDir = Join-Path $OutputDir "python"
    $portablePython = Join-Path $portablePythonDir "python.exe"
    if (-not (Test-Path -LiteralPath $portablePython)) {
        throw "ArchiveOnly requires an existing portable runtime at $OutputDir"
    }
}

# The import smoke test creates bytecode caches. Remove them again before every
# archive build so release bytes stay small and reproducible, including when
# -ArchiveOnly is used.
Get-ChildItem -LiteralPath $OutputDir -Directory -Recurse -Filter "__pycache__" `
    -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $OutputDir -File -Recurse -ErrorAction SilentlyContinue | `
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } | Remove-Item -Force

New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$coreArchive = Join-Path $ArchiveDir "PastVideo-AI-Runtime-Core-win-x64.zip"
$cudaArchive1 = Join-Path $ArchiveDir "PastVideo-AI-Runtime-CUDA-1-win-x64.zip"
$cudaArchive2 = Join-Path $ArchiveDir "PastVideo-AI-Runtime-CUDA-2-win-x64.zip"
foreach ($archive in @($coreArchive, $cudaArchive1, $cudaArchive2)) {
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
}

$cudaPattern = "^(c10_cuda|torch_cuda|cublas|cudnn|cufft|curand|cusolver|cusparse|nvrtc|nvJitLink|nvToolsExt|cupti)"
$allFiles = Get-ChildItem -LiteralPath $OutputDir -File -Recurse
$coreList = Join-Path $env:TEMP "pastvideo-runtime-core-$PID.txt"
$cudaList1 = Join-Path $env:TEMP "pastvideo-runtime-cuda-1-$PID.txt"
$cudaList2 = Join-Path $env:TEMP "pastvideo-runtime-cuda-2-$PID.txt"
try {
    $corePaths = [System.Collections.Generic.List[string]]::new()
    $cudaFiles = [System.Collections.Generic.List[object]]::new()
    $outputPrefixLength = $OutputDir.TrimEnd("\").Length + 1
    foreach ($file in $allFiles) {
        $relative = $file.FullName.Substring($outputPrefixLength).Replace("\", "/")
        if ($relative -like "python/Lib/site-packages/torch/lib/*" -and $file.BaseName -match $cudaPattern) {
            $cudaFiles.Add([pscustomobject]@{ Path = $relative; Length = $file.Length })
        } else {
            $corePaths.Add($relative)
        }
    }
    $cudaPaths1 = [System.Collections.Generic.List[string]]::new()
    $cudaPaths2 = [System.Collections.Generic.List[string]]::new()
    $cudaBytes1 = 0L
    $cudaBytes2 = 0L
    foreach ($file in ($cudaFiles | Sort-Object Length -Descending)) {
        if ($cudaBytes1 -le $cudaBytes2) {
            $cudaPaths1.Add($file.Path)
            $cudaBytes1 += $file.Length
        } else {
            $cudaPaths2.Add($file.Path)
            $cudaBytes2 += $file.Length
        }
    }
    [System.IO.File]::WriteAllLines($coreList, $corePaths, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllLines($cudaList1, $cudaPaths1, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllLines($cudaList2, $cudaPaths2, [System.Text.UTF8Encoding]::new($false))
    & tar.exe -a -c -f $coreArchive -C $OutputDir -T $coreList
    if ($LASTEXITCODE -ne 0) { throw "Could not create the core runtime archive." }
    & tar.exe -a -c -f $cudaArchive1 -C $OutputDir -T $cudaList1
    if ($LASTEXITCODE -ne 0) { throw "Could not create the first CUDA runtime archive." }
    & tar.exe -a -c -f $cudaArchive2 -C $OutputDir -T $cudaList2
    if ($LASTEXITCODE -ne 0) { throw "Could not create the second CUDA runtime archive." }
} finally {
    Remove-Item -LiteralPath $coreList, $cudaList1, $cudaList2 -Force -ErrorAction SilentlyContinue
}

foreach ($archive in @($coreArchive, $cudaArchive1, $cudaArchive2)) {
    $item = Get-Item -LiteralPath $archive
    if ($item.Length -ge 2GB) {
        throw "$($item.Name) is $([math]::Round($item.Length / 1GB, 2)) GiB; GitHub requires every release asset to be below 2 GiB."
    }
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "$($item.Name): $([math]::Round($item.Length / 1GB, 2)) GiB · SHA-256 $hash" -ForegroundColor Green
}
