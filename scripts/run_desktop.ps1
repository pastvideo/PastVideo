[CmdletBinding()]
param(
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @("run")
if ($Release) {
    $Arguments += "--release"
}
$Arguments += @("--bin", "pastvideo-desktop")

Push-Location $RepoRoot
try {
    & cargo @Arguments
} finally {
    Pop-Location
}
