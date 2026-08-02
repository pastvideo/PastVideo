[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Remote,
    [Parameter(Mandatory = $true)]
    [string]$KnownHosts,
    [int]$RemotePort = 38787,
    [int]$LocalPort = 8787
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot ".tools"
$KnownHostsPath = if ([System.IO.Path]::IsPathRooted($KnownHosts)) {
    $KnownHosts
} else {
    Join-Path $RepoRoot $KnownHosts
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path -LiteralPath $KnownHostsPath)) {
    throw "Pinned SSH host key file not found: $KnownHostsPath"
}

while ($true) {
    & "$env:WINDIR\System32\OpenSSH\ssh.exe" `
        -N -T `
        -o BatchMode=yes `
        -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=30 `
        -o ServerAliveCountMax=3 `
        -o StrictHostKeyChecking=yes `
        -o "UserKnownHostsFile=$KnownHostsPath" `
        -R "127.0.0.1:$RemotePort`:127.0.0.1:$LocalPort" `
        $Remote `
        1>> (Join-Path $LogDir "hosted-tunnel.log") `
        2>> (Join-Path $LogDir "hosted-tunnel.err")
    Start-Sleep -Seconds 5
}
