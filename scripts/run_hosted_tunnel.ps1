[CmdletBinding()]
param(
    [string]$Remote = "flowbehappy@claw9d.com",
    [int]$RemotePort = 38787,
    [int]$LocalPort = 8787
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot ".tools"
$KnownHosts = Join-Path $RepoRoot "scripts\claw9d_known_hosts"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path -LiteralPath $KnownHosts)) {
    throw "Pinned SSH host key file not found: $KnownHosts"
}

while ($true) {
    & "$env:WINDIR\System32\OpenSSH\ssh.exe" `
        -N -T `
        -o BatchMode=yes `
        -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=30 `
        -o ServerAliveCountMax=3 `
        -o StrictHostKeyChecking=yes `
        -o "UserKnownHostsFile=$KnownHosts" `
        -R "127.0.0.1:$RemotePort`:127.0.0.1:$LocalPort" `
        $Remote `
        1>> (Join-Path $LogDir "hosted-tunnel.log") `
        2>> (Join-Path $LogDir "hosted-tunnel.err")
    Start-Sleep -Seconds 5
}
