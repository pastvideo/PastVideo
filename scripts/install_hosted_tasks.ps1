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
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650)
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$ApiBinary = Join-Path $RepoRoot "target\release\pastvideo.exe"
$DataDir = Join-Path $RepoRoot ".tools\web-data"
$ClipsDir = Join-Path $RepoRoot ".tools\web-clips"
$ApiAction = New-ScheduledTaskAction `
    -Execute $ApiBinary `
    -Argument "--data-dir `"$DataDir`" --backend qwen serve --bind 127.0.0.1:8787 --clips `"$ClipsDir`"" `
    -WorkingDirectory $RepoRoot

$Ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$KnownHostsPath = if ([System.IO.Path]::IsPathRooted($KnownHosts)) {
    $KnownHosts
} else {
    Join-Path $RepoRoot $KnownHosts
}
if (-not (Test-Path -LiteralPath $KnownHostsPath)) {
    throw "Pinned SSH host key file not found: $KnownHostsPath"
}
$TunnelAction = New-ScheduledTaskAction `
    -Execute $Ssh `
    -Argument "-N -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=`"$KnownHostsPath`" -R 127.0.0.1:${RemotePort}:127.0.0.1:${LocalPort} $Remote" `
    -WorkingDirectory $RepoRoot

foreach ($Task in @(
    @{ Name = "PastVideo Hosted API"; Action = $ApiAction },
    @{ Name = "PastVideo Hosted Tunnel"; Action = $TunnelAction }
)) {
    Stop-ScheduledTask -TaskName $Task.Name -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $Task.Name -Action $Task.Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
    Start-ScheduledTask -TaskName $Task.Name
    Write-Host "Installed and started: $($Task.Name)"
}
