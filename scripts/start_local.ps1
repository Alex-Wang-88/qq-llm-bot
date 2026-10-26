param([switch]$Restart, [switch]$StopOnly)
$ErrorActionPreference='Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskLogs = Join-Path $taskRoot 'data/logs'
[void](New-Item -ItemType Directory -Force -Path $taskLogs)
# Written before any service log exists, so a double-click that dies early still
# leaves a readable reason instead of only a console message.
$taskTranscript = Join-Path $taskLogs 'startup.log'

function Write-Step([string]$Message) {
    Add-Content -LiteralPath $taskTranscript -Encoding utf8 -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message)
    Write-Host $Message
}

# PowerShell 5.1 copies the raw environment block when starting a process, so a
# duplicate pair such as HTTPS_PROXY plus https_proxy aborts Start-Process with
# "An item with the same key has already been added". Rewriting every variable
# once collapses the case-duplicate pair and the launch proceeds normally.
function Repair-ProcessEnvironment {
    $snapshot = [System.Environment]::GetEnvironmentVariables('Process')
    $names = @($snapshot.Keys)
    try {
        foreach ($name in $names) { [System.Environment]::SetEnvironmentVariable($name, $null, 'Process') }
    } finally {
        foreach ($name in $names) { [System.Environment]::SetEnvironmentVariable($name, [string]$snapshot[$name], 'Process') }
    }
}

trap {
    Write-Step ('FAILED: ' + $_.Exception.Message)
    Write-Step ('at line ' + $_.InvocationInfo.ScriptLineNumber + ': ' + ("$($_.InvocationInfo.Line)").Trim())
    Write-Step ('transcript: ' + $taskTranscript)
    exit 1
}

Repair-ProcessEnvironment
Write-Step "start root=$taskRoot restart=$Restart stoponly=$StopOnly"
$taskPython = Join-Path $env:APPDATA 'uv/tools/astrbot/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {throw "AstrBot Python not found: $taskPython"}
. (Join-Path $PSScriptRoot 'resource_preflight.ps1')
Write-Step "preflight ok; python=$taskPython"
function Get-TaskListener([int]$Port) {
    Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue|Select-Object -First 1
}
function Stop-TaskListener([int]$Port, [string]$Pattern, [switch]$WithParents) {
    $taskListener = Get-TaskListener $Port
    if (-not $taskListener) {return}
    $taskProc = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskListener.OwningProcess)"
    if (-not $taskProc -or $taskProc.CommandLine -notmatch $Pattern) {throw "Port $Port belongs to an unrelated process; left running."}
    $taskChain = @($taskProc)
    if ($WithParents) {
        1..2 | ForEach-Object {
            $taskParent = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskChain[-1].ParentProcessId)" -ErrorAction SilentlyContinue
            if ($taskParent -and $taskParent.CommandLine -match $Pattern) {$taskChain += $taskParent}
        }
    }
    foreach ($taskItem in $taskChain) {
        $taskCurrent = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskItem.ProcessId)" -ErrorAction SilentlyContinue
        if ($taskCurrent -and $taskCurrent.CommandLine -eq $taskItem.CommandLine) {Stop-Process -Id $taskCurrent.ProcessId -Force}
    }
}
if ($Restart) {
    # Match the known services before stopping; never terminate arbitrary Python jobs.
    Stop-TaskListener 6185 '(astrbot\.exe.*run|run_astrbot\.py)' -WithParents
    Stop-TaskListener 8899 '(ollama_fix_proxy|_debug_proxy)\.py'
    Write-Step 'known services stopped'
}
if ($StopOnly) {
    if (-not $Restart) {throw '-StopOnly requires -Restart.'}
    Write-Step 'Known local bot services stopped.'
    exit 0
}
if (-not (Get-TaskListener 11434)) {
    Write-Step 'starting ollama'
    & (Join-Path $PSScriptRoot 'restart_ollama_proxy.ps1') -Ensure
    Write-Step 'ollama ensured'
}
if (-not (Get-TaskListener 8899)) {
    Write-Step 'starting proxy on 8899'
    $taskProxy = Start-Process -FilePath $taskPython -ArgumentList @('-X','utf8',('"'+(Join-Path $taskRoot 'ollama_fix_proxy.py')+'"')) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'proxy.out.log') -RedirectStandardError (Join-Path $taskLogs 'proxy.err.log')
    $taskProxy.PriorityClass='BelowNormal'
    Write-Step "proxy pid $($taskProxy.Id)"
}
$taskHealth = $null
1..20 | ForEach-Object {
    if (-not $taskHealth) {
        try {$taskHealth=Invoke-RestMethod 'http://127.0.0.1:8899/health' -TimeoutSec 1} catch {Start-Sleep -Milliseconds 250}
    }
}
if ($taskHealth.proxy -ne 'chat-tuning-v1') {throw 'Port 8899 is not the tuned proxy. Use -Restart to replace the known old proxy.'}
Write-Step "proxy healthy: $($taskHealth.proxy)"
if (-not (Get-TaskListener 6185)) {
    $taskBot = Start-Process -FilePath $taskPython -ArgumentList @('-X','utf8',('"'+(Join-Path $PSScriptRoot 'run_astrbot.py')+'"')) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'astrbot.out.log') -RedirectStandardError (Join-Path $taskLogs 'astrbot.err.log')
    $taskBot.PriorityClass='BelowNormal'
    Write-Step "astrbot pid $($taskBot.Id)"
    $taskReady=$false
    foreach ($taskAttempt in 1..90) {
        $taskClient=[Net.Sockets.TcpClient]::new()
        try {$taskClient.Connect('127.0.0.1',6185);$taskReady=$true} catch {} finally {$taskClient.Dispose()}
        if ($taskReady) {break}
        $taskBot.Refresh()
        if ($taskBot.HasExited) {throw 'AstrBot exited during startup; inspect data/logs/astrbot.err.log.'}
        Start-Sleep -Milliseconds 500
    }
    if (-not $taskReady) {throw 'AstrBot dashboard did not become ready; inspect data/logs/astrbot.out.log.'}
} else {Write-Step 'AstrBot is already listening; no duplicate process started.'}
Write-Step "ready: http://127.0.0.1:6185  (transcript: $taskTranscript)"
# Explicit exit code: the launcher checks it, so it must not depend on whatever
# the last native command happened to leave behind.
Write-Step 'exit 0'
exit 0
