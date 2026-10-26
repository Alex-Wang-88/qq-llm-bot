param([switch]$Ensure)
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskPython=Join-Path $env:APPDATA 'uv/tools/astrbot/Scripts/python.exe'
$taskExecutable=(Get-Command ollama).Source
$taskListener=Get-NetTCPConnection -State Listen -LocalPort 11434 -ErrorAction SilentlyContinue|Select-Object -First 1
if ($taskListener) {
    $taskServer=Get-CimInstance Win32_Process -Filter "ProcessId=$($taskListener.OwningProcess)"
    if ($taskServer.ExecutablePath -ne $taskExecutable -or $taskServer.CommandLine -notmatch '\bserve\b') {throw 'Port 11434 is not the expected Ollama service; left running.'}
    if ($Ensure) {
        $taskVersion=Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 3
        Write-Host "Existing Ollama $($taskVersion.version) retained; no duplicate service started."
        return
    }
}
. (Join-Path $PSScriptRoot 'resource_preflight.ps1')
if ($taskListener) {
    $taskParent=Get-CimInstance Win32_Process -Filter "ProcessId=$($taskServer.ParentProcessId)" -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='ollama.exe'"|Where-Object {$_.CommandLine -match '\bpull\s+qwen3\.5:9b-q4_K_M'}|ForEach-Object {Stop-Process -Id $_.ProcessId -Force}
    if ($taskParent -and $taskParent.Name -eq 'ollama app.exe' -and (Split-Path $taskParent.ExecutablePath) -eq (Split-Path $taskExecutable)) {Stop-Process -Id $taskParent.ProcessId -Force}
    Stop-Process -Id $taskServer.ProcessId -Force -ErrorAction SilentlyContinue
}
$env:HTTPS_PROXY='http://127.0.0.1:15732'
$env:NO_PROXY='localhost,127.0.0.1,::1'
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
$env:OLLAMA_NUM_PARALLEL='1'
$taskWrapper=Start-Process -FilePath $taskPython -ArgumentList @('-X','utf8',('"'+(Join-Path $PSScriptRoot 'run_ollama.py')+'"')) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRoot 'data/logs/ollama-launch.out.log') -RedirectStandardError (Join-Path $taskRoot 'data/logs/ollama-launch.err.log')
$taskWrapper.PriorityClass='BelowNormal'
$taskReady=$false
foreach ($taskTry in 1..40) {
    try {$taskVersion=Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 1;$taskReady=$true;break} catch {Start-Sleep -Milliseconds 250}
}
if (-not $taskReady) {throw 'Ollama did not become ready; cached model files are retained.'}
Write-Host "Ollama $($taskVersion.version) ready on the existing port; download proxy is process-local."
