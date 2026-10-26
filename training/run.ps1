param(
    [ValidateSet('GAME','BALANCED','WORK','FULL')][string]$Mode='BALANCED',
    [string]$Python=(Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    [Parameter(Mandatory=$true)][string]$Script,
    [string[]]$ScriptArgs=@()
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'preflight.ps1') -Mode $Mode
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskLocal=Join-Path $taskRoot '.training-local'
$env:HF_HOME=Join-Path $taskLocal 'hf'
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:TRAINING_ROOT=$taskLocal
$taskJobKind=if([IO.Path]::GetFileName($Script) -eq 'prepare_model.py') {'download'} else {'compute'}
$taskLockPath=Join-Path $taskLocal ($taskJobKind+'-job.lock')
try {$taskLock=[IO.File]::Open($taskLockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)}
catch {throw "Another $taskJobKind job owns the resource lock. Do not start a duplicate."}
try {
    $taskRunning=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python.*\.exe$' -and $_.CommandLine -match 'training[\\/].*(train|gpu_smoke|evaluate)\.py'})
    if($taskRunning.Count -and $taskJobKind -eq 'compute') {throw "Existing training processes: $($taskRunning.ProcessId -join ', ')"}
    Set-Location -LiteralPath $taskRoot
    & $Python -u $Script @ScriptArgs
    if($LASTEXITCODE -ne 0) {throw "Python exit code $LASTEXITCODE"}
} finally {$taskLock.Dispose()}
