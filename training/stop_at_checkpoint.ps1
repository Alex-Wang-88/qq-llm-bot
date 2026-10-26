param([Parameter(Mandatory=$true)][string]$Run,[int]$Step=30,[string]$Config='training/configs/sft-9b.json')
$ErrorActionPreference='Stop'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
$taskRun=(Resolve-Path -LiteralPath $Run).Path
$taskWatch=[Diagnostics.Stopwatch]::StartNew()
while($taskWatch.Elapsed.TotalMinutes -lt 25) {
    try {
        $taskSummary=Get-Content -LiteralPath (Join-Path $taskRun 'summary.json') -Raw | ConvertFrom-Json
        $taskState=Join-Path $taskRun (('checkpoint-{0:0000}' -f $Step)+'\training-state.pt')
        if($taskSummary.optimizer_steps -ge $Step -and (Test-Path -LiteralPath $taskState)) {
            $taskProcesses=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python.*\.exe$' -and $_.CommandLine.Contains('training/train.py') -and $_.CommandLine.Contains($Config)})
            $taskActual=@($taskProcesses | Where-Object {$taskProcesses.ProcessId -contains $_.ParentProcessId})
            if($taskActual.Count -ne 1){throw 'Expected exactly one owned training child'}
            Stop-Process -Id $taskActual[0].ProcessId
            [pscustomobject]@{stopped_pid=$taskActual[0].ProcessId;preserved_checkpoint=$taskState;step=$Step;time=(Get-Date).ToString('o')} | ConvertTo-Json | Tee-Object -FilePath (Join-Path $taskRun 'controlled-restart.json')
            exit 0
        }
        if($taskSummary.status -eq 'failed'){throw 'Training failed before requested checkpoint'}
    } catch {
        if($_.Exception.Message -match 'Expected exactly|Training failed'){throw}
    }
    Start-Sleep -Seconds 1
}
throw 'Checkpoint wait timed out; no process stopped'
