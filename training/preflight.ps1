param([ValidateSet('GAME','BALANCED','WORK','FULL')][string]$Mode='BALANCED')
$ErrorActionPreference='Stop'
$taskBefore=@{}
Get-Process | ForEach-Object { try {$taskBefore[$_.Id]=$_.TotalProcessorTime.TotalSeconds} catch {} }
$taskTimer=[Diagnostics.Stopwatch]::StartNew()
$taskSamples=@()
1..4 | ForEach-Object {
    $taskSamples+=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
    Start-Sleep -Milliseconds 800
}
$taskTimer.Stop()
$taskLogical=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$taskLoads=@(Get-Process | ForEach-Object {
    try {if($taskBefore.ContainsKey($_.Id)) {
        [pscustomobject]@{name=$_.ProcessName;pid=$_.Id;cpu=[math]::Round(100*($_.TotalProcessorTime.TotalSeconds-$taskBefore[$_.Id])/($taskTimer.Elapsed.TotalSeconds*$taskLogical),1);ram_mib=[math]::Round($_.WorkingSet64/1MB);window=$_.MainWindowTitle}
    }} catch {}
})
if(-not ('TrainingForeground' -as [type])) {
    Add-Type 'using System; using System.Runtime.InteropServices; public class TrainingForeground { [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow(); [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr w,out uint p); }'
}
[uint32]$taskForegroundId=0
[void][TrainingForeground]::GetWindowThreadProcessId([TrainingForeground]::GetForegroundWindow(),[ref]$taskForegroundId)
$taskForeground=$taskLoads | Where-Object pid -eq $taskForegroundId | Select-Object -First 1
$taskGames=@($taskLoads | Where-Object {$_.name -match '^(cs2|dota2|GenshinImpact|StarRail|ZenlessZoneZero|VALORANT.*|Cyberpunk2077|eldenring|javaw|Warframe.*|BlackMyth.*|MonsterHunter.*)$'})
$taskCpu=[math]::Round(($taskSamples | Measure-Object -Average).Average,1)
$taskOs=Get-CimInstance Win32_OperatingSystem
$taskRam=[math]::Round($taskOs.FreePhysicalMemory/1MB,2)
if($taskGames.Count -gt 0 -and $Mode -ne 'FULL') {$Mode='GAME'}
$taskWorkers=@{GAME=6;BALANCED=8;WORK=12;FULL=16}[$Mode]
# This machine exposes 8 logical processors despite the CPU's nominal 16T.
# Leave two scheduler slots for the foreground except explicit FULL mode.
$taskWorkers=[math]::Min($taskWorkers, $(if($Mode -eq 'FULL') {$taskLogical} else {[math]::Max(1,$taskLogical-2)}))
if($taskCpu -ge 50 -and $taskCpu -le 70) {$taskWorkers=[math]::Max(1,[math]::Floor($taskWorkers*.75))}
if($taskCpu -gt 70) {
    if($taskGames.Count -gt 0 -or $taskForeground.cpu -ge 2) {$taskWorkers=[math]::Min($taskWorkers,4)}
    else {throw "CPU $taskCpu%, background load needs inspection: $($taskLoads | Sort-Object cpu -Descending | Select-Object -First 5 | ConvertTo-Json -Compress)"}
}
if($Mode -eq 'GAME' -and $taskForeground.cpu -ge 20) {$taskWorkers=4}
$taskResult=[ordered]@{timestamp=(Get-Date).ToString('o');cpu_percent=$taskCpu;available_ram_gib=$taskRam;game_detected=($taskGames.Count -gt 0);mode=$Mode;workers=$taskWorkers;priority='BelowNormal';foreground=$taskForeground;top_cpu=@($taskLoads | Sort-Object cpu -Descending | Select-Object -First 8);existing_heavy=@($taskLoads | Where-Object name -match '^(blender|python.*|ollama.*|uv)$');games=$taskGames}
Write-Host "Preflight: CPU $taskCpu% | available RAM $taskRam GiB | game detected $($taskGames.Count -gt 0) | mode $Mode | workers $taskWorkers | priority BelowNormal"
Write-Host ($taskResult | ConvertTo-Json -Depth 5)
if($taskRam -lt 6) {throw 'Available RAM below 6 GiB; refusing a new job.'}
foreach($taskVar in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')) {[Environment]::SetEnvironmentVariable($taskVar,"$taskWorkers",'Process')}
$env:TRAINING_WORKERS="$taskWorkers"
$env:TOKENIZERS_PARALLELISM='false'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
$taskLocal=Join-Path $PSScriptRoot '..\.training-local'
New-Item -ItemType Directory -Path $taskLocal -Force | Out-Null
$taskResult | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLocal 'latest-preflight.json') -Encoding utf8
