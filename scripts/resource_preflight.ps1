$taskCpuSamples = @()
$taskBefore = @{}
Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
    try { $taskBefore[$_.Id] = $_.TotalProcessorTime.TotalSeconds } catch {}
}
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
1..4 | ForEach-Object {
    $taskCpuSamples += (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
    Start-Sleep -Milliseconds 750
}
$taskWatch.Stop()
$taskCpu = [math]::Round(($taskCpuSamples | Measure-Object -Average).Average, 1)
$taskOs = Get-CimInstance Win32_OperatingSystem
$taskRamActual = $taskOs.FreePhysicalMemory / 1MB
$taskRam = [math]::Round($taskRamActual, 1)
$taskLogical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$taskProcesses = @(Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        if ($taskBefore.ContainsKey($_.Id)) {
            [pscustomobject]@{Id=$_.Id;Name=$_.ProcessName;CPU=[math]::Round(100*($_.TotalProcessorTime.TotalSeconds-$taskBefore[$_.Id])/($taskWatch.Elapsed.TotalSeconds*$taskLogical),1);RAM_MB=[math]::Round($_.WorkingSet64/1MB)}
        }
    } catch {}
})
if (-not ('ChatForegroundWindow' -as [type])) {
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class ChatForegroundWindow { [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow(); [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint processId); }'
}
[uint32]$taskForegroundId = 0
[void][ChatForegroundWindow]::GetWindowThreadProcessId([ChatForegroundWindow]::GetForegroundWindow(), [ref]$taskForegroundId)
$taskForeground = Get-Process -Id $taskForegroundId -ErrorAction SilentlyContinue
$taskMode = if ($taskCpu -ge 50 -or $taskRam -lt 8) {'LIGHT'} else {'NORMAL'}
$taskWorkers = if ($taskMode -eq 'LIGHT') {4} else {6}
$taskForegroundCpu = ($taskProcesses | Where-Object {$_.Id -eq $taskForegroundId} | Select-Object -First 1).CPU
if ($taskForeground.ProcessName) {
    $taskAppCpu=($taskProcesses|Where-Object {$_.Name -eq $taskForeground.ProcessName}|Measure-Object -Property CPU -Sum).Sum
    $taskForegroundCpu=[math]::Max([double]$taskForegroundCpu,[double]$taskAppCpu)
}
if ($taskForegroundCpu -ge 2) {$taskMode='LIGHT';$taskWorkers=4}
Write-Host "Preflight: CPU $taskCpu% | available RAM ${taskRam}GB | foreground $($taskForeground.ProcessName) | mode $taskMode | workers $taskWorkers | priority BelowNormal"
Write-Host ('Highest CPU: '+ (($taskProcesses|Sort-Object CPU -Descending|Select-Object -First 6|ForEach-Object {"$($_.Name)[$($_.Id)] $($_.CPU)%"}) -join ', '))
Write-Host ('Existing Blender/Python: '+ (($taskProcesses|Where-Object {$_.Name -match '^(python|blender|ollama)'}|Sort-Object CPU -Descending|Select-Object -First 8|ForEach-Object {"$($_.Name)[$($_.Id)] $($_.CPU)%"}) -join ', '))
if ($taskRamActual -lt 4) {throw 'Available RAM below 4GB; no new heavy jobs.'}
if ($taskCpu -gt 70 -and $taskForegroundCpu -lt 2) {throw 'CPU over 70%; inspect the listed background jobs before starting.'}
foreach ($taskThreadVar in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')) {
    [Environment]::SetEnvironmentVariable($taskThreadVar, "$taskWorkers", 'Process')
}
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
