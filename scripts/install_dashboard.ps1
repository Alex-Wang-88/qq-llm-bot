$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskPythonw = Join-Path $env:APPDATA 'uv/tools/astrbot/Scripts/pythonw.exe'
$taskServer = Join-Path $taskRoot 'dashboard/server.py'
if (-not (Test-Path -LiteralPath $taskPythonw)) { throw "Pythonw not found: $taskPythonw" }
$taskStartup = [Environment]::GetFolderPath('Startup')
$taskLink = Join-Path $taskStartup 'Qianwen-Control-Panel.lnk'
$taskShell = New-Object -ComObject WScript.Shell
$taskShortcut = $taskShell.CreateShortcut($taskLink)
$taskShortcut.TargetPath = $taskPythonw
$taskShortcut.Arguments = '"' + $taskServer + '"'
$taskShortcut.WorkingDirectory = $taskRoot
$taskShortcut.Description = 'Local control panel; does not start the assistant'
$taskShortcut.Save()
Write-Host "Installed: $taskLink"
