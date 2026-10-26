# Reproduce the tested isolated baseline. Does not install drivers or alter Windows.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'preflight.ps1') -Mode BALANCED
$taskPython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $taskPython)) {uv venv --python 3.12 (Join-Path $PSScriptRoot '.venv');if($LASTEXITCODE){throw 'venv failed'}}
$taskPackages=@(
 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl',
 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl',
 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl',
 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz',
 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl'
)
$env:UV_CONCURRENT_INSTALLS='4'
$env:UV_CONCURRENT_DOWNLOADS='3'
uv pip install --python $taskPython @taskPackages -r (Join-Path $PSScriptRoot 'requirements.txt')
if($LASTEXITCODE){throw 'Dependency install failed'}
uv pip freeze --python $taskPython | Set-Content -LiteralPath (Join-Path $PSScriptRoot '..\.training-local\environment-freeze.txt') -Encoding utf8
