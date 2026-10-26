<#
.SYNOPSIS
    从 GitHub Release 下载 4B GGUF 切片，拼接、校验，可选导入 Ollama。

.DESCRIPTION
    Release 单个 asset 上限 2 GiB，而 Q4_K_M 模型是 2.52 GiB，因此上游把它切成
    两个分片。本脚本下载分片 → 按序拼接 → 用 SHA256 校验拼接结果是否逐字节等于
    原始 GGUF → 删除分片（-KeepParts 可保留）。

    直连 GitHub 失败时会自动改用镜像（默认 https://gh-proxy.com/）；镜像同样支持
    断点续传，中断了直接重跑即可接着下。

    只依赖 PowerShell 5.1 和 curl.exe（Win10 1803+ 自带），不需要 gh、不需要 git。

.PARAMETER OutDir
    模型输出目录，默认当前目录下的 models\。

.PARAMETER MirrorFirst
    跳过直连、直接用镜像。GitHub 完全不通时加上这个，省掉每次 12 秒的连接超时。

.PARAMETER NoMirror
    只用直连，不走镜像。

.PARAMETER Import
    拼接校验通过后执行 ollama create（需要已安装 Ollama，且本机未验证过导入路径）。

.PARAMETER BaseUrl
    覆盖下载根地址，用于自建镜像 / 内网分发 / 本地测试（例如 http://127.0.0.1:8765）。
    给定时不再拼接 GitHub 地址，也不再走 gh-proxy。

.PARAMETER Proxy
    显式指定 HTTP 代理（如 127.0.0.1:15732 或 http://127.0.0.1:15732）。不给时自动读
    Windows 系统代理设置（VPN 客户端一般只改那里，而 curl 不读系统代理）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\fetch_model_release.ps1

.EXAMPLE
    .\fetch_model_release.ps1 -OutDir D:\models -MirrorFirst -Import
#>
[CmdletBinding()]
param(
    [string]$Repo = 'Alex-Wang-88/qq-llm-bot',
    [string]$Tag  = 'qwen35-4b-sft-v2',
    [string]$OutDir = (Join-Path (Get-Location).Path 'models'),
    [string]$Mirror = 'https://gh-proxy.com/',
    [string]$BaseUrl = '',
    [string]$Proxy = '',
    [string]$ExpectedMergedHash = '679b0f0ace7db44d25cf5e975e9ed6770f3bc049db2d4eb4d6f6d0221c0327b2',
    [long]$ExpectedMergedSize = 2708804096,
    [switch]$MirrorFirst,
    [switch]$NoMirror,
    [switch]$NoProxy,
    [switch]$KeepParts,
    [switch]$Import,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$ModelName = 'qwen35-4b-sft-v2-Q4_K_M.gguf'
$CustomBase = $BaseUrl -ne ''
if (-not $CustomBase) {
    $BaseUrl = "https://github.com/$Repo/releases/download/$Tag"
} else {
    $BaseUrl = $BaseUrl.TrimEnd('/')
    $NoMirror = [switch]$true
}
$Merged    = Join-Path $OutDir $ModelName
$ScriptName = 'fetch_model_release.ps1'

$script:Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
$script:UsedMirror = $false

# 代理：curl 不读 Windows 系统代理，而 VPN 客户端通常只改系统代理，
# 所以这里主动取出来传给 curl，否则直连 GitHub 会握手失败。
$script:ProxyUrl = ''
if (-not $NoProxy) {
    if ($Proxy -ne '') {
        $script:ProxyUrl = $Proxy
    } else {
        try {
            $ie = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction Stop
            if ($ie.ProxyEnable -eq 1 -and $ie.ProxyServer) {
                $srv = [string]$ie.ProxyServer
                if ($srv.Contains('=')) {
                    $map = @{}
                    foreach ($kv in $srv.Split(';')) {
                        $pair = $kv.Split('=')
                        if ($pair.Count -eq 2) { $map[$pair[0]] = $pair[1] }
                    }
                    if ($map.ContainsKey('https')) { $srv = $map['https'] }
                    elseif ($map.ContainsKey('http')) { $srv = $map['http'] }
                }
                $script:ProxyUrl = $srv
            }
        } catch {
            $script:ProxyUrl = ''
        }
    }
}
if ($script:ProxyUrl -ne '' -and $script:ProxyUrl -notmatch '^[a-zA-Z]+://') {
    $script:ProxyUrl = "http://$($script:ProxyUrl)"
}

function Write-Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "    $text" -ForegroundColor Green }
function Write-Note($text) { Write-Host "    $text" -ForegroundColor Yellow }

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-FreeBytes([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetPathRoot($full)
    return (New-Object System.IO.DriveInfo $root).AvailableFreeSpace
}

# 原生命令（curl）会把进度条写到 stderr。$ErrorActionPreference='Stop' 时 PowerShell 5.1
# 会把任何 stderr 输出升级成终止性 NativeCommandError，脚本会在一开始下载时就被打断。
# 所以只在调用期间降到 Continue，让进度条正常渲染，成败仍以退出码判断。
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    return $code
}

# 直连优先、失败降级镜像。每个候选内部靠 curl -C - 断点续传，整轮最多三遍。
function Get-RemoteFile {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Dest,
        [int]$MaxTimeSec = 0
    )

    $candidates = @()
    if (-not $NoMirror) {
        if ($MirrorFirst) {
            $candidates += "MIRROR|$($Mirror)$Url"
            $candidates += "DIRECT|$Url"
        } else {
            $candidates += "DIRECT|$Url"
            $candidates += "MIRROR|$($Mirror)$Url"
        }
    } else {
        $candidates += "DIRECT|$Url"
    }

    $lastErr = '未知错误'
    for ($round = 1; $round -le 3; $round++) {
        foreach ($cand in $candidates) {
            $kind = $cand.Split('|')[0]
            $uri  = $cand.Substring($cand.IndexOf('|') + 1)
            if ($kind -eq 'MIRROR') { Write-Note '直连不通，改走镜像' }

            if ($script:Curl) {
                $curlArgs = @('-fL', '-C', '-', '--retry', '2', '--retry-delay', '2',
                              '--connect-timeout', '12', '--speed-limit', '2048', '--speed-time', '30')
                if ($MaxTimeSec -gt 0) { $curlArgs += @('--max-time', "$MaxTimeSec") }
                # -x 显式指定代理时已覆盖环境变量里的代理，不需要再加 --noproxy
                if ($script:ProxyUrl -ne '') { $curlArgs += @('-x', $script:ProxyUrl) }
                $curlArgs += @('-o', $Dest, $uri)
                $code = Invoke-Native -Exe $script:Curl.Source -Arguments $curlArgs
                if ($code -eq 0) {
                    if ($kind -eq 'MIRROR') { $script:UsedMirror = $true }
                    return
                }
                $lastErr = "curl 退出码 $code"
            } else {
                try {
                    if ($script:ProxyUrl -ne '') {
                        Invoke-WebRequest -Uri $uri -OutFile $Dest -UseBasicParsing -Proxy $script:ProxyUrl
                    } else {
                        Invoke-WebRequest -Uri $uri -OutFile $Dest -UseBasicParsing
                    }
                    return
                } catch {
                    $lastErr = $_.Exception.Message
                }
            }

            if (Test-Path $Dest) {
                Write-Note "失败（$lastErr），已收到 $((Get-Item $Dest).Length) 字节，换地址续传"
            } else {
                Write-Note "失败（$lastErr）"
            }
        }
        if ($round -lt 3) {
            Write-Note "所有地址都失败，10 秒后重试（第 $round/3 轮）"
            Start-Sleep -Seconds 10
        }
    }
    throw "下载失败：$Url`n最后错误：$lastErr`n`n直连 GitHub 不通很常见，加 -MirrorFirst 强制走镜像：`n  powershell -ExecutionPolicy Bypass -File $ScriptName -MirrorFirst"
}

# ---------- 0. 准备 ----------
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}
Write-Step "输出目录 $OutDir"

if ($script:ProxyUrl -ne '') {
    Write-Step "使用代理 $($script:ProxyUrl)"
} elseif ($NoProxy) {
    Write-Step '已按 -NoProxy 禁用代理'
}

if (-not $script:Curl) {
    Write-Note 'curl.exe 未找到，回退到 Invoke-WebRequest（不支持断点续传，大文件容易失败）'
}

# ---------- 1. 取清单 ----------
Write-Step '读取 SHA256SUMS.txt'
$sumsPath = Join-Path $OutDir 'SHA256SUMS.txt'
Get-RemoteFile -Url "$BaseUrl/SHA256SUMS.txt" -Dest $sumsPath -MaxTimeSec 90

$parts = @()
foreach ($line in Get-Content -Path $sumsPath) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith('#')) { continue }
    if ($t -match '^([0-9a-fA-F]{64})\s+\*?(\S+)$') {
        $parts += [pscustomobject]@{ Hash = $matches[1].ToLowerInvariant(); Name = $matches[2] }
    }
}
if ($parts.Count -eq 0) { throw 'SHA256SUMS.txt 里没有可分片条目，文件可能损坏' }
$parts = @($parts | Sort-Object Name)
Write-Ok "$($parts.Count) 个分片：$($parts.Name -join ', ')"

# ---------- 2. 空间检查 ----------
$needBytes = ($ExpectedMergedSize * 2) + 200MB
$freeBytes = Get-FreeBytes $OutDir
$needGb = [math]::Round($needBytes / 1GB, 2)
$freeGb = [math]::Round($freeBytes / 1GB, 2)
Write-Step "磁盘：需要约 $needGb GiB（分片 + 拼接），可用 $freeGb GiB"
if ($freeBytes -lt $needBytes) {
    throw "磁盘空间不足：需要 $needGb GiB，只有 $freeGb GiB。用 -OutDir 换一个盘。"
}

# ---------- 3. 下载并校验每个分片 ----------
foreach ($p in $parts) {
    $dest = Join-Path $OutDir $p.Name
    if ((Test-Path $dest) -and -not $Force) {
        $have = Get-Sha256 $dest
        if ($have -eq $p.Hash) {
            Write-Step "$($p.Name) 已存在且哈希正确，跳过"
            continue
        }
        Write-Note "$($p.Name) 已存在但哈希不符，重新下载"
        Remove-Item -Force $dest
    }

    Write-Step "下载 $($p.Name)"
    Get-RemoteFile -Url "$BaseUrl/$($p.Name)" -Dest $dest

    $got = Get-Sha256 $dest
    if ($got -ne $p.Hash) {
        throw "$($p.Name) 哈希不符`n  期望 $($p.Hash)`n  实际 $got`n删掉这个文件重跑即可（会断点续传）。"
    }
    Write-Ok "哈希一致 $got"
}

# ---------- 4. 拼接 ----------
$skipMerge = $false
if ((Test-Path $Merged) -and -not $Force) {
    if ((Get-Sha256 $Merged) -eq $ExpectedMergedHash) {
        Write-Step '拼接结果已存在且哈希正确，跳过拼接'
        $skipMerge = $true
    } else {
        Write-Note '已有文件哈希不符，重新拼接'
        Remove-Item -Force $Merged
    }
}

if (-not $skipMerge) {
    Write-Step "拼接 $($parts.Count) 个分片 → $ModelName"
    $out = [System.IO.File]::Create($Merged)
    try {
        foreach ($p in $parts) {
            $src = Join-Path $OutDir $p.Name
            $in = [System.IO.File]::OpenRead($src)
            try { $in.CopyTo($out) } finally { $in.Dispose() }
        }
    } finally {
        $out.Dispose()
    }
}

# ---------- 5. 校验拼接结果 ----------
Write-Step '校验拼接结果'
$size = (Get-Item $Merged).Length
if ($size -ne $ExpectedMergedSize) {
    throw "大小不符：$size 字节，应为 $ExpectedMergedSize 字节。分片顺序或数量有问题。"
}
$hash = Get-Sha256 $Merged
if ($hash -ne $ExpectedMergedHash) {
    throw "哈希不符：`n  期望 $ExpectedMergedHash`n  实际 $hash"
}
Write-Ok "大小 $size 字节，SHA256 $hash"

if (-not $KeepParts) {
    Write-Step '删除分片'
    foreach ($p in $parts) { Remove-Item -Force (Join-Path $OutDir $p.Name) }
    Write-Ok '已删除（保留 SHA256SUMS.txt 备查）'
}

# ---------- 6. 可选：导入 Ollama ----------
if ($Import) {
    Write-Step '导入 Ollama'
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) {
        Write-Note '未找到 ollama，跳过。装好后手动执行：'
        Write-Note '  ollama create qwen3.5:4b-qq-v2 -f config\Modelfile.qianwen35-4b'
    } else {
        $modelfile = Join-Path $PSScriptRoot '..\config\Modelfile.qianwen35-4b'
        if (-not (Test-Path $modelfile)) {
            Write-Note "找不到 $modelfile，跳过导入"
        } else {
            $tmp = Join-Path $env:TEMP 'Modelfile.qwen35-4b.generated'
            $content = Get-Content $modelfile | ForEach-Object {
                if ($_ -match '^FROM\s') { "FROM $Merged" } else { $_ }
            }
            Set-Content -Path $tmp -Value $content -Encoding ASCII
            & $ollama.Source create qwen3.5:4b-qq-v2 -f $tmp
            if ($LASTEXITCODE -ne 0) {
                Write-Note "ollama create 失败（退出码 $LASTEXITCODE），模型文件本身已校验通过"
            } else {
                Write-Ok '已创建 qwen3.5:4b-qq-v2'
                Write-Note 'Ollama 关闭 thinking 的写法未验证：输出出现 [Start thinking] 或 <think> 即未生效'
            }
        }
    }
}

# ---------- 7. 下一步 ----------
Write-Host ''
Write-Host '完成。模型：' -NoNewline
Write-Host $Merged -ForegroundColor Green
if ($script:UsedMirror) {
    Write-Host '（本次走了镜像，但文件哈希已校验，与源文件逐字节一致）' -ForegroundColor DarkGray
}
Write-Host ''
Write-Host 'llama.cpp 启动（8 GB 显存足够，注意必须关 thinking）：'
Write-Host "  llama-server.exe -m `"$Merged`" -ngl 99 -c 8192 --port 8080 --chat-template-kwargs '{`"enable_thinking`": false}'"
Write-Host ''
Write-Host 'Ollama 导入：'
Write-Host '  ollama create qwen3.5:4b-qq-v2 -f config\Modelfile.qianwen35-4b'
