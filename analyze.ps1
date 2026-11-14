# 流水线启动入口：输入链接、本地视频或已有结果目录，传递模型、设备、媒体保留、低码率代理及增量追加 Clips 选项。
# 进程级环境变量将临时文件和缓存放在 D 盘；下载由现有下载器的独立 venv 执行。
[CmdletBinding()]
param(
    [Parameter(Position=0)][string]$Url,
    [string]$Video,
    [string]$AppendClips,
    [string]$BatchFile,
    [string]$ResumeBatch,
    [switch]$RetryDegraded,
    [switch]$StopOnError,
    [int]$BatchTimeout = 0,
    [string]$YtdlpBin,
    [switch]$KeepMedia,
    [switch]$SaveProxy,
    [int]$ProxyMaxDim = 0,
    [double]$ProxyFps = 0,
    [double]$MaxProxyMb = 0,
    [double]$MaxResultMb = 0,
    [string]$Model = 'large-v3',
    [ValidateSet('auto', 'cuda', 'cpu')][string]$Device = 'auto',
    [string]$Clips = '',
    [int]$MaxOcrFrames = 0,
    [switch]$HfMirror,
    [string]$FetchPython = $(if ($env:DOUYIN_FETCH_PYTHON) { $env:DOUYIN_FETCH_PYTHON } else { Join-Path $env:USERPROFILE 'douyin-browser-mcp\.venv\Scripts\python.exe' })
)
$root = $PSScriptRoot
if (Test-Path 'D:\ffmpeg\bin') {
    $env:PATH = 'D:\ffmpeg\bin;' + $env:PATH
}
$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobeCmd = Get-Command ffprobe -ErrorAction SilentlyContinue
if ($ffmpegCmd) { $env:DOUYIN_MCP_FFMPEG = $ffmpegCmd.Source }
if ($ffprobeCmd) { $env:DOUYIN_MCP_FFPROBE = $ffprobeCmd.Source }
$env:DOUYIN_MCP_BROWSER_CHANNEL = 'chrome'
$env:TEMP = Join-Path $root 'tmp\_sys'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $root 'cache\pip'
$env:UV_CACHE_DIR = Join-Path $root 'cache\uv'
$env:HF_HOME = Join-Path $root 'models\hf'
$env:HF_HUB_CACHE = Join-Path $root 'models\hf\hub'
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:XDG_CACHE_HOME = Join-Path $root 'cache\xdg'
$env:TORCH_HOME = Join-Path $root 'cache\torch'
$env:PYTHONPYCACHEPREFIX = Join-Path $root 'cache\pycache'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
if ($HfMirror) { $env:HF_ENDPOINT = 'https://hf-mirror.com' }
foreach ($dir in @($env:TEMP, $env:PIP_CACHE_DIR, $env:UV_CACHE_DIR, $env:HF_HOME, $env:HF_HUB_CACHE, $env:XDG_CACHE_HOME, $env:TORCH_HOME, $env:PYTHONPYCACHEPREFIX)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$script = Join-Path $root 'analyze.py'
if (-not (Test-Path $FetchPython)) {
    $fallbackFetch = Join-Path $env:USERPROFILE 'douyin-browser-mcp\.venv\Scripts\python.exe'
    if (Test-Path $fallbackFetch) { $FetchPython = $fallbackFetch }
}
if ($BatchFile -or $ResumeBatch) {
    if ($Url -or $Video -or $AppendClips) {
        Write-Error '-BatchFile / -ResumeBatch cannot be specified with URL, -Video, or -AppendClips.'
        exit 2
    }
    if ($BatchFile -and $ResumeBatch) {
        Write-Error '-BatchFile and -ResumeBatch cannot be specified together.'
        exit 2
    }
} else {
    if ($Url -and $AppendClips) {
        Write-Error 'URL and -AppendClips cannot be specified together.'
        exit 2
    }
    if ($AppendClips -and -not $Clips) {
        Write-Error '-AppendClips requires -Clips.'
        exit 2
    }
    if ($Url -and $Video) {
        Write-Error 'URL and -Video cannot be specified together.'
        exit 2
    }
    if (-not $Url -and -not $Video -and -not $AppendClips) {
        Write-Error 'Either URL, -Video, -AppendClips, -BatchFile, or -ResumeBatch is required.'
        exit 2
    }
}
$arguments = @($script, '--fetch-python', $FetchPython, '--model', $Model, '--device', $Device, '--results-root', (Join-Path $root 'results'), '--tmp-root', (Join-Path $root 'tmp'))
if ($Url) { $arguments += $Url }
if ($Video) { $arguments += @('--video', $Video) }
if ($AppendClips) { $arguments += @('--append-clips', $AppendClips) }
if ($BatchFile) { $arguments += @('--batch-file', $BatchFile) }
if ($ResumeBatch) { $arguments += @('--resume-batch', $ResumeBatch) }
if ($RetryDegraded) { $arguments += '--retry-degraded' }
if ($StopOnError) { $arguments += '--stop-on-error' }
if ($BatchTimeout -gt 0) { $arguments += @('--batch-timeout', $BatchTimeout) }
if ($YtdlpBin) { $arguments += @('--ytdlp-bin', $YtdlpBin) }
if ($KeepMedia) { $arguments += '--keep-media' }
if ($SaveProxy) { $arguments += '--save-proxy' }
if ($ProxyMaxDim -gt 0) { $arguments += @('--proxy-max-dim', $ProxyMaxDim) }
if ($ProxyFps -gt 0) { $arguments += @('--proxy-fps', $ProxyFps) }
if ($MaxProxyMb -gt 0) { $arguments += @('--max-proxy-mb', $MaxProxyMb) }
if ($MaxResultMb -gt 0) { $arguments += @('--max-result-mb', $MaxResultMb) }
if ($Clips) { $arguments += @('--clips', $Clips) }
if ($MaxOcrFrames -gt 0) { $arguments += @('--max-ocr-frames', $MaxOcrFrames) }
& $python @arguments
exit $LASTEXITCODE
