# 环境搭建入口：设置进程级项目目录缓存，创建 venv、安装依赖并预下载指定模型。
# 此脚本未在目标机器实测；失败时保留日志并返回退出码。
[CmdletBinding()]
param(
    [string]$Model = 'large-v3',
    [switch]$HfMirror,
    [string]$BasePython = $env:DOUYIN_BASE_PYTHON
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
if (-not $BasePython) {
    $cand = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (Test-Path $cand) { $BasePython = $cand }
    else {
        $cmdPy = Get-Command python -ErrorAction SilentlyContinue
        if ($cmdPy) { $BasePython = $cmdPy.Source }
    }
}
if (-not $BasePython -or -not (Test-Path $BasePython)) {
    Write-Error "Base Python not found: $BasePython"
    exit 1
}
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'setup.log'
"Setup started: $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
& $BasePython -m venv (Join-Path $root '.venv') 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venvPython -m pip install -r (Join-Path $root 'requirements.txt') 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venvPython (Join-Path $root 'prefetch_model.py') --model $Model 2>&1 | Tee-Object -FilePath $log -Append
exit $LASTEXITCODE
