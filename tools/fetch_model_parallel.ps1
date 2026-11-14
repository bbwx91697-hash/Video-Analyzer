# large-v3 模型下载助手：16 路分段下载，按序合并并校验 SHA-256 后才正式命名。
# 64 MB 分块及并发数 16 用于改善该机器上单连接速度；分块长度不符时重新排队。
# Parallel ranged download helper for faster-whisper large-v3 model.bin from hf-mirror,
# verified by SHA-256. Single-connection HF/mirror speed on this PC was ~0.5-1.2 MB/s;
# 16 parallel range requests reached ~10 MB/s (3.09 GB in ~295 s).
$ErrorActionPreference = 'Stop'
$base = 'https://hf-mirror.com/Systran/faster-whisper-large-v3/resolve/main'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dst = Join-Path $repoRoot 'models\faster-whisper-large-v3'
$parts = Join-Path $repoRoot 'tmp\_sys\mparts'
New-Item -ItemType Directory -Force $dst, $parts | Out-Null
foreach ($f in 'config.json','preprocessor_config.json','tokenizer.json','vocabulary.json') { curl.exe -s -L --retry 5 -o "$dst\$f" "$base/$f" }
[int64]$size = 3087284237
[int64]$chunk = 64MB
$n = [math]::Ceiling($size / $chunk)
$start = Get-Date
$queue = [System.Collections.Queue]::new(); 0..($n-1) | % { $queue.Enqueue($_) }
$running = @{}
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  foreach ($k in @($running.Keys)) {
    $p = $running[$k]
    if ($p.HasExited) {
      $running.Remove($k)
      $exp = [math]::Min($chunk, [int64]($size - [int64]$k*$chunk))
      $len = if (Test-Path "$parts\p$k") { (Get-Item "$parts\p$k").Length } else { 0 }
      if ($len -ne $exp) { $queue.Enqueue($k) }
    }
  }
  while ($running.Count -lt 16 -and $queue.Count -gt 0) {
    $k = $queue.Dequeue(); [int64]$a = [int64]$k*$chunk; [int64]$b = [math]::Min([int64]($a+$chunk), $size) - 1
    $running[$k] = Start-Process curl.exe -ArgumentList "-s -L --retry 3 --speed-limit 20000 --speed-time 30 -r $a-$b -o $parts\p$k $base/model.bin" -NoNewWindow -PassThru
  }
  Start-Sleep -Milliseconds 500
}
$out = [System.IO.File]::Create("$dst\model.bin.partial")
for ($k=0; $k -lt $n; $k++) { $in = [System.IO.File]::OpenRead("$parts\p$k"); $in.CopyTo($out); $in.Close() }
$out.Close()
Remove-Item $parts -Recurse -Force
$h = (Get-FileHash "$dst\model.bin.partial" -Algorithm SHA256).Hash.ToLower()
"secs=$(((Get-Date)-$start).TotalSeconds) sha=$h"
if ($h -eq '69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1') { Move-Item "$dst\model.bin.partial" "$dst\model.bin" -Force; 'HASH OK' } else { 'HASH MISMATCH' }
