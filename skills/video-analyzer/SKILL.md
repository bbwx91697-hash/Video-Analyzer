---
name: video-analyzer
description: >-
  Analyze Douyin, Bilibili, YouTube, or local videos using the local video-analyzer pipeline.
  Use when the user asks to parse, transcribe, extract keyframes/storyboards/OCR/audio energy,
  cut frame clips, run serial batch analysis, or resume a failed batch on video URLs or local video files.
  触发场景：用户要求分析抖音/B站/YouTube链接或本地视频、提取分镜/故事板/语音转写/字幕OCR/音频能量、截取片段(-Clips)、批量串行跑视频或断点续跑批次。
---

# Video Analyzer

Run the local GPU/FFmpeg video analysis pipeline via `analyze.ps1` in the repository root (`schema 1.8`, `vlm_experimental=false`).

## Operational Constraints

- **Run serially**: Never launch concurrent video analysis jobs against the same machine; pipeline runs share GPU VRAM (`faster-whisper` + `RapidOCR`) and optional headed browser sessions.
- **Keep artifacts in the project directory**: Store temporary assets, model caches, and results inside the repository's `tmp/`, `models/`, and `results/` folders.
- **Whitelist-only URLs**: Only Douyin (`douyin.com`, `iesdouyin.com`, `amemv.com`, `v.douyin.com`), Bilibili (`bilibili.com`, `b23.tv`, `bili2233.cn`), and YouTube (`youtube.com`, `youtu.be`, `youtube-nocookie.com`) URLs are allowed. Do not attempt workarounds or fallback downloaders for unsupported domains.
- **No fabricated findings**: Report only metrics, transcripts, OCR, warnings, and visual details directly observed in output files and images.

## Quick Commands (PowerShell)

Always invoke `analyze.ps1` (from `<REPO_ROOT>`) so project-local caches, UTF-8 encoding, and FFmpeg paths are configured automatically.

### 1. Single Video (URL or Local File)

```powershell
# Douyin / Bilibili / YouTube URL
& '.\analyze.ps1' '<URL>' [-Clips '00:00-00:05'] [-SaveProxy] [-KeepMedia] [-MaxOcrFrames 96] [-Device auto|cuda|cpu]

# Local video (Unicode/space paths supported)
& '.\analyze.ps1' -Video 'path\to\video.mp4' [-Clips '00:03-00:09'] [-SaveProxy]
```

Key single-run flags:
- `-Clips "00:00-00:05,00:20-00:25"`: Export 10 fps clip strips (`clips/*.jpg`) for specific time windows.
- `-SaveProxy`: Keep lightweight low-bitrate `proxy/proxy.mp4` for later clip extraction without keeping the full source video.
- `-KeepMedia`: Retain the downloaded original video (normally auto-cleaned after proxy & report persistence).
- `-MaxOcrFrames N`: Cap OCR frame budget (reports coverage gaps honestly if exceeded).

### 2. Append Clips to an Existing Result

Reuses `proxy/proxy.mp4` automatically, or re-validates `-Video` via SHA256 if no proxy was saved:

```powershell
& '.\analyze.ps1' -AppendClips '.\results\<result_dir>' -Clips '00:15-00:22' [-Video 'path\to\original.mp4']
```

### 3. Serial Batch Analysis (`-BatchFile`)

Create a `.txt` or `.json` manifest (`.txt` ignores blank lines and `#` comments):

```text
# tasks.txt format: <url_or_local_path> | optional key=value overrides
https://v.douyin.com/example/ | save_proxy=true
https://www.bilibili.com/video/BV1jMWY6JEx7 | clips=00:00-00:04
videos\sample.mp4 | label=sample_clip | max_ocr_frames=48
```

Supported per-line overrides in `.txt` (and object keys in `.json` `tasks` list): `clips`, `label`, `save_proxy`, `keep_media`, `max_ocr_frames`.

Run serially:

```powershell
& '.\analyze.ps1' -BatchFile 'path\to\tasks.txt' [-StopOnError] [-BatchTimeout 1800]
```

- Default behavior isolates per-task failures (`failed`) and continues remaining tasks.
- `-StopOnError` halts after the first failure and marks remaining tasks `not_executed`.
- `-BatchFile` and `-ResumeBatch` are mutually exclusive with each other and with `<URL>`, `-Video`, and `-AppendClips` (exit code `2` on conflict).

### 4. Resume Failed Batch (`-ResumeBatch`)

Creates a new `batch-<new_id>` folder, reuses previously `success` (and `degraded` unless `-RetryDegraded` is passed) task outputs without overwriting old directories, and reruns only unfinished/failed tasks:

```powershell
& '.\analyze.ps1' -ResumeBatch '.\results\batch-<old_id>' [-RetryDegraded]
```

## Reading Results & Verification

Check the exit code and the final JSON status line printed to stdout, then inspect the output directory:

1. **Single run (`<result_dir>` under `results\`)**:
   - Read `analysis.md` for narrative summary, ASR segments, OCR events, audio energy notes, metadata sources, and explicit `missing` field reasons.
   - Read `analysis.json` (`schema_version: "1.8"`) for structured timestamps, `status`, `cleanup_ok`, and budgets.
   - Inspect visual outputs hierarchically to save quota: start with `analysis.md` + `storyboard.jpg` (add `transitions.jpg` or the single matching `timeline_00X.jpg` only if details are unclear). **Do NOT** run `-Clips` / `-AppendClips` on the whole video just to force 10 fps; only export `-Clips` for narrow time windows when the user explicitly asks for second-level zoom (`vlm_experimental=false`).
2. **Batch run (`results\batch-<id>\`)**:
   - Read `batch_summary.md` and `batch_index.json` for counts (`total`, `success`, `degraded`, `failed`, `not_executed`), per-task warnings, sampling limits, and links to each task's `analysis.md` and `analysis.json`.

## Common Failure Modes

- **Exit code `1` with unsupported host / route error**: URL hostname is outside the Douyin/Bilibili/YouTube whitelist. Report the rejection rather than scraping manually.
- **`TARGET_STREAM_UNBOUND` on Douyin**: Browser capture could not bind the target stream (the downloader intentionally refuses falling back to unrelated feed videos). Ask the user to check the logged-in browser session or retry.
- **External HTTP rate/risk control on Bilibili or YouTube**: Fix input or wait and rerun only failed items via `-ResumeBatch`.
- **Missing `yt-dlp`**: Required only for Bilibili/YouTube URLs; local videos and Douyin links continue working without it.
