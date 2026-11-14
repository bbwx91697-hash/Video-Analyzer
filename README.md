# Video Analyzer

面向 AI Agent 与本地工作流的视频深度解析流水线（Windows / PowerShell + Python）：将本地视频或支持的在线链接自动转为 **分层视觉拼图 + 本地 GPU 语音转写 + 字幕 OCR + 客观音频能量统计**，并支持串行批处理与零覆盖断点续跑。

设计目标是用本地算力预先消化音频、字幕与冗余帧，大幅减少直接把整段视频塞给云端多模态大模型时的 Token 消耗与遗漏。

## 核心能力

- **动态自适应抽帧与分层速读**：结合场景切点（`scene`）与运动峰值抽取非均匀代表帧，生成三层视觉产物：
  1. `storyboard.jpg`（精选核心分镜总览）
  2. `transitions.jpg`（硬切转场与高动态动作 4 帧连拍）
  3. `timeline_00X.jpg`（100% 完整保留全部动态代表帧的时间轴拼图）
- **本地离线语音转写 & OCR**：集成 `faster-whisper`（支持 CUDA `large-v3` 与 CPU 回退）及 `RapidOCR`，按分镜配额覆盖识别并审计覆盖率、去偏置统计高频常驻文字。
- **客观音频能量分析**：计算全片 RMS/峰值 dBFS、窗口静音占比、低能量区间与能量突增时间点，以及音视频起点偏移 `av_start_offset`。
- **可选低码率代理与增量截取 Clips**：`-SaveProxy` 保存轻量级 `proxy/proxy.mp4`（保持 `1:1` 源时间轴），事后可通过 `-AppendClips` 在不重跑全片的前提下按时间段导出 `10 fps` 局部细节条。
- **严格主机名白名单外链路由（`schema 1.8`）**：
  - 抖音系列：`douyin.com`、`iesdouyin.com`、`amemv.com`、`v.douyin.com`（需本机另配兼容浏览器抓取器，内置 `TARGET_STREAM_UNBOUND` 防串台拦截）。
  - 外站直连：`bilibili.com`、`b23.tv`、`bili2233.cn`、`youtube.com`、`youtu.be`、`youtube-nocookie.com`（依赖虚拟环境内可选安装的 `yt-dlp`；缺失 `yt-dlp` 时不影响本地视频与抖音流程）。
  - 拒绝未知域名与下载器之间静默回退。
- **串行批处理与零覆盖续跑**：`-BatchFile` 严格串行调度 `.txt` / `.json` 清单，隔离单条故障并输出可追溯的 `batch_summary.md` / `batch_index.json`；`-ResumeBatch` 写入全新批次目录，复用已有成功任务且绝不覆盖旧成果。
- **内置跨 Agent 通用 Skill**：自带 [skills/video-analyzer/SKILL.md](skills/video-analyzer/SKILL.md)（Agent Skills 标准格式），方便直接挂载到兼容 Agent 中按分层速读协议省额度运行。

## 目录结构

```text
.
├── analyze.ps1                  # Windows PowerShell 主入口（自动配置隔离缓存与编码）
├── analyze.py                   # 单条分析与报告生成核心流水线（schema 1.8）
├── batch_analyze.py             # 串行批处理与断点续跑编排器
├── media_router.py              # URL 域名白名单路由与可选 yt-dlp 统一元数据适配
├── dynamic_frames.py            # 自适应动态抽帧与代表帧筛选
├── douyin_fetch.py              # 抖音抓取桥接与元数据脱敏提取（带目标流绑定校验）
├── prefetch_model.py            # Whisper 模型预下载与 CUDA 设备检查
├── setup.ps1                    # 一键创建 .venv、安装依赖并拉取模型
├── requirements.txt             # 核心依赖（不含强制 yt-dlp）
├── skills/
│   └── video-analyzer/SKILL.md  # 通用 Agent Skill 说明文件
├── tools/
│   └── fetch_model_parallel.ps1 # large-v3 16 路分段下载与 SHA-256 校验脚本
└── tests/
    ├── test_analyzer.py         # 流水线、外链路由、批处理与边界回归测试（22 项）
    └── test_dynamic_frames.py   # 动态选帧单元测试（2 项）
```

## 环境要求与安装

- **系统与基础软件**：Windows 10/11、Python 3.10+、`ffmpeg` 与 `ffprobe`（已加入 `PATH`，或置于 `D:\ffmpeg\bin`）。
- **GPU（可选但推荐）**：NVIDIA 显卡及 CUDA 驱动（默认优先 CUDA 跑 `faster-whisper`，不可用时可在命令行传 `-Device cpu`）。

### 1. 基础环境搭建

```powershell
# 创建项目局部 .venv、安装 core 依赖并下载默认 large-v3 模型
.\setup.ps1

# 若国内下载 Hugging Face 较慢，可启用镜像：
.\setup.ps1 -HfMirror
```

如单连接下载 `faster-whisper-large-v3` 较慢，可使用 [tools/fetch_model_parallel.ps1](tools/fetch_model_parallel.ps1) 进行 16 路并发断点拉取并校验 SHA-256。

### 2. 可选外链下载支持（B站 / YouTube）

本仓库默认不将在线视频下载器写入强制依赖，如需解析 B站 / YouTube 链接，请在虚拟环境内可选安装 `yt-dlp`：

```powershell
.\.venv\Scripts\python.exe -m pip install yt-dlp
```

### 3. 可选抖音抓取配置

`douyin_fetch.py` 默认调用本机独立的 `douyin-browser-mcp` 虚拟环境执行浏览器页面抓包，可通过环境变量 `DOUYIN_FETCH_PYTHON` 或参数 `-FetchPython` 指定兼容抓包器 Python 路径；若仅分析本地视频或 B站/YouTube 链接，则无需配置该项。

## 快速使用

### 单条视频分析

```powershell
# 分析本地视频（支持中文与空格路径）并保留低码率代理
.\analyze.ps1 -Video "path\to\video.mp4" -SaveProxy

# 分析 B站 / YouTube 白名单链接，并导出指定时间段 10 fps 动作条
.\analyze.ps1 "https://www.bilibili.com/video/BV1jMWY6JEx7" -Clips "00:00-00:04"

# 分析抖音链接
.\analyze.ps1 "https://v.douyin.com/example/" -Device auto -SaveProxy
```

### 基于已有结果追加 Clips

无需重跑下载、抽帧、ASR 与 OCR，自动复用 `proxy/proxy.mp4` 或通过 `-Video` 校验原片 SHA-256：

```powershell
.\analyze.ps1 -AppendClips ".\results\<result_dir>" -Clips "00:15-00:22"
```

### 串行批量分析与断点续跑

准备 `.txt` 清单（支持 `#` 注释及每行用 `|` 附带 `clips`、`label`、`save_proxy`、`keep_media`、`max_ocr_frames` 覆盖参数）或 `.json` 清单：

```text
# tasks.txt 示例
https://www.bilibili.com/video/BV1jMWY6JEx7 | clips=00:00-00:04
videos\sample.mp4 | label=sample | save_proxy=true
```

运行与续跑：

```powershell
# 串行执行批量任务（单条失败默认隔离，不阻断后续任务）
.\analyze.ps1 -BatchFile ".\tasks.txt"

# 遇错即停（剩余任务标记为 not_executed）
.\analyze.ps1 -BatchFile ".\tasks.txt" -StopOnError

# 断点续跑（生成全新 batch-<run_id> 目录，绝不覆盖已有成功成果）
.\analyze.ps1 -ResumeBatch ".\results\batch-<old_id>"
```

## 常用参数速查

| 参数 | 说明 |
| --- | --- |
| `<Url>` / `-Video` | 输入在线白名单链接或本地视频路径（互斥） |
| `-AppendClips` | 基于已有结果目录增量导出 `-Clips` |
| `-BatchFile` / `-ResumeBatch` | 串行批处理清单 / 基于旧批次索引断点续跑 |
| `-RetryDegraded` | 配合 `-ResumeBatch` 时重跑此前降级任务 |
| `-StopOnError` | 批处理遇到首条失败即终止 |
| `-BatchTimeout` | 批处理单条子任务超时秒数（默认 `1800`） |
| `-SaveProxy` / `-KeepMedia` | 保留低码率代理 `proxy/proxy.mp4` / 保留下载原片 |
| `-Clips` | 导出指定时间区间 `10 fps` 局部细查图（如 `"00:10-00:18"`） |
| `-MaxOcrFrames` | 限制扩展 OCR 最大扫描帧数（超限如实标注覆盖不足） |
| `-Model` / `-Device` | Whisper 模型名（默认 `large-v3`）与设备（`auto` / `cuda` / `cpu`） |
| `-YtdlpBin` | 手动指定 `yt-dlp` 可执行路径 |

## 输出产物说明

每次单条运行都会在 `results/` 下生成独立目录：
- `analysis.md`：分层速读报告（风格画像、元数据及来源/缺失原因、音视频指标、OCR 审计、转写全文、智能分镜时间轴）
- `analysis.json`：结构化全量数据（`schema_version: 1.8`）
- `storyboard.jpg`、`transitions.jpg`、`timeline_00X.jpg`：三层渐进式视觉拼图
- `clips/`（指定 `-Clips` 时）与 `proxy/proxy.mp4`（指定 `-SaveProxy` 时）

## 作为 Agent Skill 使用

仓库内附带了符合通用 Agent Skills 规范的 [skills/video-analyzer/SKILL.md](skills/video-analyzer/SKILL.md)，可复制到各支持该规范的 Agent 全局或项目级 `skills/video-analyzer/` 目录中使用。核心策略是让 Agent 默认只读取 `analysis.md` + `storyboard.jpg`（按需补看 `transitions.jpg`），避免全片切图浪费上下文额度。

## 运行测试

```powershell
# 确保 ffmpeg/ffprobe 在 PATH 中
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 已知边界与局限

- **外链可用性依赖外部平台风控**：未登录或匿名抓取抖音/B站时，可能受到目标站风控、跳转推荐流（由 `TARGET_STREAM_UNBOUND` 主动拦截防止串台）或 `HTTP Error 567` 等限流影响；建议为高稳定性场景配置独立持久化登录环境或配合 `-ResumeBatch` 重试。
- **无内置视觉大模型推理**：当前流水线保持 `vlm_experimental=false`，只负责生成确定性视听证据与分层拼图，视觉语义理解由调用方 Agent 或人工完成。

## License

[MIT](LICENSE)
