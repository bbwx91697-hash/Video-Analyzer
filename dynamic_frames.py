#!/usr/bin/env python3
# 内容自适应抽帧：分析源视频，保留代表帧并输出可审计的选择原因。
"""Content-adaptive video frame extraction. Requires ffmpeg, ffprobe, OpenCV, NumPy and Pillow."""
import argparse
import bisect
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _resolve_tool(name, env_key):
    override = os.environ.get(env_key)
    if override:
        p = Path(override)
        if not p.is_file():
            raise RuntimeError(f'Configured {name} not found: {override}')
        return str(p)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f'Required tool not found in PATH: {name}')


# 输入命令参数列表 args；返回标准输出和标准错误，失败或超时时抛异常。
def command(args, timeout=300):
    try:
        p = subprocess.run(
            [str(x) for x in args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        tail = (e.stderr or '')[-1000:] if e.stderr else ''
        raise RuntimeError(f"Command timed out after {timeout}s: {args[0]}\n{tail}") from None
    if p.returncode:
        raise RuntimeError(f"Command failed (exit {p.returncode}): {args[0]}\n{p.stderr[-4000:]}")
    return p.stdout, p.stderr


# 输入历史 values、下限 floor、倍数 k；返回 median + k×MAD 与下限中的较大值，避免低噪声时阈值过低。
def adaptive_threshold(values, floor, k):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return floor
    median = float(np.median(values))
    return max(floor, median + k * float(np.median(np.abs(values - median))))


# 输入时间戳 pts、切点 cuts、切点后帧数 burst、短镜头秒数；返回强制保留帧及原因。cut_burst=3 包括切点和后两帧，另保留切点前一帧；不超过 1 秒的短镜头最多均匀保留 3 帧。
def boundary_frames(pts, cuts, burst, short_seconds):
    forced = {0: ['first_frame'], len(pts)-1: ['last_frame']}
    for i in cuts:
        forced.setdefault(i-1, []).append('scene_before')
        for j in range(i, min(i+burst, len(pts))):
            forced.setdefault(j, []).append('scene_after' if j == i else 'scene_burst')
    bounds = [0] + sorted(cuts) + [len(pts)]
    for start, end in zip(bounds, bounds[1:]):
        if end - start > 1 and pts[end-1] - pts[start] <= short_seconds:
            for j in np.linspace(start, end-1, min(3, end-start)).round().astype(int):
                forced.setdefault(int(j), []).append('short_shot')
    return forced


# 输入源宽高与最大宽高；返回等比例缩放尺寸。480×216 是最大包围盒，不拉伸画面。
def analysis_size(source_width, source_height, max_width, max_height):
    scale = min(max_width / source_width, max_height / source_height)
    return max(1, round(source_width * scale)), max(1, round(source_height * scale))


# 输入视频路径 path 和参数 a；返回时间戳、灰度图、采样索引、指标、强制帧、切点和场景分数。
# 默认以 10 fps 分析；源帧不足时只使用可获得的帧，不制造重复信息。
# ffmpeg 对每个源帧计算场景分数，0.30 用于判定切点，而非只看 10 fps 采样。
# 灰度差按 16×16 网格取局部最大值；滑动 2 秒噪声历史取中位数 + 3×MAD，局部阈值下限 8、全局静态阈值下限 0.6。
def analyze(path, a):
    ffprobe_bin = _resolve_tool('ffprobe', 'DOUYIN_MCP_FFPROBE')
    ffmpeg_bin = _resolve_tool('ffmpeg', 'DOUYIN_MCP_FFMPEG')
    cmd_timeout = getattr(a, 'cmd_timeout', 300)
    raw, _ = command([ffprobe_bin, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                      'frame=best_effort_timestamp_time', '-of', 'json', str(path)], timeout=cmd_timeout)
    pts = [float(f['best_effort_timestamp_time']) for f in json.loads(raw)['frames']
           if 'best_effort_timestamp_time' in f]
    if not pts or any(y < x for x, y in zip(pts, pts[1:])):
        raise ValueError('Video needs monotonic decoded-frame timestamps')
    start = pts[0]
    pts = [t - start for t in pts]
    max_samples = getattr(a, 'max_analysis_samples', 24000)
    if max_samples > 0 and pts:
        est_samples = int(max(1.0, (pts[-1] * a.analysis_fps) + 16))
        if est_samples > max_samples:
            raise RuntimeError(f'Estimated analysis samples ({est_samples}) exceed budget ({max_samples})')
    # ffmpeg's scene filter is evaluated on every source frame, not just analysis samples.
    _, log = command([ffmpeg_bin, '-hide_banner', '-nostats', '-copyts', '-i', str(path), '-an',
                      '-vf', "select='gte(scene,0)',metadata=print", '-f', 'null', '-'], timeout=cmd_timeout)
    scenes = []
    last_t = None
    for line in log.splitlines():
        m = re.search(r'frame:\s*\d+\s+pts:\s*[-\d]+\s+pts_time:([\d.eE+-]+)', line)
        if m:
            last_t = float(m.group(1))
        m = re.search(r'lavfi\.scene_score=([\d.eE+-]+)', line)
        if m and last_t is not None:
            scenes.append((max(0, min(len(pts)-1, bisect.bisect_left(pts, last_t - start - 1e-5))), float(m.group(1))))
    cuts = {i: score for i, score in scenes if score >= a.scene_threshold and i > 0}
    forced = boundary_frames(pts, cuts, a.cut_burst, a.short_shot_seconds)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    samples, images = [], {}
    next_t = 0.0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index >= len(pts):
            raise ValueError('Decoded frames exceed ffprobe timestamp count')
        sampled = pts[index] + 1e-7 >= next_t
        if sampled or index in forced:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = analysis_size(gray.shape[1], gray.shape[0], a.width, a.height)
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            images[index] = gray
            if sampled:
                samples.append(index)
                while next_t <= pts[index] + 1e-7:
                    next_t += 1.0 / a.analysis_fps
        index += 1
    cap.release()
    if index != len(pts):
        raise ValueError(f'Decoded frame count {index} does not match {len(pts)} timestamps')
    metrics = {}
    previous = None
    global_history, noise_history = [], []
    window = max(3, round(a.adaptive_window * a.analysis_fps))
    for i in samples:
        if previous is None:
            global_diff = local = noise = 0.0
        else:
            d = cv2.absdiff(images[i], images[previous])
            global_diff = float(d.mean())
            cells = [float(cell.mean()) for row in np.array_split(d, a.grid, axis=0)
                     for cell in np.array_split(row, a.grid, axis=1)]
            local, noise = max(cells), float(np.median(cells))
        # Use low-motion cells and the lower half of recent global differences as noise estimates.
        if previous is not None:
            noise_history.append(noise)
            global_history.append(global_diff)
        ns = noise_history[-window:]
        gs = global_history[-window:]
        lower = [v for v in gs if v <= np.median(gs)] if gs else []
        local_limit = adaptive_threshold(ns, a.local_threshold, a.adaptive_k)
        global_limit = adaptive_threshold(lower, a.static_threshold, a.adaptive_k)
        metrics[i] = (global_diff, local, global_limit, local_limit, noise)
        previous = i
    return pts, images, samples, metrics, forced, cuts, scenes


# 输入时间戳、灰度图、采样指标、强制帧、切点和参数；返回带时间、索引、原因和分数的选中帧列表。
# 局部运动密集模式为 5 fps（v3 从 8 降低以节省约 50%）；普通运动 2.5 fps，静态间隔 2.5 秒。
# 密集定时帧相对上次选中帧的全局平均灰度差须达到 0.2，或距上次选中至少 0.6 秒；后者仅作用于密集定时分支，并非全局最大间隔。
# 0.2 只是轻量去重，并非光流或真正的运动一致性；参数仅在三条素材调试。
# 累计差达到 12 且间隔至少 0.2 秒可触发；普通最小间隔 0.1 秒，强制/场景帧可更密。
def select(pts, images, samples, metrics, forced, cuts, a):
    selected = {}
    last_i = None
    dense_due = None
    for i in sorted(set(samples) | set(forced)):
        reasons = list(dict.fromkeys(forced.get(i, [])))
        global_diff, local, global_limit, local_limit, noise = metrics.get(i, (None, None, None, None, None))
        cumulative = float(cv2.absdiff(images[i], images[last_i]).mean()) if last_i is not None else 0.0
        if i in metrics and last_i is not None:
            elapsed = pts[i] - pts[last_i]
            if local > local_limit:
                mode, interval = 'local_motion', 1 / a.dense_fps
            elif global_diff < global_limit:
                mode, interval = 'static_interval', a.static_interval
            else:
                mode, interval = 'normal_motion', 1 / a.normal_fps
            if mode == 'local_motion':
                if dense_due is None:
                    dense_due = pts[last_i] + interval
                due = pts[i] + 1e-7 >= dense_due
            else:
                dense_due = None
                due = elapsed + 1e-7 >= interval
            if due and (mode != 'local_motion' or cumulative >= a.selected_diff_threshold
                        or elapsed >= a.motion_max_interval):
                reasons.append(mode)
            if cumulative >= a.cumulative_threshold and elapsed + 1e-7 >= a.cumulative_interval:
                reasons.append('cumulative_change')
            if elapsed + 1e-7 < a.min_interval and not forced.get(i):
                reasons = []
        if reasons:
            scores = {'global_diff': global_diff, 'local_max_diff': local,
                      'cumulative_diff': cumulative, 'scene_score': cuts.get(i, cuts.get(i+1)),
                      'global_threshold': global_limit, 'local_threshold': local_limit, 'noise_estimate': noise}
            selected[i] = {'t': round(pts[i], 6), 'frame_index': i, 'reasons': reasons,
                           'scores': {k: round(v, 5) if v is not None else None for k, v in scores.items()}}
            if 'local_motion' in reasons and not forced.get(i):
                dense_due += 1 / a.dense_fps
                dense_due = max(dense_due, pts[i])
            else:
                dense_due = None
            last_i = i
    return list(selected.values())


# 输入原图字典、选帧列表、输出路径和数量；生成时间分箱缩略拼图并返回实际选中的条目。
def contact_sheet(frames, entries, path, count):
    count = min(count, len(entries))
    # Select a high-scoring image per equal time bin, preferring scene boundaries.
    chosen = []
    end = entries[-1]['t']
    for n in range(count):
        candidates = [e for e in entries if (n * end / count <= e['t'] < (n+1)*end/count)
                      or (n == count-1 and e['t'] == end)]
        if not candidates:
            continue
        midpoint = (n + .5) * end / count
        e = max(candidates, key=lambda x: (2 * ('scene_after' in x['reasons']) +
                min((x['scores']['local_max_diff'] or 0) / 30, 1) -
                abs(x['t']-midpoint) / max(end/count, .001)))
        chosen.append(e)
    cols = 3
    rows = math.ceil(len(chosen) / cols)
    thumb_w, thumb_h, label_h = 400, 180, 26
    sheet = Image.new('RGB', (cols*thumb_w, rows*(thumb_h+label_h)), '#171717')
    draw = ImageDraw.Draw(sheet)
    for n, e in enumerate(chosen):
        img = Image.fromarray(cv2.cvtColor(frames[e['frame_index']], cv2.COLOR_BGR2RGB))
        img.thumbnail((thumb_w, thumb_h))
        x, y = (n % cols)*thumb_w, (n // cols)*(thumb_h+label_h)
        sheet.paste(img, (x+(thumb_w-img.width)//2, y+(thumb_h-img.height)//2))
        draw.text((x+8, y+thumb_h+5), f"{e['t']:.2f}s  frame {e['frame_index']}", fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return chosen


# 读取命令行参数并写出 JSON、可选帧图和拼图；无返回值。
# JSON 包含 video、时间与帧编号原点、源帧数、分析采样数、parameters、分布、scene_cuts、selected_count 和 frames；每帧含 t、frame_index、reasons、scores。
# --keep-frames 默认保留 JPEG；--cleanup 不保留 JPEG，JSON 和指定拼图仍会写出。
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--out-dir', type=Path, default=Path('frames'))
    p.add_argument('--json', type=Path, default=Path('frames.json'))
    p.add_argument('--contact', type=Path)
    p.add_argument('--contact-count', type=int, default=9, choices=range(6, 10))
    group = p.add_mutually_exclusive_group()
    group.add_argument('--keep-frames', dest='keep', action='store_true', help='Keep extracted JPEGs (default)')
    group.add_argument('--cleanup', dest='keep', action='store_false', help='Do not retain extracted frame images')
    p.set_defaults(keep=True)
    p.add_argument('--analysis-fps', type=float, default=10.0)
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=216)
    p.add_argument('--scene-threshold', type=float, default=.30)
    p.add_argument('--local-threshold', type=float, default=8.0)
    p.add_argument('--static-threshold', type=float, default=0.6)
    p.add_argument('--static-interval', type=float, default=2.5)
    p.add_argument('--dense-fps', type=float, default=5.0)
    p.add_argument('--selected-diff-threshold', type=float, default=0.2)
    p.add_argument('--motion-max-interval', type=float, default=.6)
    p.add_argument('--normal-fps', type=float, default=2.5)
    p.add_argument('--cumulative-threshold', type=float, default=12.0)
    p.add_argument('--cumulative-interval', type=float, default=.2)
    p.add_argument('--min-interval', type=float, default=.1)
    p.add_argument('--grid', type=int, default=16)
    p.add_argument('--adaptive-window', type=float, default=2.0)
    p.add_argument('--adaptive-k', type=float, default=3.0)
    p.add_argument('--cut-burst', type=int, default=3)
    p.add_argument('--short-shot-seconds', type=float, default=1.0)
    p.add_argument('--cmd-timeout', type=float, default=300.0)
    p.add_argument('--max-analysis-samples', type=int, default=24000)
    p.add_argument('--max-selected-frames', type=int, default=6000)
    a = p.parse_args()
    if min(a.analysis_fps, a.static_interval, a.dense_fps, a.normal_fps, a.motion_max_interval, a.cumulative_interval) <= 0 or a.grid < 1 or a.grid > min(a.width, a.height) or a.adaptive_window <= 0 or a.adaptive_k < 0 or a.cut_burst < 1 or a.short_shot_seconds < 0 or min(a.width, a.height) < 8 or min(a.scene_threshold, a.local_threshold, a.static_threshold, a.cumulative_threshold, a.min_interval, a.selected_diff_threshold) < 0 or a.cmd_timeout <= 0:
        p.error('Frequencies, intervals and timeout must be positive, dimensions >= 8, thresholds nonnegative')
    pts, images, samples, metrics, forced, cuts, scenes = analyze(a.video, a)
    entries = select(pts, images, samples, metrics, forced, cuts, a)
    if a.max_selected_frames > 0 and len(entries) > a.max_selected_frames:
        raise RuntimeError(f'Selected frame count ({len(entries)}) exceeds budget ({a.max_selected_frames})')
    distributions = {}
    for name, values in [('global_diff', [metrics[i][0] for i in samples[1:]]),
                         ('local_max_diff', [metrics[i][1] for i in samples[1:]]),
                         ('global_threshold', [metrics[i][2] for i in samples[1:]]),
                         ('local_threshold', [metrics[i][3] for i in samples[1:]]),
                         ('scene_score', [s for _, s in scenes])]:
        distributions[name] = dict(zip(['p50', 'p75', 'p90', 'p95', 'p99', 'max'],
                                   [round(float(x), 5) for x in np.percentile(values or [0], [50,75,90,95,99,100])]))
    result = {'video': str(a.video), 'time_origin': 'first decoded video frame',
              'frame_index_origin': 0, 'source_frame_count': len(pts), 'analysis_sample_count': len(samples),
              'parameters': {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items() if k != 'video'},
              'distributions': distributions, 'scene_cuts': [{'frame_index': i, 't': round(pts[i], 6), 'score': s} for i,s in cuts.items()],
              'selected_count': len(entries), 'frames': entries}
    if a.keep:
        a.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {e['frame_index'] for e in entries}
    full = {}
    cap = cv2.VideoCapture(str(a.video))
    index = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Could not decode all selected frames')
        if index in wanted:
            if a.keep:
                target = a.out_dir / f'frame_{index:08d}.jpg'
                if target.exists():
                    raise FileExistsError(f'Refusing to overwrite {target}')
                enc_ok, buf = cv2.imencode('.jpg', frame)
                if not enc_ok:
                    raise RuntimeError(f'Cannot encode {target}')
                target.write_bytes(buf.tobytes())
                if not target.is_file() or target.stat().st_size <= 0:
                    raise RuntimeError(f'Cannot write {target}')
            if a.contact:
                full[index] = frame
            wanted.remove(index)
        index += 1
    cap.release()
    if a.contact:
        result['contact_frames'] = [e['frame_index'] for e in contact_sheet(full, entries, a.contact, a.contact_count)]
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'samples': len(samples), 'selected': len(entries), 'cuts': result['scene_cuts'],
                      'distributions': distributions}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
