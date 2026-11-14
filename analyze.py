# 本地流水线：下载或读取视频、抽帧、转写、逐段光影色彩追踪、核心分镜通用 OCR、对齐、输出与清理。
"""Local orchestration for the Douyin analysis pipeline (AI-token-efficient universal edition)."""
import argparse
import bisect
from collections import Counter
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from media_router import (
    UnsupportedRouteError,
    classify_url_route,
    fetch_external_with_ytdlp,
    normalize_media_result,
    resolve_ytdlp_runner,
)

ROOT = Path(__file__).resolve().parent
URL_RE = re.compile(r'https?://[^\s<>"\']+', re.I)


# 输入任意文本；若安装了 zhconv 则将繁体字转换为简体中文。
def to_simplified(text):
    if not isinstance(text, str) or not text:
        return text
    try:
        import zhconv
        return zhconv.convert(text, 'zh-cn')
    except Exception:
        return text


# 输入任意值；返回去除 URL 查询和常见凭据字段的文本。
def safe_text(value):
    def clean(match):
        try:
            p = urlsplit(match.group())
            return urlunsplit((p.scheme, p.netloc, p.path, '', ''))
        except ValueError:
            return '[URL]'
    value = URL_RE.sub(clean, str(value))
    return re.sub(r'(?i)\b(cookie|token|authorization|signature|x-bogus|msToken)\s*[:=]\s*[^\s,;]+', r'\1=[redacted]', value)


# 输入嵌套数据；递归过滤敏感键并清理字符串，返回清理后的数据。
def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if not any(x in str(k).lower() for x in ('cookie', 'token', 'sign', 'header'))}
    if isinstance(value, list):
        return [scrub(x) for x in value]
    return safe_text(value) if isinstance(value, str) else value


# 输入带时间的帧和语音片段；返回按时间排序的语音与无语音行，每帧只分配一次。
def align_timeline(frames, segments):
    """Assign each frame to the first matching segment, or a no-speech run."""
    frames = sorted(frames, key=lambda f: (f['t'], f['index']))
    segments = sorted(segments, key=lambda s: (s['start'], s['end'], s['id']))
    assigned = [[] for _ in segments]
    gaps = []
    gap = []
    gap_region = None
    for f in frames:
        match = next((i for i, s in enumerate(segments) if s['start'] <= f['t'] < s['end']), None)
        if match is None:
            region = sum(s['end'] <= f['t'] for s in segments)
            if gap and region != gap_region:
                gaps.append(gap); gap = []
            gap_region = region
            gap.append(f)
        else:
            if gap:
                gaps.append(gap); gap = []
            assigned[match].append(f)
    if gap:
        gaps.append(gap)
    rows = [{'type': 'speech', 'start': s['start'], 'end': s['end'], 'segment_id': s['id'], 'text': s.get('text', ''), 'frames': assigned[i]} for i, s in enumerate(segments)]
    rows += [{'type': 'no-speech', 'start': g[0]['t'], 'end': g[-1]['t'], 'frames': g} for g in gaps]
    return sorted(rows, key=lambda r: (r['start'], 0 if r['type'] == 'speech' else 1))


# 输入秒数；返回分:秒.百分秒文本，无效输入返回占位符。
def stamp(seconds):
    try:
        n = max(0, round(float(seconds) * 100))
        return f'{n // 6000:02d}:{(n // 100) % 60:02d}.{n % 100:02d}'
    except (TypeError, ValueError, OverflowError):
        return '--:--.--'


# 加载支持中文的字体；失败时回退默认字体。
def load_font(size=14):
    from PIL import ImageFont
    for candidate in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyhbd.ttc', r'C:\Windows\Fonts\simhei.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


# 输入亮度、对比度、饱和度、冷暖差均值；返回简短可读的画面光影色调标签。
def format_lighting_tag(luma, contrast, sat, warm):
    b_tag = '昏暗' if luma < 65 else ('偏暗' if luma < 95 else ('明亮' if luma > 160 else '明暗适中'))
    c_tag = '高反差' if contrast > 60 else ('低反差' if contrast < 35 else '')
    s_tag = '浓烈高饱和' if sat > 48 else ('低饱和素雅' if sat < 20 else '')
    w_tag = '暖调' if warm > 14 else ('冷调' if warm < -12 else '冷暖均衡')
    parts = [x for x in (b_tag, c_tag, s_tag, w_tag) if x]
    return f"{'·'.join(parts)}(亮{round(luma)})"


# 对所有代表帧提取轻量级光影色彩指标（写入每帧 visual 字段），并汇总全片画幅、明暗变化幅度与色彩基调。
def analyze_visual_features(items, frame_dir, media):
    if not items or not frame_dir or not Path(frame_dir).is_dir():
        return {}
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {}
    lumas, contrasts, sats, warms, top_bars = [], [], [], [], []
    for f in items:
        p = Path(frame_dir) / f"frame_{f.get('frame_index', 0):08d}.jpg"
        if not p.is_file():
            raise RuntimeError(f"[visual_features] Missing frame image: {p.name} (frame #{f.get('index')})")
        try:
            with Image.open(p) as im:
                im.load()
                small = im.convert('RGB').resize((96, 54))
                rgb = np.asarray(small, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"[visual_features] Corrupt frame image {p.name} (frame #{f.get('index')}): {safe_text(e)}") from None
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        luma = round(float(y.mean()), 1)
        contrast = round(float(y.std()), 1)
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        sat = round(float(np.where(cmax > 1.0, (cmax - cmin) / np.maximum(cmax, 1.0), 0.0).mean() * 100.0), 1)
        warm = round(float((r - b).mean()), 1)
        f['visual'] = {'luma': luma, 'contrast': contrast, 'sat': sat, 'warm': warm}
        lumas.append(luma)
        contrasts.append(contrast)
        sats.append(sat)
        warms.append(warm)
        h = y.shape[0]
        edge_luma = 0.5 * (float(y[:6, :].mean()) + float(y[h - 6:, :].mean()))
        center_luma = float(y[14:h - 14, :].mean())
        top_bars.append(edge_luma < 14.0 and center_luma > 35.0)
    if not lumas:
        return {}
    mean_luma = round(sum(lumas) / len(lumas), 1)
    min_luma, max_luma = round(min(lumas), 1), round(max(lumas), 1)
    luma_range = round(max_luma - min_luma, 1)
    mean_contrast = round(sum(contrasts) / len(contrasts), 1)
    mean_sat = round(sum(sats) / len(sats), 1)
    mean_warm = round(sum(warms) / len(warms), 1)
    brightness_tag = '偏暗/深色背景' if mean_luma < 85 else ('明亮/高调' if mean_luma > 165 else '明暗适中')
    if luma_range >= 55.0:
        brightness_tag += f'（全片存在剧烈明暗光影转折：最低 {min_luma} → 最高 {max_luma}）'
    contrast_tag = '高反差/强视觉冲击' if mean_contrast > 62 else ('柔和低反差' if mean_contrast < 38 else '中等对比度')
    sat_tag = '高饱和鲜艳/浓烈色彩' if mean_sat > 42 else ('低饱和素雅/写实' if mean_sat < 22 else '色彩饱和适中')
    warm_tag = '暖色调偏重' if mean_warm > 10 else ('冷色调偏重' if mean_warm < -10 else '冷暖均衡')
    w, h = media.get('width') or 0, media.get('height') or 0
    if w and h:
        ratio = round(w / h, 2)
        orientation = '横屏 (16:9 宽画幅)' if ratio >= 1.3 else ('竖屏 (9:16 移动画幅)' if ratio <= 0.8 else '方画幅/接近 1:1')
    else:
        orientation = '未知画幅'
    letterboxed = sum(top_bars) > len(top_bars) * 0.6 if top_bars else False
    if letterboxed:
        orientation += '（含上下黑边遮幅）'
    return {
        'orientation': orientation,
        'mean_luma_0_255': mean_luma,
        'luma_min_max': [min_luma, max_luma],
        'mean_contrast': mean_contrast,
        'mean_saturation_pct': mean_sat,
        'warmth_r_minus_b': mean_warm,
        'visual_summary': f'{orientation}；{brightness_tag}，{contrast_tag}（对比 {mean_contrast}），{sat_tag}（饱和度 {mean_sat}%），{warm_tag}'
    }


# 检测无硬切长镜头内部的光影/色调显著转折点（例如进出隧道、明暗骤变、冷暖转场），返回转折帧 index 集合。
def detect_lighting_transitions(items, min_gap_seconds=1.5):
    if len(items) < 6:
        return set()
    transitions = set()
    last_t = -999.0
    win = 3
    for idx in range(win, len(items) - win):
        f = items[idx]
        if f['t'] - last_t < min_gap_seconds:
            continue
        prev_v = [x.get('visual') for x in items[idx - win:idx] if x.get('visual')]
        next_v = [x.get('visual') for x in items[idx:idx + win] if x.get('visual')]
        if len(prev_v) < 2 or len(next_v) < 2:
            continue
        luma_before = sum(v['luma'] for v in prev_v) / len(prev_v)
        luma_after = sum(v['luma'] for v in next_v) / len(next_v)
        warm_before = sum(v['warm'] for v in prev_v) / len(prev_v)
        warm_after = sum(v['warm'] for v in next_v) / len(next_v)
        if abs(luma_after - luma_before) >= 38.0 or abs(warm_after - warm_before) >= 28.0:
            transitions.add(f['index'])
            last_t = f['t']
    return transitions


# 从全片代表帧中精选最多 max_frames 张高信息量核心分镜帧（涵盖首尾、台词、硬切点、光影转折点与全时间轴均匀区间）。
def select_storyboard_frames(items, scene_cuts, segments, duration, lighting_shifts=None, max_frames=24):
    if not items:
        return []
    if len(items) <= max_frames:
        return [dict(f, highlight_reason='代表帧') for f in items]
    lighting_shifts = lighting_shifts or set()
    chosen = {}

    def pick(frame, label):
        idx = frame['index']
        if idx not in chosen:
            chosen[idx] = dict(frame, highlight_reason=label)
        elif label not in chosen[idx]['highlight_reason']:
            chosen[idx]['highlight_reason'] += f'/{label}'

    first_frame = items[0]
    if (first_frame.get('visual') or {}).get('luma', 50.0) < 18.0:
        for cand in items[1:12]:
            if cand['t'] <= 2.5 and (cand.get('visual') or {}).get('luma', 0.0) >= 18.0:
                first_frame = cand
                break
    pick(first_frame, '开场首帧')
    pick(items[-1], '结尾末帧')
    for s in (segments or []):
        mid = 0.5 * (float(s.get('start', 0)) + float(s.get('end', 0)))
        nearest = min(items, key=lambda f: abs(f['t'] - mid))
        txt = (s.get('text') or '').strip()
        short_txt = (txt[:10] + '…') if len(txt) > 11 else txt
        pick(nearest, f'台词:{short_txt}' if short_txt else '人声段')
    for f in items:
        reasons = f.get('reasons') or []
        if 'scene_after' in reasons and all(abs(f['t'] - c['t']) >= 0.8 for c in chosen.values()):
            pick(f, '镜头切换')
        elif f['index'] in lighting_shifts and all(abs(f['t'] - c['t']) >= 0.8 for c in chosen.values()):
            pick(f, '光影转折')
    bins = max(12, min(max_frames, 24))
    span = max(float(duration or items[-1]['t'] or 1.0), 1.0)
    for b in range(bins):
        t0, t1 = b * span / bins, (b + 1) * span / bins
        if any(t0 <= c['t'] <= t1 for c in chosen.values()):
            continue
        cands = [f for f in items if t0 <= f['t'] <= t1]
        if not cands:
            continue
        best = max(cands, key=lambda f: (
            2.0 if 'cumulative_change' in (f.get('reasons') or []) else 0.0,
            float((f.get('scores') or {}).get('local_max_diff') or 0.0),
            float((f.get('scores') or {}).get('cumulative_diff') or 0.0)
        ))
        pick(best, '画面峰值')
    result = sorted(chosen.values(), key=lambda f: (f['t'], f['index']))
    if len(result) > max_frames:
        # 按全片时间轴均匀分桶优选，避免前段快切占满名额导致后半段长镜头断档
        slot_picked = {result[0]['index']: result[0], result[-1]['index']: result[-1]}
        inner_slots = max(1, max_frames - 2)
        for s_idx in range(inner_slots):
            t0, t1 = s_idx * span / inner_slots, (s_idx + 1) * span / inner_slots
            slot_cands = [f for f in result if t0 <= f['t'] <= t1 and f['index'] not in slot_picked]
            if not slot_cands:
                mid_t = 0.5 * (t0 + t1)
                slot_cands = [min((f for f in result if f['index'] not in slot_picked), key=lambda f: abs(f['t'] - mid_t), default=None)]
            slot_cands = [f for f in slot_cands if f is not None]
            if slot_cands:
                best_slot = max(slot_cands, key=lambda f: (
                    3 if '台词' in f.get('highlight_reason', '') else (2 if '镜头切换' in f.get('highlight_reason', '') else (1 if '光影转折' in f.get('highlight_reason', '') else 0)),
                    float((f.get('scores') or {}).get('local_max_diff') or 0.0)
                ))
                slot_picked[best_slot['index']] = best_slot
        result = sorted(slot_picked.values(), key=lambda f: (f['t'], f['index']))
    return result


# 将视听对齐时间轴划分为初步分镜段（与智能分镜切分口径一致），供扩展 OCR 与分镜汇总复用。
def partition_preliminary_shots(timeline, lighting_shifts=None, max_silent_span=15.0):
    lighting_shifts = lighting_shifts or set()
    preliminary = []
    for row in (timeline or []):
        frames = row.get('frames') or []
        row_type = row.get('type', 'no-speech')
        if row_type == 'speech' or (len(frames) <= 10 and not any(f['index'] in lighting_shifts for f in frames[1:])):
            preliminary.append({
                'shot_id': len(preliminary) + 1,
                'type': row_type,
                'start': row.get('start', 0.0),
                'end': row.get('end', 0.0),
                'text': row.get('text', ''),
                'frames': list(frames),
            })
            continue
        chunk = []
        for f in frames:
            reasons = f.get('reasons') or []
            is_cut = 'scene_after' in reasons and chunk and (f['t'] - chunk[0]['t'] >= 1.2)
            is_light_shift = f['index'] in lighting_shifts and chunk and (f['t'] - chunk[0]['t'] >= 1.2)
            is_long = chunk and (f['t'] - chunk[0]['t'] >= max_silent_span)
            if is_cut or is_light_shift or is_long:
                preliminary.append({
                    'shot_id': len(preliminary) + 1,
                    'type': 'no-speech',
                    'start': chunk[0]['t'],
                    'end': chunk[-1]['t'],
                    'text': '',
                    'frames': list(chunk),
                })
                chunk = [f]
            else:
                chunk.append(f)
        if chunk:
            preliminary.append({
                'shot_id': len(preliminary) + 1,
                'type': 'no-speech',
                'start': chunk[0]['t'],
                'end': chunk[-1]['t'],
                'text': '',
                'frames': list(chunk),
            })
    return preliminary


# 基于精选分镜与初步分镜列表规划 OCR 目标帧：
# 1) 与 storyboard 重合的帧（同 frame_index）不重复识别；
# 2) 每个有可用帧的初步分镜尽量覆盖 1-2 个不同源帧；不足 2 帧如实记录；
# 3) 受 max_ocr_frames 预算约束，超预算时明确标记覆盖不足。
def select_ocr_frames(storyboard_items, preliminary_shots, max_ocr_frames=96):
    budget = int(max_ocr_frames) if max_ocr_frames is not None else 0
    shot_map_by_fi = {}
    shots_with_frames = []
    for sh in (preliminary_shots or []):
        sid = sh.get('shot_id')
        unique_frames = []
        seen_fi = set()
        for f in (sh.get('frames') or []):
            fi = f.get('frame_index')
            if fi is None:
                continue
            shot_map_by_fi.setdefault(fi, [])
            if sid not in shot_map_by_fi[fi]:
                shot_map_by_fi[fi].append(sid)
            if fi not in seen_fi:
                seen_fi.add(fi)
                unique_frames.append(f)
        if unique_frames:
            shots_with_frames.append((sh, unique_frames))

    selected_by_fi = {}
    deduplicated_overlap_count = 0
    budget_exceeded = False

    def add_target(f, role):
        nonlocal deduplicated_overlap_count
        fi = f.get('frame_index')
        if fi is None:
            return False
        idx = f.get('index')
        if fi in selected_by_fi:
            entry = selected_by_fi[fi]
            if idx is not None and idx not in entry['matched_indices']:
                entry['matched_indices'].append(idx)
            if role == 'storyboard' and 'storyboard' not in entry['source_role']:
                entry['source_role'] = 'storyboard+shot' if entry['source_role'] == 'shot_supplement' else 'storyboard'
            elif role == 'shot_supplement' and entry['source_role'] == 'storyboard':
                entry['source_role'] = 'storyboard+shot'
                deduplicated_overlap_count += 1
            return False
        selected_by_fi[fi] = {
            'index': idx,
            'frame_index': fi,
            't': float(f.get('t') or 0.0),
            'source_role': 'storyboard+shot' if (role == 'storyboard' and shot_map_by_fi.get(fi)) else role,
            'shot_ids': list(shot_map_by_fi.get(fi, [])),
            'matched_indices': [idx] if idx is not None else [],
        }
        return True

    # 去重收集 storyboard 帧（若本身超出预算，按均匀间隔受限截取并标记预算超限）
    unique_sb = []
    seen_sb_fi = set()
    for f in (storyboard_items or []):
        fi = f.get('frame_index')
        if fi is not None and fi not in seen_sb_fi:
            seen_sb_fi.add(fi)
            unique_sb.append(f)
        elif fi in seen_sb_fi:
            deduplicated_overlap_count += 1

    if budget > 0 and len(unique_sb) > budget:
        budget_exceeded = True
        if budget == 1:
            picked_sb = [unique_sb[0]]
        else:
            picked_sb = [unique_sb[round(k * (len(unique_sb) - 1) / (budget - 1))] for k in range(budget)]
            # 去重保序
            seen_p = set()
            picked_sb = [x for x in picked_sb if not (x['frame_index'] in seen_p or seen_p.add(x['frame_index']))]
    else:
        picked_sb = unique_sb

    for f in picked_sb:
        add_target(f, 'storyboard')

    def count_shot_covered(unique_frames):
        return [f for f in unique_frames if f.get('frame_index') in selected_by_fi]

    # 若 storyboard 已和某个初步分镜重合，计为重合复用（不重复识别）
    for _, unique_frames in shots_with_frames:
        for f in unique_frames:
            if f.get('frame_index') in selected_by_fi and selected_by_fi[f['frame_index']]['source_role'] == 'storyboard+shot':
                deduplicated_overlap_count += 1

    # 第 1 轮：确保每个有可用帧的初步分镜至少覆盖 1 个不同源帧
    for sh, unique_frames in shots_with_frames:
        if count_shot_covered(unique_frames):
            continue
        if budget > 0 and len(selected_by_fi) >= budget:
            budget_exceeded = True
            break
        mid_t = 0.5 * (float(unique_frames[0].get('t') or 0.0) + float(unique_frames[-1].get('t') or 0.0))
        best = max(unique_frames, key=lambda f: (
            float((f.get('scores') or {}).get('local_max_diff') or 0.0),
            -abs(float(f.get('t') or 0.0) - mid_t),
        ))
        add_target(best, 'shot_supplement')

    # 第 2 轮：对可用不同源帧 >= 2 但当前仅覆盖 1 帧的初步分镜，再补第 2 个不同源帧
    for sh, unique_frames in shots_with_frames:
        if len(unique_frames) < 2:
            continue
        covered = count_shot_covered(unique_frames)
        if len(covered) >= 2:
            continue
        if budget > 0 and len(selected_by_fi) >= budget:
            budget_exceeded = True
            break
        anchor_ts = [float(f.get('t') or 0.0) for f in covered] or [float(unique_frames[0].get('t') or 0.0)]
        cands = [f for f in unique_frames if f.get('frame_index') not in selected_by_fi]
        if not cands:
            continue
        best = max(cands, key=lambda f: (
            min(abs(float(f.get('t') or 0.0) - at) for at in anchor_ts),
            float((f.get('scores') or {}).get('local_max_diff') or 0.0),
        ))
        add_target(best, 'shot_supplement')

    # 审计每个有可用帧的初步分镜覆盖状态
    covered_shot_count = 0
    fully_covered_shot_count = 0
    single_frame_shot_count = 0
    insufficient_shots = []
    for sh, unique_frames in shots_with_frames:
        covered_cnt = len(count_shot_covered(unique_frames))
        avail_cnt = len(unique_frames)
        target_cnt = min(2, avail_cnt)
        if covered_cnt >= 1:
            covered_shot_count += 1
        if covered_cnt >= target_cnt:
            fully_covered_shot_count += 1
        if avail_cnt < 2:
            single_frame_shot_count += 1
            insufficient_shots.append({
                'shot_id': sh.get('shot_id'),
                'start': round(float(sh.get('start') or 0.0), 2),
                'end': round(float(sh.get('end') or 0.0), 2),
                'available_unique_source_frames': avail_cnt,
                'covered_unique_source_frames': covered_cnt,
                'reason': 'fewer_than_2_available_frames' if covered_cnt >= avail_cnt else 'budget_exceeded',
            })
        elif covered_cnt < 2:
            budget_exceeded = True
            insufficient_shots.append({
                'shot_id': sh.get('shot_id'),
                'start': round(float(sh.get('start') or 0.0), 2),
                'end': round(float(sh.get('end') or 0.0), 2),
                'available_unique_source_frames': avail_cnt,
                'covered_unique_source_frames': covered_cnt,
                'reason': 'budget_exceeded',
            })

    targets = sorted(selected_by_fi.values(), key=lambda x: (x['t'], x['frame_index']))
    sb_cnt = sum(1 for x in targets if 'storyboard' in x['source_role'])
    supp_cnt = sum(1 for x in targets if x['source_role'] == 'shot_supplement')
    audit = {
        'max_ocr_frames_budget': budget,
        'budget_exceeded': budget_exceeded,
        'coverage_sufficient': not budget_exceeded,
        'preliminary_shot_count': len(preliminary_shots or []),
        'preliminary_shots_with_frames_count': len(shots_with_frames),
        'covered_shot_count': covered_shot_count,
        'fully_covered_shot_count': fully_covered_shot_count,
        'single_frame_shot_count': single_frame_shot_count,
        'has_single_frame_shots': single_frame_shot_count > 0,
        'storyboard_frames_count': sb_cnt,
        'supplemented_frames_count': supp_cnt,
        'deduplicated_overlap_count': deduplicated_overlap_count,
        'insufficient_shots': insufficient_shots,
    }
    return targets, audit


# 通用多场景核心/分镜扩展 OCR：零硬编码词表，使用「视觉显著度 × 时间分桶去偏逆帧频率 IDF」
# 将贯穿全片多时段且跨多时间桶的文字归入「常驻角标/固定UI」（解决短片 0.25 阈值及不均匀采样误判），
# 保存逐帧输入帧号、PTS、原始识别结果、处理状态及失败数。
def extract_keyframe_ocr(storyboard_items, frame_dir, media=None, coverage_audit=None):
    # 按 frame_index 去重，保证即使传入未去重列表也不重复推理同一源帧
    dedup_items = []
    seen_fi = set()
    fi_to_indices = {}
    for f in (storyboard_items or []):
        fi = f.get('frame_index', 0)
        idx = f.get('index')
        fi_to_indices.setdefault(fi, [])
        if idx is not None and idx not in fi_to_indices[fi]:
            fi_to_indices[fi].append(idx)
        for m_idx in (f.get('matched_indices') or []):
            if m_idx not in fi_to_indices[fi]:
                fi_to_indices[fi].append(m_idx)
        if fi not in seen_fi:
            seen_fi.add(fi)
            dedup_items.append(f)

    total_count = len(dedup_items)
    status = {
        'status': 'skipped',
        'total_frames': total_count,
        'processed_frames': 0,
        'failed_frames': 0,
        'error': None,
        'frames': [],
    }
    if coverage_audit:
        status.update({
            k: coverage_audit[k] for k in (
                'max_ocr_frames_budget', 'budget_exceeded', 'coverage_sufficient',
                'preliminary_shot_count', 'preliminary_shots_with_frames_count',
                'covered_shot_count', 'fully_covered_shot_count', 'single_frame_shot_count',
                'has_single_frame_shots', 'storyboard_frames_count', 'supplemented_frames_count',
                'deduplicated_overlap_count', 'insufficient_shots',
            ) if k in coverage_audit
        })

    if not dedup_items or not frame_dir or not Path(frame_dir).is_dir():
        return {}, [], status
    try:
        import cv2
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR(use_angle_cls=False)
    except Exception as e:
        status['status'] = 'unavailable'
        status['failed_frames'] = total_count
        status['error'] = safe_text(e)[:300]
        return {}, [], status

    noise_re = re.compile(r'^[\d\s.,:;+\-*/%<>()\[\]|\\~`!@#$^&_=]+$')
    ok_entries = []
    frame_records = []
    last_err = None

    for f in dedup_items:
        fi = f.get('frame_index', 0)
        pts_val = round(float(f.get('t') or 0.0), 6)
        p = Path(frame_dir) / f"frame_{fi:08d}.jpg"
        rec = {
            'index': f.get('index'),
            'matched_indices': list(fi_to_indices.get(fi, [])),
            'frame_index': fi,
            'pts': pts_val,
            'source_role': f.get('source_role', 'storyboard'),
            'shot_ids': list(f.get('shot_ids') or []),
            'status': 'failed',
            'error': None,
            'raw_results': [],
            'ocr_texts': [],
        }
        if not p.is_file():
            status['failed_frames'] += 1
            err_msg = f"missing frame file: {p.name}"
            last_err = last_err or err_msg
            rec['error'] = err_msg
            frame_records.append(rec)
            continue
        try:
            raw_bytes = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR) if raw_bytes.size else None
            if img is None:
                status['failed_frames'] += 1
                err_msg = f"unreadable image: {p.name}"
                last_err = last_err or err_msg
                rec['error'] = err_msg
                frame_records.append(rec)
                continue
            h, w = img.shape[:2]
            if max(h, w) > 960:
                scale = 960.0 / max(h, w)
                img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            res, _ = engine(img, use_cls=False)
            status['processed_frames'] += 1
            rec['status'] = 'ok'
        except Exception as e:
            status['failed_frames'] += 1
            err_msg = safe_text(e)[:300]
            last_err = err_msg
            rec['error'] = err_msg
            frame_records.append(rec)
            continue

        seen_in_frame = {}
        for item in (res or []):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            box, raw_txt, score = item[0], str(item[1]).strip(), float(item[2] or 0.0)
            txt = to_simplified(safe_text(raw_txt) or '').strip()
            bbox_pts = []
            try:
                bbox_pts = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in box]
            except Exception:
                bbox_pts = []
            filter_reason = None
            if score < 0.68:
                filter_reason = 'low_confidence'
            elif len(txt) < 2:
                filter_reason = 'too_short'
            elif noise_re.match(txt):
                filter_reason = 'punctuation_or_digits_only'
            else:
                has_zh = bool(re.search(r'[\u4e00-\u9fff]', txt))
                has_en_word = bool(re.search(r'[A-Za-z]{3,}', txt))
                has_price = bool(re.search(r'[$¥€£]\s*\d+(?:\.\d+)?', txt))
                if not (has_zh or has_en_word or has_price):
                    filter_reason = 'no_semantic_token'
            rec['raw_results'].append({
                'raw_text': safe_text(raw_txt),
                'text': txt,
                'score': round(score, 4),
                'bbox': bbox_pts,
                'accepted': filter_reason is None,
                'filter_reason': filter_reason,
            })
            if filter_reason is not None:
                continue
            try:
                ys = [pt[1] for pt in box]
                xs = [pt[0] for pt in box]
                rel_h = (max(ys) - min(ys)) / max(h, 1)
                cx = 0.5 * (max(xs) + min(xs)) / max(w, 1)
                cy = 0.5 * (max(ys) + min(ys)) / max(h, 1)
            except Exception:
                rel_h, cx, cy = 0.03, 0.5, 0.5
            center_weight = 1.35 - 0.7 * abs(cx - 0.5)
            len_weight = 1.0 + 0.12 * min(len(txt), 14)
            visual_salience = rel_h * center_weight * len_weight
            if txt not in seen_in_frame or visual_salience > seen_in_frame[txt][0]:
                seen_in_frame[txt] = (visual_salience, cy, txt)

        frame_records.append(rec)
        ok_entries.append({
            'rec': rec,
            'frame_obj': f,
            'frame_index': fi,
            't': pts_val,
            'shot_ids': list(rec['shot_ids']),
            'seen': seen_in_frame,
        })

    # 时间分桶去偏常驻文字统计（消除局部密集采样与短片低阈值导致的误判）
    txt_frame_hits = Counter()
    txt_bins = {}
    txt_ts = {}
    txt_shots = {}
    active_bins = set()
    total_shots_covered = len({sid for e in ok_entries for sid in e['shot_ids']})
    if ok_entries:
        t_vals = [e['t'] for e in ok_entries]
        dur = float((media or {}).get('duration') or 0.0)
        ref_start = 0.0 if dur > 0.0 else min(t_vals)
        ref_end = max(dur, max(t_vals))
        total_span = max(ref_end - ref_start, 1e-3)
        num_bins = max(4, min(12, len(ok_entries)))
        for e in ok_entries:
            bin_id = min(num_bins - 1, max(0, int((e['t'] - ref_start) / total_span * num_bins)))
            e['bin_id'] = bin_id
            active_bins.add(bin_id)
            for txt in e['seen']:
                txt_frame_hits[txt] += 1
                txt_bins.setdefault(txt, set()).add(bin_id)
                txt_ts.setdefault(txt, []).append(e['t'])
                for sid in e['shot_ids']:
                    txt_shots.setdefault(txt, set()).add(sid)
    else:
        total_span = 1.0

    active_bin_count = max(1, len(active_bins))
    raw_recurring_ranked = []
    if len(ok_entries) >= 3 and len(active_bins) >= 2:
        for txt, f_cnt in txt_frame_hits.items():
            b_cnt = len(txt_bins.get(txt, ()))
            bin_cov = b_cnt / active_bin_count
            ts_list = txt_ts.get(txt) or [0.0]
            t_span_ratio = (max(ts_list) - min(ts_list)) / total_span
            shot_cnt = len(txt_shots.get(txt, ()))
            cross_shot_ok = (shot_cnt >= 2) if total_shots_covered >= 2 else (b_cnt >= 3)
            if f_cnt >= 3 and bin_cov >= 0.50 and t_span_ratio >= 0.45 and cross_shot_ok:
                raw_recurring_ranked.append((bin_cov, f_cnt, txt))
    raw_recurring_ranked.sort(key=lambda x: (-x[0], -x[1], -len(x[2]), x[2]))
    raw_recurring = [txt for _, _, txt in raw_recurring_ranked[:12]]

    # 合并常驻角标的残缺子串（例如 "拟演绎·仅供娱乐" 归并入完整的 "虚拟演绎·仅供娱乐·内容AI生成"）
    recurring = []
    for t in sorted(raw_recurring, key=len, reverse=True):
        if not any(t in longer for longer in recurring):
            recurring.append(t)

    def is_recurring_variant(txt):
        return any(txt in r or r in txt for r in recurring if len(txt) >= 4 and len(r) >= 4)

    ocr_by_index = {}
    for e in ok_entries:
        candidates = list(e['seen'].values())
        if not candidates:
            continue
        ranked = []
        for vis_sal, cy, txt in candidates:
            if is_recurring_variant(txt):
                continue
            bin_df = len(txt_bins.get(txt, ())) or 1
            idf = math.log(1.0 + active_bin_count / bin_df)
            final_score = vis_sal * idf
            ranked.append((final_score, cy, txt))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        top_texts = [txt for _, _, txt in ranked[:5]]
        if top_texts:
            e['rec']['ocr_texts'] = top_texts
            e['frame_obj']['ocr_texts'] = top_texts
            for m_idx in e['rec']['matched_indices']:
                ocr_by_index[m_idx] = top_texts

    # 同步回写传给函数调用者的原列表中的 ocr_texts（防止重复 frame_index 的镜像条目漏掉）
    for f in (storyboard_items or []):
        idx = f.get('index')
        if idx in ocr_by_index and not f.get('ocr_texts'):
            f['ocr_texts'] = ocr_by_index[idx]

    top_recurring = recurring[:8]
    if status['failed_frames'] and status['processed_frames']:
        status['status'] = 'partial_error'
    elif status['failed_frames'] and not status['processed_frames']:
        status['status'] = 'error'
    else:
        status['status'] = 'ok'
    status['error'] = last_err
    status['frames'] = frame_records
    return ocr_by_index, top_recurring, status


# 基于相邻两帧与转场分数，生成客观的转场手法与前后物理/OCR 变化说明。
def describe_frame_delta(prev_f, curr_f, cut_score=None, ocr_by_index=None, extra_note=''):
    parts = []
    pv = (prev_f or {}).get('visual') or {}
    cv = (curr_f or {}).get('visual') or {}
    p_luma = pv.get('luma')
    c_luma = cv.get('luma')
    p_warm = pv.get('warm')
    c_warm = cv.get('warm')

    if prev_f is None:
        if c_luma is not None and c_luma < 24:
            parts.append('暗场起幅')
        else:
            parts.append('开场画面')
    else:
        if p_luma is not None and c_luma is not None:
            cut_prefix = f'硬切(分{cut_score:.2f})且' if cut_score is not None else ''
            if p_luma >= 160 and c_luma <= 105:
                parts.append(f'{cut_prefix}高亮转暗(亮{round(p_luma)}→{round(c_luma)})')
            elif p_luma < 22 and c_luma >= 32:
                parts.append(f'{cut_prefix}极暗转亮(亮{round(p_luma)}→{round(c_luma)})')
            elif c_luma < 22 and p_luma >= 40:
                parts.append(f'{cut_prefix}亮度骤降至极暗(亮{round(p_luma)}→{round(c_luma)})')
            elif cut_score is not None:
                parts.append(f'硬切转场(切镜分{cut_score:.2f})')
            elif abs(c_luma - p_luma) >= 35:
                arrow = '骤亮' if c_luma > p_luma else '骤暗'
                parts.append(f'亮度{arrow}(亮{round(p_luma)}→{round(c_luma)})')
            else:
                parts.append('连续推进' if not extra_note else extra_note)
        elif cut_score is not None:
            parts.append(f'硬切转场(切镜分{cut_score:.2f})')
        else:
            parts.append(extra_note or '镜头衔接')

        if p_luma is not None and c_luma is not None and abs(c_luma - p_luma) >= 22 and not any('亮' in x for x in parts):
            parts.append(f'亮度{round(p_luma)}→{round(c_luma)}')
        if p_warm is not None and c_warm is not None and abs(c_warm - p_warm) >= 18:
            w_desc = '转暖调' if c_warm > p_warm else '转冷调'
            parts.append(f'{w_desc}(温{round(p_warm)}→{round(c_warm)})')

    if extra_note and extra_note not in ' '.join(parts):
        parts.append(extra_note)

    if ocr_by_index and curr_f:
        p_txt = set(ocr_by_index.get(prev_f['index'], [])) if prev_f else set()
        c_txt = ocr_by_index.get(curr_f['index'], [])
        added = [t for t in c_txt if t not in p_txt]
        if added:
            parts.append('新增字:' + '/'.join(added[:2]))
    return ' · '.join(parts) if parts else '镜头延续'


# 挑选全片最关键的切镜点与高动态密帧连拍区间（最多 max_strips 条，每条 4 帧），用于生成 transitions.jpg。
def select_transition_strips(items, scene_cuts, lighting_shifts=None, ocr_by_index=None, max_strips=10):
    if len(items) < 2:
        return []
    lighting_shifts = lighting_shifts or set()
    ocr_by_index = ocr_by_index or {}
    pos_by_idx = {f['index']: i for i, f in enumerate(items)}
    cut_scores = {f['index']: (f.get('scores') or {}).get('scene_score') for f in items if 'scene_after' in (f.get('reasons') or [])}

    candidates = []
    # 1. 硬切点：抓取切前一帧 -> 切入首帧 -> 切后连拍帧（最多 4 帧）
    for i, f in enumerate(items):
        reasons = f.get('reasons') or []
        if 'scene_after' in reasons and i >= 1:
            end_i = min(len(items), max(0, i - 1) + 4)
            start_i = max(0, end_i - 4)
            win = items[start_i:end_i]
            prev_f = items[i - 1]
            sc = cut_scores.get(f['index'])
            luma_delta = abs(((f.get('visual') or {}).get('luma') or 0) - ((prev_f.get('visual') or {}).get('luma') or 0))
            priority = 100.0 + float(sc or 0.35) * 40.0 + luma_delta * 0.4
            desc = describe_frame_delta(prev_f, f, cut_score=sc, ocr_by_index=ocr_by_index)
            candidates.append({'kind': '硬切与后继连拍', 'anchor_t': f['t'], 't0': win[0]['t'], 't1': win[-1]['t'],
                               'priority': priority, 'frames': win[:4], 'summary': desc})

    # 2. 无硬切光影转折点
    for idx in sorted(lighting_shifts):
        i = pos_by_idx.get(idx)
        if i is None or i < 1:
            continue
        end_i = min(len(items), max(0, i - 2) + 4)
        start_i = max(0, end_i - 4)
        win = items[start_i:end_i]
        desc = describe_frame_delta(win[0], items[i], cut_score=None, ocr_by_index=ocr_by_index, extra_note='连续光影转折')
        candidates.append({'kind': '光影渐变连拍', 'anchor_t': items[i]['t'], 't0': win[0]['t'], 't1': win[-1]['t'],
                           'priority': 75.0, 'frames': win[:4], 'summary': desc})

    # 3. 高动态连续密集抽帧段落（把 dynamic_frames 多抽出来的局部动作帧真正展现为 4 格动作连拍）
    run = []
    bursts = []
    for f in items:
        rs = set(f.get('reasons') or [])
        local = float((f.get('scores') or {}).get('local_max_diff') or 0.0)
        if ('local_motion' in rs or 'cumulative_change' in rs) and local >= 18.0:
            if not run or (f['t'] - run[-1]['t'] <= 0.8 and f['t'] - run[0]['t'] <= 6.0):
                run.append(f)
            else:
                if len(run) >= 4:
                    bursts.append(run)
                run = [f]
        else:
            if len(run) >= 4:
                bursts.append(run)
            run = []
    if len(run) >= 4:
        bursts.append(run)
    for b in bursts:
        picks = [b[min(len(b) - 1, round(k * (len(b) - 1) / 3))] for k in range(4)]
        peak = max(float((x.get('scores') or {}).get('local_max_diff') or 0.0) for x in b)
        desc = describe_frame_delta(picks[0], picks[-1], cut_score=None, ocr_by_index=ocr_by_index,
                                    extra_note=f'共{len(b)}密帧(运动峰值{round(peak, 1)})')
        candidates.append({'kind': '高动态动作连拍', 'anchor_t': picks[1]['t'], 't0': b[0]['t'], 't1': b[-1]['t'],
                           'priority': 50.0 + min(peak, 80.0) * 0.5 + min(len(b), 20), 'frames': picks, 'summary': desc})

    # 去重后按时间排序：同一 1.5s 邻近窗口只保留优先级最高的连拍条
    selected = []
    for c in sorted(candidates, key=lambda x: -x['priority']):
        if len(selected) >= max_strips:
            break
        if any(abs(c['anchor_t'] - s['anchor_t']) < 1.5 for s in selected):
            continue
        selected.append(c)
    selected.sort(key=lambda x: x['anchor_t'])
    for idx, s in enumerate(selected, 1):
        s['id'] = idx
    return selected


# 将长无语音区间按「场景硬切点 (scene_after)」、「光影/色调显著转折点」或最大时间跨度拆分为易读的分镜段落，并计算跨镜转场说明、所在备查图索引及局部音频能量摘要。
def build_smart_shots(timeline, keyframe_indices, ocr_by_index=None, lighting_shifts=None, max_silent_span=15.0, sheet_map=None, audio_energy=None):
    ocr_by_index = ocr_by_index or {}
    lighting_shifts = lighting_shifts or set()
    sheet_map = sheet_map or {}
    preliminary = partition_preliminary_shots(timeline, lighting_shifts=lighting_shifts, max_silent_span=max_silent_span)
    shots = [
        _summarize_shot(
            p['type'], p['start'], p['end'], p['text'], p['frames'],
            keyframe_indices, ocr_by_index, lighting_shifts, sheet_map, audio_energy=audio_energy
        )
        for p in preliminary
    ]
    cleaned = []
    for sh in shots:
        if sh['type'] == 'no-speech' and sh['duration'] <= 0.35 and not sh['ocr_texts'] and sh['frame_count'] <= 2 and cleaned:
            prev = cleaned[-1]
            merged_frames = (prev.get('_frames') or []) + (sh.get('_frames') or [])
            new_end = max(prev['end'], sh['end'])
            cleaned[-1] = _summarize_shot(
                prev['type'], prev['start'], new_end, prev['text'],
                merged_frames, keyframe_indices, ocr_by_index, lighting_shifts, sheet_map, audio_energy=audio_energy
            )
            continue
        cleaned.append(sh)

    prev_tail = None
    for idx, sh in enumerate(cleaned):
        sh.pop('_frames', None)
        head = sh.pop('_head_frame', None)
        tail = sh.pop('_tail_frame', None)
        if idx == 0:
            sh['transition_desc'] = describe_frame_delta(None, head, None, ocr_by_index)
        else:
            cut_sc = (head.get('scores') or {}).get('scene_score') if head and 'scene_after' in (head.get('reasons') or []) else None
            sh['transition_desc'] = describe_frame_delta(prev_tail, head, cut_sc, ocr_by_index)
        if tail is not None:
            prev_tail = tail
    expected_ids = [f['index'] for row in (timeline or []) for f in (row.get('frames') or [])]
    got_ids = [idx for sh in cleaned for idx in (sh.get('frame_indices') or [])]
    if len(got_ids) != len(expected_ids) or len(set(got_ids)) != len(expected_ids) or set(got_ids) != set(expected_ids):
        raise RuntimeError(f'[align] Smart shot frame partition mismatch: expected {len(expected_ids)}, got {len(got_ids)}')
    return cleaned


def _summarize_shot(kind, start, end, text, frames, keyframe_indices, ocr_by_index, lighting_shifts=None, sheet_map=None, audio_energy=None):
    lighting_shifts = lighting_shifts or set()
    sheet_map = sheet_map or {}
    shot_audio = summarize_interval_audio_energy(audio_energy, start, end) if audio_energy else None
    if not frames:
        res = {'type': kind, 'start': start, 'end': end, 'duration': round(max(0.0, float(end) - float(start)), 2),
               'text': text, 'frame_count': 0, 'frame_indices': [], 'frame_range': '—', 'timeline_sheets': [], 'motion_tag': '无帧', 'keyframes': [], 'ocr_texts': [], '_frames': [], '_head_frame': None, '_tail_frame': None}
        if shot_audio is not None:
            res['audio_energy'] = shot_audio
        return res
    first_i, last_i = frames[0]['index'], frames[-1]['index']
    frame_range = f"#{first_i}" if first_i == last_i else f"#{first_i}–#{last_i} ({len(frames)}帧)"
    sheets_used = list(dict.fromkeys(sheet_map[f['index']] for f in frames if f['index'] in sheet_map))
    reasons_flat = [r for f in frames for r in (f.get('reasons') or [])]
    cuts = sum(1 for r in reasons_flat if r == 'scene_after')
    has_light_shift = any(f['index'] in lighting_shifts for f in frames)
    cum_changes = sum(1 for r in reasons_flat if r == 'cumulative_change')
    static_cnt = sum(1 for r in reasons_flat if r == 'static_interval')
    local_diffs = [float((f.get('scores') or {}).get('local_max_diff') or 0.0) for f in frames if (f.get('scores') or {}).get('local_max_diff') is not None]
    mean_local = round(sum(local_diffs) / len(local_diffs), 1) if local_diffs else 0.0
    tags = []
    if cuts:
        tags.append(f'含{cuts}次切镜')
    elif has_light_shift:
        tags.append('光影/场景过渡')
    if static_cnt > len(frames) * 0.4:
        tags.append('静态/慢节奏')
    elif cum_changes > len(frames) * 0.6 or mean_local >= 35.0:
        tags.append(f'高动态(运动{mean_local})')
    elif mean_local >= 18.0:
        tags.append(f'中动态(运动{mean_local})')
    else:
        tags.append('平稳动态')
    vis_list = [f['visual'] for f in frames if f.get('visual')]
    if vis_list:
        v_luma = sum(v['luma'] for v in vis_list) / len(vis_list)
        v_con = sum(v['contrast'] for v in vis_list) / len(vis_list)
        v_sat = sum(v['sat'] for v in vis_list) / len(vis_list)
        v_warm = sum(v['warm'] for v in vis_list) / len(vis_list)
        tags.append(format_lighting_tag(v_luma, v_con, v_sat, v_warm))
    kfs = [f"★#{f['index']}({stamp(f['t'])})" for f in frames if f['index'] in keyframe_indices]
    if not kfs:
        mid = frames[len(frames) // 2]
        kfs = [f"#{mid['index']}({stamp(mid['t'])})"]
    shot_ocr = []
    for f in frames:
        for t in ocr_by_index.get(f['index'], []):
            if t not in shot_ocr:
                shot_ocr.append(t)
    res = {
        'type': kind,
        'start': round(float(start), 2),
        'end': round(float(end), 2),
        'duration': round(max(0.0, float(end) - float(start)), 2),
        'text': text if kind == 'speech' else '',
        'frame_count': len(frames),
        'frame_indices': [f['index'] for f in frames],
        'frame_range': frame_range,
        'timeline_sheets': sheets_used,
        'motion_tag': ' · '.join(tags),
        'keyframes': kfs,
        'ocr_texts': shot_ocr[:5],
        '_frames': list(frames),
        '_head_frame': frames[0],
        '_tail_frame': frames[-1]
    }
    if shot_audio is not None:
        res['audio_energy'] = shot_audio
    return res


# 综合计算视频风格、剪辑节奏、声音结构、客观音频能量、核心画面文字与叙事画像，供 AI 速读。
def build_style_profile(media, frames_info, transcript, timeline, visual_features, storyboard_items, ocr_by_index=None, recurring_ocr=None, lighting_shifts=None, sheet_map=None, transition_strips=None, audio_energy=None):
    duration = float((media or {}).get('duration') or 0.0)
    cuts = (frames_info or {}).get('scene_cuts') or []
    segments = (transcript or {}).get('segments') or []
    lighting_shifts = lighting_shifts or set()

    cut_times = sorted(float(c.get('t', 0.0)) for c in cuts)
    bounds = [0.0] + cut_times + ([duration] if duration > 0 else [])
    shot_lengths = [round(b - a, 2) for a, b in zip(bounds, bounds[1:]) if b - a > 0.02]
    avg_shot = round(sum(shot_lengths) / len(shot_lengths), 2) if shot_lengths else round(duration, 2)
    min_shot = min(shot_lengths) if shot_lengths else round(duration, 2)
    max_shot = max(shot_lengths) if shot_lengths else round(duration, 2)
    if len(cuts) == 0:
        pacing_tag = f'一镜到底 / 无硬切长镜头（检出 {len(lighting_shifts)} 处连续光影/场景过渡）' if lighting_shifts else '一镜到底 / 无硬切长镜头'
    elif avg_shot < 3.0:
        pacing_tag = '高频快剪型（镜头切换极快）'
    elif avg_shot < 8.0:
        pacing_tag = '中等剪辑节奏'
    else:
        pacing_tag = '长镜头为主 + 局部节点快切'

    burst_zone = None
    if len(cut_times) >= 3:
        best_cnt, best_span = 0, None
        for i in range(len(cut_times)):
            j = i
            while j < len(cut_times) and cut_times[j] - cut_times[i] <= 12.0:
                j += 1
            if j - i > best_cnt:
                best_cnt = j - i
                best_span = (cut_times[i], cut_times[j - 1])
        if best_cnt >= 3 and best_span:
            burst_zone = f"{stamp(best_span[0])}–{stamp(best_span[1])}（{round(best_span[1]-best_span[0], 1)}秒内连续切镜 {best_cnt} 次）"

    speech_dur = round(sum(max(0.0, float(s.get('end', 0)) - float(s.get('start', 0))) for s in segments), 2)
    speech_ratio = round(speech_dur / duration * 100.0, 1) if duration > 0 else 0.0
    total_chars = sum(len(re.sub(r'\s+', '', s.get('text') or '')) for s in segments)
    cpm = round(total_chars / (speech_dur / 60.0)) if speech_dur > 1.0 else 0
    has_audio = (media or {}).get('has_audio')
    if has_audio is False:
        audio_style = '无音轨（未进行语音识别）'
    elif not segments:
        audio_style = '纯配乐/环境音展示（未检测到语音片段）'
    elif speech_ratio >= 65.0:
        audio_style = f'高密度口播/解说驱动（人声占 {speech_ratio}%，说话段语速约 {cpm} 字/分）'
    elif speech_ratio <= 25.0:
        audio_style = f'画面与配乐驱动 + 关键节点人声点睛（人声仅占 {speech_ratio}%，其余 {round(100-speech_ratio, 1)}% 为纯画面/BGM留白）'
    else:
        audio_style = f'解说与画面留白交替（人声占 {speech_ratio}%，说话段语速约 {cpm} 字/分）'

    kf_set = {f['index'] for f in (storyboard_items or [])}
    smart_shots = build_smart_shots(timeline, kf_set, ocr_by_index=ocr_by_index, lighting_shifts=lighting_shifts, sheet_map=sheet_map, audio_energy=audio_energy)

    structure_parts = []
    if segments and duration > 0:
        silent_rows = [r for r in (timeline or []) if r.get('type') == 'no-speech']
        longest_silent = max(silent_rows, key=lambda r: float(r.get('end', 0)) - float(r.get('start', 0)), default=None)
        if longest_silent and (float(longest_silent['end']) - float(longest_silent['start'])) >= duration * 0.35:
            s0, s1 = float(longest_silent['start']), float(longest_silent['end'])
            if s0 > 1.5:
                structure_parts.append(f"开场人声引入 ({stamp(0)}–{stamp(s0)})")
            structure_parts.append(f"中段纯配乐/画面高能展示 ({stamp(s0)}–{stamp(s1)}，长达 {round(s1-s0, 1)}s)")
            if duration - s1 > 1.5:
                structure_parts.append(f"后段人声高潮与收尾 ({stamp(s1)}–{stamp(duration)})")
    if not structure_parts and smart_shots:
        if len(smart_shots) > 1 and not segments:
            structure_parts = [f"段落{idx+1} ({stamp(sh['start'])}–{stamp(sh['end'])}: {sh['motion_tag'].split(' · ')[-1]})" for idx, sh in enumerate(smart_shots[:6])]
        else:
            structure_parts.append(f"全片连续展开 ({stamp(0)}–{stamp(duration)}，共 {len(segments)} 段台词 / {len(smart_shots)} 个分镜段)")

    audio_energy_summary = None
    if isinstance(audio_energy, dict) and audio_energy.get('status') == 'ok':
        audio_energy_summary = {
            'status': 'ok',
            'rms_dbfs': audio_energy.get('rms_dbfs'),
            'peak_dbfs': audio_energy.get('peak_dbfs'),
            'window_silence_ratio': audio_energy.get('window_silence_ratio'),
            'silent_duration_ratio': audio_energy.get('silent_duration_ratio'),
            'av_start_offset_seconds': audio_energy.get('av_start_offset_seconds'),
            'empty_window_count': audio_energy.get('empty_window_count'),
            'low_energy_regions': audio_energy.get('low_energy_regions', []),
            'energy_surges': audio_energy.get('energy_surges', []),
            'interpretation_note': '能量指标仅反映声压级客观变化，不能仅凭此判断是否为平稳配乐或无人声。',
        }
    elif isinstance(audio_energy, dict):
        audio_energy_summary = {
            'status': audio_energy.get('status'),
            'has_audio': audio_energy.get('has_audio'),
        }

    return {
        'visual': visual_features or {},
        'editing_pacing': {
            'scene_cut_count': len(cuts),
            'lighting_transition_count': len(lighting_shifts),
            'avg_shot_seconds': avg_shot,
            'min_shot_seconds': min_shot,
            'max_shot_seconds': max_shot,
            'pacing_tag': pacing_tag,
            'fast_cut_burst_zone': burst_zone
        },
        'audio_narrative': {
            'speech_segment_count': len(segments),
            'speech_duration_seconds': speech_dur,
            'speech_coverage_pct': speech_ratio,
            'total_chinese_chars': total_chars,
            'speech_rate_cpm': cpm,
            'audio_style': audio_style,
            'narrative_structure': ' → '.join(structure_parts)
        },
        'audio_energy': audio_energy_summary,
        'recurring_onscreen_texts': recurring_ocr or [],
        'storyboard_keyframe_count': len(storyboard_items or []),
        'transition_strip_count': len(transition_strips or []),
        'transition_strips': [
            {'id': s['id'], 'kind': s['kind'], 't0': round(s['t0'], 2), 't1': round(s['t1'], 2),
             'frame_indices': [f['index'] for f in s['frames']], 'summary': s['summary']}
            for s in (transition_strips or [])
        ],
        'smart_shots': smart_shots
    }


# 输入分析结果字典；返回高信息密度、分级按需读取、极省 AI Token 的结构化 Markdown 报告。
def render_markdown(data):
    def cell(v):
        if v is None or v == '': return '—'
        if isinstance(v, (dict, list)): v = json.dumps(v, ensure_ascii=False)
        return safe_text(v).replace('|', '\\|').replace('\n', ' ')

    meta = data.get('metadata') or {}
    media = data.get('media') or {}
    frames = data.get('frames') or {}
    transcript = data.get('transcript') or {}
    style = data.get('style_profile') or {}
    visual = style.get('visual') or {}
    pacing = style.get('editing_pacing') or {}
    narrative = style.get('audio_narrative') or {}
    audio_energy = style.get('audio_energy') or data.get('audio_energy') or {}
    recurring_ocr = style.get('recurring_onscreen_texts') or []
    trans_strips = style.get('transition_strips') or []
    clips_info = frames.get('exported_clips') or []

    lines = ['# 视频深度分析报告（AI 分层速读版）', '']
    run_status = data.get('status')
    if run_status and run_status != 'ok':
        fail_stage = data.get('failed_stage') or 'unknown'
        err_note = cell(data.get('error') or '')
        lines += [
            f"> ⚠️ **警告：本次运行未完整成功（状态：`{run_status}`，阶段：`{fail_stage}`）**。以下仅包含失败前记录与诊断信息，**不得将部分产物视为完整验收结果**。错误：{err_note}",
            ''
        ]
    lines += [
        '> **AI 分层读片指引（先粗后细、省 Token 不丢动作）**：\n'
        '> 1. **第一层（全片总览）**：优先读 `storyboard.jpg`（精选核心分镜）+ 下面结构化表格。\n'
        '> 2. **第二层（转场与高动态连拍）**：直接读 `transitions.jpg`（将关键硬切前后 4 帧与高动态动作段并排连拍，专治画面跨度大）。\n'
        '> 3. **第三层（逐帧局部细查）**：`timeline_00X.jpg` **100% 完整保留了动态抽帧选出的全部代表帧**（零丢帧），根据分镜表最右列精准只打开对应那 1 张图即可。',
        ''
    ]

    ocr_status = frames.get('ocr_status') or {}
    ocr_state = ocr_status.get('status')
    if ocr_state == 'unavailable':
        ocr_summary = f"OCR 识别器不可用（{ocr_status.get('error') or '未安装或初始化失败'}）"
    else:
        base_ocr = '、'.join(recurring_ocr) if recurring_ocr else '未检出明显高频常驻字'
        if ocr_state in ('error', 'partial_error'):
            ocr_summary = f"{base_ocr}（OCR 状态：{ocr_state}，失败 {ocr_status.get('failed_frames')}/{ocr_status.get('total_frames')} 帧）"
        else:
            ocr_summary = base_ocr

    ocr_cov_line = None
    if ocr_status and ocr_status.get('preliminary_shots_with_frames_count') is not None:
        cov_shots = ocr_status.get('covered_shot_count', 0)
        tot_shots = ocr_status.get('preliminary_shots_with_frames_count', 0)
        full_shots = ocr_status.get('fully_covered_shot_count', 0)
        sb_cnt = ocr_status.get('storyboard_frames_count', 0)
        supp_cnt = ocr_status.get('supplemented_frames_count', 0)
        dedup_cnt = ocr_status.get('deduplicated_overlap_count', 0)
        single_cnt = ocr_status.get('single_frame_shot_count', 0)
        cov_parts = [
            f"共识别 {ocr_status.get('processed_frames', 0)}/{ocr_status.get('total_frames', 0)} 帧（精选分镜 {sb_cnt} 帧 + 初步分镜补充 {supp_cnt} 帧，跳过重合 {dedup_cnt} 次，失败 {ocr_status.get('failed_frames', 0)} 帧）",
            f"有帧初步分镜至少覆盖 1 帧：{cov_shots}/{tot_shots}，满足目标(1-2帧)：{full_shots}/{tot_shots}",
        ]
        if single_cnt:
            cov_parts.append(f"其中 {single_cnt} 个初步分镜可用源帧不足 2 帧")
        if ocr_status.get('budget_exceeded'):
            cov_parts.append(f"⚠️ 覆盖不足（超出 OCR 预算上限 {ocr_status.get('max_ocr_frames_budget')} 帧，未完成全部初步分镜扫描）")
        ocr_cov_line = '；'.join(cov_parts)

    audio_energy_line = None
    if media.get('has_audio') is False or audio_energy.get('status') == 'no_audio_track':
        audio_energy_line = '无音轨（未提取音频能量指标）'
    elif audio_energy.get('status') == 'empty_audio':
        audio_energy_line = '音轨为空或无有效音频样本覆盖（未产出有效 RMS/峰值指标）'
    elif audio_energy.get('status') == 'ok':
        ae_parts = [
            f"全片 RMS **{audio_energy.get('rms_dbfs')} dBFS**",
            f"峰值 **{audio_energy.get('peak_dbfs')} dBFS**",
            f"窗口静音占比 **{round(float(audio_energy.get('window_silence_ratio') or 0.0) * 100.0, 1)}%**",
        ]
        low_regs = audio_energy.get('low_energy_regions') or []
        if low_regs:
            reg_str = '、'.join(f"{stamp(r['start'])}–{stamp(r['end'])}" for r in low_regs[:4])
            ae_parts.append(f"低能量区间 {len(low_regs)} 段（{reg_str}{' 等' if len(low_regs) > 4 else ''}）")
        else:
            ae_parts.append('未检出连续低能量区间')
        surges = audio_energy.get('energy_surges') or []
        if surges:
            s_str = '、'.join(f"{stamp(sg['start'])}" for sg in surges[:4])
            ae_parts.append(f"能量突增 {len(surges)} 处（{s_str}{' 等' if len(surges) > 4 else ''}）")
        else:
            ae_parts.append('未检出显著能量突增')
        av_off = float(audio_energy.get('av_start_offset_seconds') or 0.0)
        if abs(av_off) >= 0.02:
            ae_parts.append(f"音视频起点偏移 {av_off:+.3f}s")
        empty_cnt = int(audio_energy.get('empty_window_count') or 0)
        if empty_cnt > 0:
            ae_parts.append(f"无样本覆盖空窗口 {empty_cnt} 个")
        ae_parts.append('注：仅作声压级客观描述，不单独判定配乐/人声')
        audio_energy_line = '；'.join(ae_parts)

    if style:
        lines += [
            '## 一、 视频风格与视听节奏画像',
            '',
            f"- **画面与色彩基调**：{cell(visual.get('visual_summary') or '未提取')}",
            f"- **剪辑与镜头节奏**：{cell(pacing.get('pacing_tag'))}（全片硬切点 **{cell(pacing.get('scene_cut_count'))}** 处，平均镜头时长 **{cell(pacing.get('avg_shot_seconds'))}s**，最短 **{cell(pacing.get('min_shot_seconds'))}s** / 最长 **{cell(pacing.get('max_shot_seconds'))}s**"
            + (f"；快剪爆发区：**{cell(pacing.get('fast_cut_burst_zone'))}**）" if pacing.get('fast_cut_burst_zone') else '）'),
            f"- **声音与叙事驱动**：{cell(narrative.get('audio_style'))}",
        ]
        if audio_energy_line:
            lines.append(f"- **音频能量指标（客观波形统计）**：{cell(audio_energy_line)}")
        lines += [
            f"- **全片篇章结构**：{cell(narrative.get('narrative_structure'))}",
            f"- **高频常驻画面文字 / 角标 (OCR)**：{cell(ocr_summary)}",
        ]
        if ocr_cov_line:
            lines.append(f"- **OCR 覆盖审计**：{cell(ocr_cov_line)}")
        lines.append('')
    warnings_list = [w for w in (data.get('warnings') or []) if w]
    if warnings_list:
        lines += ['## 运行提示与告警', ''] + [f"- {cell(w)}" for w in warnings_list] + ['']

    lines += ['## 二、 元数据与作品信息', '', '| 字段 | 值 | 来源 |', '| --- | --- | --- |']
    sources = data.get('metadata_sources') or {}
    inp = data.get('input') or {}
    if inp.get('platform'):
        lines.append(f"| route_platform | {cell(inp.get('platform'))} | {cell(inp.get('route_kind') or 'route')} |")
    if inp.get('platform_item_id'):
        lines.append(f"| platform_item_id | {cell(inp.get('platform_item_id'))} | {cell(inp.get('downloader') or 'route')} |")
    for k, v in meta.items():
        if k in ('platform', 'platform_item_id') and inp.get(k):
            continue
        lines.append(f'| {cell(k)} | {cell(v)} | {cell(sources.get(k))} |')
    if not meta:
        lines.append('| 元数据 | 不可用 | 本地文件或未获取 |')
    missing_meta = data.get('missing_metadata') or {}
    if missing_meta:
        miss_items = [f"{k}: {v}" for k, v in list(missing_meta.items())[:8]]
        if len(missing_meta) > 8:
            miss_items.append(f"等共 {len(missing_meta)} 项缺失字段")
        lines += ['', f"- **元数据缺失说明**：{cell('；'.join(miss_items))}"]
    lines += [
        '',
        '## 三、 媒体与抽帧统计',
        '',
        '| 字段 | 值 |',
        '| --- | --- |',
        *[f'| {cell(k)} | {cell(v)} |' for k, v in media.items()],
        '',
        f"- **自适应代表帧总数**：{cell(frames.get('count'))} 帧（对比 10 fps 基准 {cell(frames.get('baseline_10fps'))} 帧，节省比例：{cell(frames.get('saving_ratio'))}）",
        f"- **精选核心分镜 (`storyboard.jpg`)**：{cell(style.get('storyboard_keyframe_count') or len(frames.get('storyboard_frames') or []))} 帧",
        f"- **关键转场与动作连拍 (`transitions.jpg`)**：{cell(len(trans_strips))} 组连拍条（每组最多 4 帧）",
        f"- **全量动态帧时间轴拼图 (`timeline_00X.jpg`)**：{cell(frames.get('timeline_sheet_count') or 0)} 张（100% 覆盖全部 {cell(frames.get('count'))} 帧，零删减）",
        ''
    ]
    if clips_info:
        clip_parts = []
        for c in clips_info:
            imgs = ', '.join(x for x in (c.get('images') or [c.get('image')]) if x) or '未产出图'
            pts0 = c.get('actual_first_pts')
            pts1 = c.get('actual_last_pts')
            pts_span = f"PTS {stamp(pts0)}–{stamp(pts1)}" if pts0 is not None and pts1 is not None else "PTS 无"
            reasons = c.get('coverage_insufficient_reasons') or []
            reason_note = f"; 覆盖说明:{','.join(reasons)}" if reasons else ''
            src_label = c.get('source_quality_label') or c.get('source_kind') or '原片'
            batch_note = f", 追加批次#{c.get('append_batch_id')}" if c.get('appended') else ''
            clip_parts.append(
                f"{cell(imgs)} (请求 {stamp(c.get('requested_start', c.get('start')))}–{stamp(c.get('requested_end', c.get('end')))} | {pts_span}, {c.get('frame_count')}帧, 来源:{cell(src_label)}, 时间源:{c.get('time_source', '—')}{batch_note}{reason_note})"
            )
        lines.append(f"- **已导出指定片段细查图 (`--clips`)**：{'; '.join(clip_parts)}")
        lines.append('')

    proxy_info = data.get('proxy') or {}
    if proxy_info.get('enabled'):
        p_gen = proxy_info.get('generation_params') or {}
        p_map = proxy_info.get('time_mapping') or {}
        if proxy_info.get('available'):
            lines += [
                f"- **低码率视频代理 (`{cell(proxy_info.get('path'))}`)**：尺寸 **{cell(proxy_info.get('width'))}×{cell(proxy_info.get('height'))}**，"
                f"帧率 **{cell(round(float(proxy_info.get('fps') or 0.0), 3))} fps**，时长 **{cell(round(float(proxy_info.get('duration') or 0.0), 3))}s**，"
                f"大小 **{cell(proxy_info.get('size_mb'))} MB**；参数 `{cell(p_gen.get('video_codec'))}` CRF={cell(p_gen.get('crf'))} / `fps_mode={cell(p_gen.get('fps_mode'))}`；"
                f"时间映射 `{cell(p_map.get('mode'))}`（最大 PTS 偏差 {cell(p_map.get('max_abs_pts_diff_seconds'))}s，是否降帧：{cell(proxy_info.get('downsampled'))}）；"
                f"说明：{cell(proxy_info.get('limitation_note'))}",
                ''
            ]
        else:
            lines.append(f"- **低码率视频代理**：未可用（状态：{cell(proxy_info.get('status'))}，原因：{cell(proxy_info.get('error'))}）")
            lines.append('')

    append_hist = data.get('clip_append_history') or []
    if append_hist:
        hist_parts = []
        for h in append_hist:
            hist_parts.append(
                f"批次#{h.get('batch_id')} ({cell(h.get('requested_clips'))}, 来源:{cell(h.get('source_kind'))}, 执行:{','.join(h.get('executed_stages') or [])}, 跳过:{','.join(h.get('skipped_stages') or [])}, 新增Clips:{h.get('appended_clip_ids')})"
            )
        lines.append(f"- **Clips 增量追加记录**：{'; '.join(hist_parts)}")
        lines.append('')

    lines += ['## 四、 语音转写全文（带时间戳）', '']
    for s in transcript.get('segments', []):
        lines.append(f"- **[{stamp(s.get('start'))}–{stamp(s.get('end'))}]** {cell(s.get('text'))}")
    if not transcript.get('segments'):
        if media.get('has_audio') is False:
            lines.append('视频没有音轨，未进行语音识别。')
        else:
            lines.append('未检测到语音片段。')
    filtered_segs = transcript.get('filtered_segments') or []
    if filtered_segs:
        reason_counts = Counter(fs.get('reason') or 'unknown' for fs in filtered_segs)
        reason_text = '，'.join(f'{k}: {v}条' for k, v in sorted(reason_counts.items()))
        lines.append(f"- *(ASR 过滤追溯：共过滤 {len(filtered_segs)} 条候选片段 [{reason_text}]，明细见 `analysis.json` `transcript.filtered_segments`)*")

    if trans_strips:
        lines += ['', '## 五、 关键转场与高动态动作连拍索引（对应 `transitions.jpg`）', '']
        lines += ['| 连拍组 | 类型 | 时间区间 | 包含帧号 | 客观转场与光影/文字变化 |', '| --- | --- | --- | --- | --- |']
        for st in trans_strips:
            fids = ' → '.join(f'#{i}' for i in st.get('frame_indices') or [])
            lines.append(f"| 第{st['id']}行 | {cell(st.get('kind'))} | {stamp(st.get('t0'))}–{stamp(st.get('t1'))} | {fids} | {cell(st.get('summary'))} |")

    lines += ['', '## 六、 智能分镜与视听对齐时间轴（含转场过渡与按需详图索引）', '']
    smart_shots = style.get('smart_shots')
    if smart_shots:
        lines += [
            '| 时间区间 (时长) | 同期人声台词 | 画面显著文字 (OCR) | 转场入镜与前后变化 | 镜头运动与光影色调 | 核心分镜 (`storyboard.jpg`) | 全部动态帧 / 对应细节图 (`timeline`) |',
            '| --- | --- | --- | --- | --- | --- | --- |'
        ]
        for sh in smart_shots:
            speech_col = sh.get('text') if sh.get('type') == 'speech' else '*(未检出语音)*'
            ocr_col = ' / '.join(sh.get('ocr_texts') or []) or '—'
            trans_col = sh.get('transition_desc') or '—'
            kfs = ', '.join(sh.get('keyframes') or [])
            sheets_list = sh.get('timeline_sheets') or []
            sheet_ref = (f" · {', '.join(sheets_list)}") if sheets_list else ''
            lines.append(
                f"| {stamp(sh.get('start'))}–{stamp(sh.get('end'))} ({sh.get('duration')}s) | {cell(speech_col)} | {cell(ocr_col)} | {cell(trans_col)} | {cell(sh.get('motion_tag'))} | {cell(kfs)} | {cell(sh.get('frame_range') + sheet_ref)} |"
            )
    else:
        lines += ['| 时间 | 类型 / 字幕 | 帧编号、时间及选择原因 |', '| --- | --- | --- |']
        for row in data.get('timeline') or []:
            fs = '; '.join(f"#{f['index']} {stamp(f['t'])} ({', '.join(map(str, f.get('reasons') or []))})" for f in row.get('frames', []))
            lines.append(f"| {stamp(row.get('start'))}–{stamp(row.get('end'))} | {cell(row.get('text') if row.get('type') == 'speech' else '未检出语音')} | {cell(fs)} |")
    return '\n'.join(lines) + '\n'


# 输入子进程参数与超时；返回标准输出，失败或超时仅报告脱敏后的 stderr 尾部，避免泄露诊断中的凭据。
def command(args, timeout=600):
    prog = Path(str(args[0])).name
    try:
        return subprocess.run(
            [str(x) for x in args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
        ).stdout
    except subprocess.TimeoutExpired as e:
        tail = safe_text((e.stderr or '')[-500:]).strip() if e.stderr else ''
        msg = f'{prog} timed out after {timeout}s'
        if tail:
            msg += f': {tail}'
        raise RuntimeError(msg) from None
    except subprocess.CalledProcessError as e:
        tail = safe_text((e.stderr or '')[-500:]).strip() if e.stderr else ''
        msg = f'{prog} failed (exit {e.returncode})'
        if tail:
            msg += f': {tail}'
        raise RuntimeError(msg) from None


def _find_media_tool(name, cli_override=None):
    if cli_override:
        p = Path(cli_override)
        if not p.is_file():
            raise RuntimeError(f'[preflight] Configured {name} not found: {cli_override}')
        return str(p)
    env_key = f'DOUYIN_MCP_{name.upper()}'
    env_val = os.environ.get(env_key)
    if env_val is not None:
        if not env_val.strip() or not Path(env_val).is_file():
            raise RuntimeError(f'[preflight] {env_key} points to missing executable: {env_val}')
        return env_val
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f'[preflight] Required tool not found in PATH: {name}')


def _sha256_file(path, chunk_size=1048576):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def _dir_size_bytes(path):
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


# 输入视频路径；返回 ffprobe 解析的时长、尺寸、帧率、编码、音轨、音视频起点、大小及 SHA256。
def probe(path, ffprobe_bin=None):
    p = json.loads(command([ffprobe_bin or _find_media_tool('ffprobe'), '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path], timeout=60))
    streams = p.get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    if not video: raise RuntimeError('[probe] No video stream found')
    audio = next((s for s in streams if s.get('codec_type') == 'audio'), None)
    try: fps = float(Fraction(video.get('avg_frame_rate', '0/1')))
    except (ValueError, ZeroDivisionError): fps = None

    def _opt_float(val, default=0.0):
        try:
            f = float(val)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    def _opt_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    v_start = round(_opt_float(video.get('start_time'), 0.0), 6)
    a_start = round(_opt_float(audio.get('start_time'), 0.0), 6) if audio else None
    av_offset = round(a_start - v_start, 6) if (audio and a_start is not None) else None
    return {
        'duration': float(p.get('format', {}).get('duration') or video.get('duration') or 0),
        'width': video.get('width'),
        'height': video.get('height'),
        'fps': fps,
        'video_codec': video.get('codec_name'),
        'video_start_time': v_start,
        'has_audio': bool(audio),
        'audio_codec': audio.get('codec_name') if audio else None,
        'audio_start_time': a_start,
        'av_start_offset': av_offset,
        'audio_channels': _opt_int(audio.get('channels')) if audio else None,
        'audio_sample_rate': _opt_int(audio.get('sample_rate')) if audio else None,
        'size_bytes': path.stat().st_size,
        'sha256': _sha256_file(path),
    }


def _paste_tile(sheet, frame_file, x, y, cell_w, img_h, resample, pad=(4, 2), stage='render', frame_idx=None):
    from PIL import Image
    p = Path(frame_file)
    idx_note = f' (frame #{frame_idx})' if frame_idx is not None else ''
    if not p.is_file():
        raise RuntimeError(f'[{stage}] Missing frame image: {p.name}{idx_note}')
    try:
        with Image.open(p) as im:
            im.load()
            im.thumbnail((cell_w - pad[0], img_h - pad[1]), resample)
            sheet.paste(im, (x + (cell_w - im.width) // 2, y + (img_h - im.height) // 2))
    except Exception as e:
        raise RuntimeError(f'[{stage}] Failed to read frame image {p.name}{idx_note}: {safe_text(e)}') from None


def _save_jpeg(sheet, target, stage, **kwargs):
    try:
        sheet.save(target, format='JPEG', **kwargs)
    except Exception as e:
        raise RuntimeError(f'[{stage}] Failed to write {target.name}: {safe_text(e)}') from None
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f'[{stage}] Failed to write non-empty {target.name}')


# 生成单张高清核心分镜板 storyboard.jpg（自适应横屏 400×226 或竖屏 340×560，LANCZOS 重采样 + 4:4:4 全色度无损色彩）。
def make_storyboard(storyboard_items, directory, output, media=None):
    if not storyboard_items:
        return None
    from PIL import Image, ImageDraw
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    w_src = (media or {}).get('width') or 16
    h_src = (media or {}).get('height') or 9
    is_vertical = h_src > w_src * 1.15
    cols = 4
    cell_w, img_h, label_h = (340, 560, 52) if is_vertical else (400, 226, 48)
    cell_h = img_h + label_h
    rows = (len(storyboard_items) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), '#121212')
    draw = ImageDraw.Draw(sheet)
    font_main = load_font(13)
    font_sub = load_font(12)
    pasted = 0
    for i, f in enumerate(storyboard_items):
        frame_file = Path(directory) / f"frame_{f['frame_index']:08d}.jpg"
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        _paste_tile(sheet, frame_file, x, y, cell_w, img_h, resample, pad=(4, 4), stage='storyboard', frame_idx=f.get('index'))
        pasted += 1
        reason = str(f.get('highlight_reason') or '')
        title_line = f"★#{f['index']}  {stamp(f['t'])}  [{reason[:15]}]"
        ocr_str = ' / '.join(f.get('ocr_texts') or [])
        trans_str = str(f.get('transition_desc') or '').strip()
        vis = f.get('visual') or {}
        vis_hint = format_lighting_tag(vis['luma'], vis['contrast'], vis['sat'], vis['warm']) if vis else ''
        if ocr_str:
            sub_line, sub_color = f"画面字: {ocr_str}", '#b8e994'
        elif trans_str and trans_str not in ('开场画面', '镜头延续'):
            sub_line, sub_color = f"变化: {trans_str}", '#7ec8e3'
        elif vis_hint:
            sub_line, sub_color = f"光影: {vis_hint}", '#cccccc'
        else:
            sub_line, sub_color = f"触发: {', '.join(f.get('reasons') or [])}", '#cccccc'
        draw.rectangle([x, y + img_h, x + cell_w - 1, y + cell_h - 1], fill='#1e1e1e')
        draw.text((x + 6, y + img_h + 3), title_line[:31], fill='#ffd166', font=font_main)
        draw.text((x + 6, y + img_h + 25), sub_line[:28 if is_vertical else 32], fill=sub_color, font=font_sub)
    if pasted != len(storyboard_items):
        raise RuntimeError(f'[storyboard] Storyboard tile mismatch: expected {len(storyboard_items)}, wrote {pasted}')
    target = Path(output) / 'storyboard.jpg'
    _save_jpeg(sheet, target, 'storyboard', quality=92, subsampling=0, optimize=True)
    return target


# 生成关键转场与高动态动作连拍专板 transitions.jpg（每行 1 组 4 帧连拍条，完整展现切镜前后与密帧动作演变）。
def make_transitions_sheet(strips, directory, output, media=None):
    if not strips:
        return None
    from PIL import Image, ImageDraw
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    w_src = (media or {}).get('width') or 16
    h_src = (media or {}).get('height') or 9
    is_vertical = h_src > w_src * 1.15
    cols = 4
    cell_w, img_h, cap_h, hdr_h = (270, 440, 22, 34) if is_vertical else (360, 202, 22, 34)
    row_h = hdr_h + img_h + cap_h
    sheet = Image.new('RGB', (cols * cell_w, len(strips) * row_h), '#141414')
    draw = ImageDraw.Draw(sheet)
    font_hdr = load_font(14)
    font_cap = load_font(12)
    expected_tiles = 0
    pasted = 0
    for row_idx, st in enumerate(strips):
        y0 = row_idx * row_h
        draw.rectangle([0, y0, cols * cell_w - 1, y0 + hdr_h - 1], fill='#1f2833')
        hdr_text = f"第{st['id']}组 [{st['kind']}] {stamp(st['t0'])}–{stamp(st['t1'])}  |  {st['summary']}"
        draw.text((10, y0 + 8), hdr_text[:86 if not is_vertical else 66], fill='#7ec8e3' if '高动态' in st['kind'] else '#ffd166', font=font_hdr)
        row_frames = st['frames'][:cols]
        expected_tiles += len(row_frames)
        for col_idx, f in enumerate(row_frames):
            x = col_idx * cell_w
            y_img = y0 + hdr_h
            frame_path = Path(directory) / f"frame_{f['frame_index']:08d}.jpg"
            _paste_tile(sheet, frame_path, x, y_img, cell_w, img_h, resample, pad=(4, 2), stage='transitions', frame_idx=f.get('index'))
            pasted += 1
            rs = ','.join(r.replace('_motion', '').replace('_change', '') for r in (f.get('reasons') or [])[:2])
            draw.rectangle([x, y_img + img_h, x + cell_w - 1, y0 + row_h - 1], fill='#1b1b1b')
            draw.text((x + 6, y_img + img_h + 3), f"#{f['index']} {stamp(f['t'])} [{rs}]", fill='#eaeaea', font=font_cap)
    if pasted != expected_tiles:
        raise RuntimeError(f'[transitions] Transition sheet tile mismatch: expected {expected_tiles}, wrote {pasted}')
    target = Path(output) / 'transitions.jpg'
    _save_jpeg(sheet, target, 'transitions', quality=88, subsampling=0, optimize=True)
    return target


# 输入选帧、帧目录和输出目录；100% 保留全部动态抽帧（零丢帧），返回 (拼图路径列表, {帧索引: 文件名})。
def make_sheets(items, directory, output, media=None):
    from PIL import Image, ImageDraw
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    w_src = (media or {}).get('width') or 16
    h_src = (media or {}).get('height') or 9
    is_vertical = h_src > w_src * 1.15
    cols = 5 if is_vertical else 4
    per_sheet = 20 if is_vertical else 32
    cell_w, img_h, label_h = (240, 426, 24) if is_vertical else (320, 180, 24)
    cell_h = img_h + label_h
    font = load_font(12)

    if not items:
        return [], {}
    paths = []
    sheet_map = {}
    for offset in range(0, len(items), per_sheet):
        chunk = items[offset:offset + per_sheet]
        sheet_name = f'timeline_{offset // per_sheet + 1:03d}.jpg'
        sheet = Image.new('RGB', (cols * cell_w, max(1, (len(chunk) + cols - 1) // cols) * cell_h), '#181818')
        draw = ImageDraw.Draw(sheet)
        pasted = 0
        for i, f in enumerate(chunk):
            if f['index'] in sheet_map:
                raise RuntimeError(f"[sheets] Duplicate frame index #{f['index']} in timeline items")
            sheet_map[f['index']] = sheet_name
            frame_path = Path(directory) / f"frame_{f['frame_index']:08d}.jpg"
            x, y = (i % cols) * cell_w, (i // cols) * cell_h
            _paste_tile(sheet, frame_path, x, y, cell_w, img_h, resample, pad=(4, 2), stage='sheets', frame_idx=f.get('index'))
            pasted += 1
            rs = ','.join(r.replace('_motion', '').replace('_change', '') for r in (f.get('reasons') or [])[:2])
            draw.text((x + 6, y + img_h + 2), f"#{f['index']} {stamp(f['t'])} [{rs}]", fill='#f0f0f0', font=font)
        if pasted != len(chunk):
            raise RuntimeError(f'[sheets] Timeline sheet {sheet_name} tile mismatch: expected {len(chunk)}, wrote {pasted}')
        target = Path(output) / sheet_name
        _save_jpeg(sheet, target, 'sheets', quality=85, subsampling=0, optimize=True)
        if target.stat().st_size > 2_500_000:
            for quality, sub in ((78, 0), (72, 1)):
                _save_jpeg(sheet, target, 'sheets', quality=quality, subsampling=sub, optimize=True)
                if target.stat().st_size <= 2_500_000:
                    break
        paths.append(target)
    if len(sheet_map) != len(items):
        raise RuntimeError(f'[sheets] Timeline mapping mismatch: expected {len(items)}, mapped {len(sheet_map)}')
    return paths, sheet_map


# 解析时间戳参数（如 "50"、"00:50"、"01:05.5"）为有限非负秒数。
def parse_time_token(tok):
    raw_parts = str(tok).strip().split(':')
    if not raw_parts or len(raw_parts) > 3:
        raise ValueError(f'Invalid time token: {tok}')
    parts = [float(x) for x in raw_parts]
    if any((not math.isfinite(x) or x < 0) for x in parts):
        raise ValueError(f'Invalid non-finite or negative time token: {tok}')
    if len(parts) == 1:
        total = parts[0]
    elif len(parts) == 2:
        total = parts[0] * 60.0 + parts[1]
    else:
        total = parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    if not math.isfinite(total) or total < 0:
        raise ValueError(f'Invalid time token: {tok}')
    return total


# 解析 --clips 区间列表（如 "00:50-01:05,02:19-02:30"），返回 [(t0, t1), ...]。
def parse_clip_ranges(spec):
    if not spec:
        return []
    ranges = []
    for raw in str(spec).split(','):
        part = raw.strip()
        if not part:
            continue
        if '-' not in part:
            raise ValueError(f'Invalid clip range (expected start-end): {part}')
        a_s, b_s = part.split('-', 1)
        t0, t1 = parse_time_token(a_s), parse_time_token(b_s)
        if not math.isfinite(t0) or not math.isfinite(t1) or t0 < 0 or t1 <= t0:
            raise ValueError(f'Clip end must exceed start with finite values: {part}')
        ranges.append((t0, t1))
    return ranges


CLIP_TARGET_FPS = 10.0
CLIP_STEP_SECONDS = 0.1
CLIP_BOUNDARY_TOLERANCE = 0.05


def _probe_video_pts(video_path, ffprobe_bin=None):
    tool = ffprobe_bin or _find_media_tool('ffprobe')
    try:
        raw = command([
            tool, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time',
            '-of', 'json', str(video_path)
        ], timeout=120)
        frames = json.loads(raw).get('frames') or []
    except Exception as e:
        raise RuntimeError(f'[clips] Failed to probe video PTS for {Path(video_path).name}: {safe_text(e)}') from None
    pts = []
    for idx, f in enumerate(frames):
        val = None
        for key in ('best_effort_timestamp_time', 'pkt_pts_time', 'pkt_dts_time'):
            raw_v = f.get(key)
            if raw_v not in (None, 'N/A', ''):
                try:
                    cand = float(raw_v)
                    if math.isfinite(cand):
                        val = cand
                        break
                except (TypeError, ValueError):
                    pass
        if val is None:
            raise RuntimeError(f'[clips] Missing finite PTS on frame {idx} of {Path(video_path).name}')
        pts.append(val)
    if not pts:
        raise RuntimeError(f'[clips] No decoded frame PTS found in {Path(video_path).name}')
    if any(y < x for x, y in zip(pts, pts[1:])):
        raise RuntimeError(f'[clips] Non-monotonic PTS sequence in {Path(video_path).name}')
    start = pts[0]
    return [t - start for t in pts], 'ffprobe_pts'


def _clip_coverage_metrics(frames_list, t0, eff_end):
    ts = sorted(float(cf['t']) for cf in frames_list if t0 - CLIP_BOUNDARY_TOLERANCE <= float(cf['t']) <= eff_end + CLIP_BOUNDARY_TOLERANCE)
    if not ts:
        return (0, float('inf'), 0.0, 0)
    bins = {int(max(0.0, t - t0) / CLIP_STEP_SECONDS + 1e-6) for t in ts}
    gaps = [max(0.0, ts[0] - t0), max(0.0, eff_end - ts[-1])] + [b - a for a, b in zip(ts, ts[1:])]
    max_gap = max(gaps) if gaps else 0.0
    span_cov = ts[-1] - ts[0] if len(ts) >= 2 else 0.0
    return (len(bins), max_gap, span_cov, len(ts))


def _coverage_sort_key(metrics_tuple):
    bins_cov, max_gap, span_cov, count = metrics_tuple
    inv_gap = -round(max_gap, 4) if math.isfinite(max_gap) else float('-inf')
    return (bins_cov, inv_gap, round(span_cov, 4), count)


def _clip_meets_10fps_target(metrics_tuple, t0, eff_end, src_fps=None):
    bins_cov, max_gap, _, count = metrics_tuple
    if count == 0:
        return False
    span = max(0.0, eff_end - t0)
    if span <= CLIP_STEP_SECONDS:
        return count >= 1
    if src_fps is not None and 0 < float(src_fps) < CLIP_TARGET_FPS - 0.05:
        expected_bins = max(1, math.floor(span * float(src_fps)))
        allowed_gap = (1.0 / float(src_fps)) * 1.25 + 0.02
        return bins_cov >= expected_bins and max_gap <= allowed_gap
    expected_bins = max(1, math.ceil(span / CLIP_STEP_SECONDS - 1e-6))
    return bins_cov >= expected_bins and max_gap <= 0.15


def _extract_clip_frames_from_video(video_path, t0, t1, thumb_w, thumb_h, resample, pts=None, label_prefix='源帧'):
    import cv2
    from PIL import Image
    if not pts:
        raise RuntimeError(f'[clips] Missing PTS list for {Path(video_path).name}')
    keep_indices = []
    keep_times = {}
    next_t = None
    for idx, t_val in enumerate(pts):
        if t_val < t0 - CLIP_BOUNDARY_TOLERANCE:
            continue
        if t_val > t1 + CLIP_BOUNDARY_TOLERANCE:
            break
        if next_t is None:
            next_t = t_val
        if t_val + 1e-7 >= next_t:
            keep_indices.append(idx)
            keep_times[idx] = t_val
            while next_t <= t_val + 1e-7:
                next_t += CLIP_STEP_SECONDS
    if not keep_indices:
        return [], {'decode_ended_early': False, 'expected_count': 0, 'decoded_count': 0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'[clips] Failed to open video {Path(video_path).name}')
    extracted = []
    decode_ended_early = False
    try:
        keep_set = set(keep_indices)
        last_idx = keep_indices[-1]
        for cur in range(last_idx + 1):
            if not cap.grab():
                decode_ended_early = True
                break
            if cur in keep_set:
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    decode_ended_early = True
                    continue
                t_val = keep_times[cur]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                im.thumbnail((thumb_w, thumb_h), resample)
                extracted.append({'t': t_val, 'label': f"{label_prefix}{cur} {stamp(t_val)}", 'path': None, 'pil': im})
    finally:
        cap.release()
    return extracted, {'decode_ended_early': decode_ended_early, 'expected_count': len(keep_indices), 'decoded_count': len(extracted)}


def _close_clip_pils(frames_list):
    for cf in frames_list:
        pil = cf.get('pil')
        if pil is not None:
            pil.close()
            cf['pil'] = None


# 清理前按需导出指定时间区间的局部细查拼图（按 10fps 时间网格规则与覆盖质量择优，记录请求范围、首尾 PTS、时间来源、画质来源与覆盖说明）。
def export_clips(
    clip_ranges,
    items,
    frame_dir,
    video_path,
    output,
    media=None,
    warnings=None,
    ffprobe_bin=None,
    max_clip_frames=1200,
    *,
    start_id=1,
    pts_override=None,
    pts_source_override=None,
    source_kind='original',
    proxy_meta=None,
    existing_clips_dir=None,
    appended=False,
    append_batch_id=0,
):
    if not clip_ranges:
        return []
    from PIL import Image, ImageDraw
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    w_src = (media or {}).get('width') or 16
    h_src = (media or {}).get('height') or 9
    duration = float((media or {}).get('duration') or 0.0)
    src_fps = (media or {}).get('fps')
    is_vertical = h_src > w_src * 1.15
    cols = 5 if is_vertical else 4
    per_sheet = 40
    cell_w, img_h, label_h = (260, 460, 24) if is_vertical else (360, 202, 24)
    cell_h = img_h + label_h
    font = load_font(12)
    is_proxy = (source_kind == 'proxy')
    proxy_downsampled = bool((proxy_meta or {}).get('downsampled')) if is_proxy else False
    header_h = 34 if is_proxy else 0
    header_font = load_font(13) if is_proxy else None
    clips_dir = Path(output) / 'clips'
    clips_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    pts_cache = pts_override
    pts_source = pts_source_override or ('proxy_ffprobe_pts' if is_proxy else None)
    label_prefix = '代理帧' if is_proxy else '源帧'

    if is_proxy:
        p_gen = (proxy_meta or {}).get('generation_params') or {}
        p_fps = float((proxy_meta or {}).get('fps') or src_fps or 0.0)
        if proxy_downsampled:
            source_quality_label = f"降帧低码率代理抽帧 ({w_src}x{h_src}, {p_fps:.2f}fps, 不可补回原片缺失动作)"
        else:
            source_quality_label = f"低码率代理抽帧 ({w_src}x{h_src}, {p_gen.get('video_codec') or 'h264'} crf={p_gen.get('crf', '—')}, 非原片)"
    else:
        codec_name = (media or {}).get('video_codec') or 'unknown'
        prefix_zh = '补传原片直抽' if source_kind == 'original_reprovided' else ('留存原片直抽' if source_kind == 'preserved_media' else '原片直抽')
        source_quality_label = f"{prefix_zh} ({w_src}x{h_src}, {codec_name})"

    for offset_idx, (t0, t1) in enumerate(clip_ranges):
        c_idx = int(start_id) + offset_idx
        if duration > 0 and t0 >= duration:
            if warnings is not None:
                warnings.append(f'Clip {c_idx} ({stamp(t0)}–{stamp(t1)}) 超出视频总时长 {duration:.2f}s。')
            entry_oor = {
                'id': c_idx,
                'requested_start': round(t0, 3),
                'requested_end': round(t1, 3),
                'start': round(t0, 2),
                'end': round(t1, 2),
                'actual_first_pts': None,
                'actual_last_pts': None,
                'frame_count': 0,
                'time_source': 'out_of_range',
                'sampling_rule': '10fps_0.1s_grid_no_duplicate',
                'boundary_tolerance_seconds': CLIP_BOUNDARY_TOLERANCE,
                'selection_basis': 'out_of_range',
                'coverage_insufficient_reasons': ['out_of_video_duration'],
                'source_kind': source_kind,
                'source_quality_label': source_quality_label,
                'can_recover_original_motion': False if proxy_downsampled else True,
                'appended': bool(appended),
                'append_batch_id': int(append_batch_id),
                'image': None,
                'images': [],
            }
            if is_proxy:
                entry_oor['proxy_downsampled'] = proxy_downsampled
            exported.append(entry_oor)
            continue

        eff_end = min(t1, duration) if duration > 0 else t1
        reasons_insufficient = []
        if duration > 0 and t1 > duration + 1e-3:
            reasons_insufficient.append('clamped_to_video_end')
        if src_fps is not None and 0 < float(src_fps) < CLIP_TARGET_FPS - 0.05:
            reasons_insufficient.append('source_fps_below_10fps')
        if proxy_downsampled:
            reasons_insufficient.append('proxy_downsampled_cannot_recover_original_motion')
            if warnings is not None:
                warnings.append(f'Clip {c_idx} ({stamp(t0)}–{stamp(eff_end)}) 基于降帧代理抽帧，代理已丢弃部分原片帧，不得承诺补回原片动作信息。')

        dyn_frames = []
        if frame_dir is not None and items:
            in_range = [f for f in items if t0 - CLIP_BOUNDARY_TOLERANCE <= f['t'] <= eff_end + CLIP_BOUNDARY_TOLERANCE]
            for f in in_range:
                p = Path(frame_dir) / f"frame_{f['frame_index']:08d}.jpg"
                if not p.is_file():
                    raise RuntimeError(f"[clips] Missing frame image for clip {c_idx}: {p.name} (frame #{f.get('index')})")
                dyn_frames.append({'t': float(f['t']), 'label': f"#{f['index']} {stamp(f['t'])}", 'path': p, 'pil': None})

        dyn_metrics = _clip_coverage_metrics(dyn_frames, t0, eff_end)
        chosen = dyn_frames
        time_source = 'dynamic_frames_pts'
        supplemented = False
        ext_meta = {}

        if is_proxy or not _clip_meets_10fps_target(dyn_metrics, t0, eff_end, src_fps):
            if video_path and Path(video_path).is_file():
                if pts_cache is None:
                    pts_cache, probed_src = _probe_video_pts(video_path, ffprobe_bin=ffprobe_bin)
                    if not pts_source:
                        pts_source = probed_src
                extracted, ext_meta = _extract_clip_frames_from_video(
                    video_path, t0, eff_end, cell_w - 4, img_h - 2, resample, pts=pts_cache, label_prefix=label_prefix
                )
                if ext_meta.get('decode_ended_early'):
                    reasons_insufficient.append('decode_ended_early')
                    if warnings is not None:
                        warnings.append(f'Clip {c_idx} ({stamp(t0)}–{stamp(eff_end)}) 源视频解码提前结束（期望 {ext_meta.get("expected_count")} 帧，实得 {ext_meta.get("decoded_count")} 帧）。')
                ext_metrics = _clip_coverage_metrics(extracted, t0, eff_end)
                if is_proxy or _coverage_sort_key(ext_metrics) > _coverage_sort_key(dyn_metrics):
                    _close_clip_pils(chosen)
                    chosen = extracted
                    time_source = pts_source or ('proxy_ffprobe_pts' if is_proxy else 'ffprobe_pts')
                    supplemented = True
                else:
                    _close_clip_pils(extracted)
            else:
                reasons_insufficient.append('source_video_unavailable_for_supplement')

        if not chosen:
            _close_clip_pils(chosen)
            raise RuntimeError(f'[clips] No frames available or decoded for clip {c_idx} ({stamp(t0)}–{stamp(t1)})')

        if max_clip_frames > 0 and len(chosen) > max_clip_frames:
            _close_clip_pils(chosen)
            raise RuntimeError(f'[clips] Clip {c_idx} frame count ({len(chosen)}) exceeds budget ({max_clip_frames})')

        final_metrics = _clip_coverage_metrics(chosen, t0, eff_end)
        if not _clip_meets_10fps_target(final_metrics, t0, eff_end, src_fps) and 'source_fps_below_10fps' not in reasons_insufficient and 'decode_ended_early' not in reasons_insufficient:
            reasons_insufficient.append('sparse_coverage_gap')

        actual_first_pts = round(min(cf['t'] for cf in chosen), 3)
        actual_last_pts = round(max(cf['t'] for cf in chosen), 3)
        page_count = (len(chosen) + per_sheet - 1) // per_sheet
        rel_images = []
        for page_idx, offset in enumerate(range(0, len(chosen), per_sheet), 1):
            chunk = chosen[offset:offset + per_sheet]
            rows = (len(chunk) + cols - 1) // cols
            sheet = Image.new('RGB', (cols * cell_w, header_h + rows * cell_h), '#151515')
            draw = ImageDraw.Draw(sheet)
            if is_proxy:
                bar_bg = '#331c0b' if proxy_downsampled else '#0d262b'
                bar_fg = '#ffd166' if proxy_downsampled else '#80ffdb'
                draw.rectangle([0, 0, cols * cell_w, header_h - 2], fill=bar_bg)
                p_gen = (proxy_meta or {}).get('generation_params') or {}
                p_fps_val = float((proxy_meta or {}).get('fps') or src_fps or 0.0)
                if proxy_downsampled:
                    banner_txt = f"[降帧代理抽帧 | 非原片·不可补回原片动作 | {w_src}x{h_src} {p_fps_val:.2f}fps | {stamp(t0)}–{stamp(eff_end)} | 第{page_idx}/{page_count}页]"
                else:
                    banner_txt = f"[低码率代理抽帧 | 非原片·保留源时间轴 | {w_src}x{h_src} {p_fps_val:.2f}fps CRF{p_gen.get('crf', '—')} | {stamp(t0)}–{stamp(eff_end)} | 第{page_idx}/{page_count}页]"
                draw.text((8, 8), banner_txt, fill=bar_fg, font=header_font)
            pasted = 0
            for i, cf in enumerate(chunk):
                x, y = (i % cols) * cell_w, header_h + (i // cols) * cell_h
                if cf['pil'] is not None:
                    im = cf['pil']
                    try:
                        sheet.paste(im, (x + (cell_w - im.width) // 2, y + (img_h - im.height) // 2))
                    finally:
                        im.close()
                        cf['pil'] = None
                else:
                    _paste_tile(sheet, cf['path'], x, y, cell_w, img_h, resample, pad=(4, 2), stage=f'clips[{c_idx}]')
                pasted += 1
                draw.text((x + 6, y + img_h + 2), cf['label'], fill='#9be7ff', font=font)
            if pasted != len(chunk):
                raise RuntimeError(f'[clips] Clip {c_idx} tile mismatch: expected {len(chunk)}, wrote {pasted}')
            if page_count == 1:
                name = f"clip_{c_idx:02d}_{int(t0)}_{int(t1)}.jpg"
            else:
                name = f"clip_{c_idx:02d}_{int(t0)}_{int(t1)}_{page_idx:03d}.jpg"
            if existing_clips_dir is not None and (Path(existing_clips_dir) / name).exists():
                raise RuntimeError(f'[clips] Target clip filename already exists and cannot be overwritten: {name}')
            target = clips_dir / name
            if target.exists():
                raise RuntimeError(f'[clips] Staged clip filename collision: {name}')
            _save_jpeg(sheet, target, f'clips[{c_idx}]', quality=90, subsampling=0, optimize=True)
            rel_images.append(f'clips/{name}')
        entry = {
            'id': c_idx,
            'requested_start': round(t0, 3),
            'requested_end': round(t1, 3),
            'start': round(t0, 2),
            'end': round(eff_end, 2),
            'actual_first_pts': actual_first_pts,
            'actual_last_pts': actual_last_pts,
            'frame_count': len(chosen),
            'time_source': time_source,
            'sampling_rule': '10fps_0.1s_grid_no_duplicate',
            'boundary_tolerance_seconds': CLIP_BOUNDARY_TOLERANCE,
            'supplemented': supplemented,
            'selection_basis': 'coverage_quality',
            'coverage_insufficient_reasons': reasons_insufficient,
            'source_kind': source_kind,
            'source_quality_label': source_quality_label,
            'can_recover_original_motion': not proxy_downsampled,
            'appended': bool(appended),
            'append_batch_id': int(append_batch_id),
            'image': rel_images[0],
        }
        if is_proxy:
            entry['proxy_downsampled'] = proxy_downsampled
        if len(rel_images) > 1:
            entry['images'] = rel_images
        exported.append(entry)
    return exported


# 无输入；在 Windows 注册 NVIDIA DLL 目录并返回需保活的句柄列表。
def add_cuda_paths():
    if os.name != 'nt': return []
    dirs = []
    for site in map(Path, sys.path):
        nvidia = site / 'nvidia'
        if nvidia.is_dir():
            for p in nvidia.glob('*/bin'):
                if p.is_dir():
                    dirs.append(os.add_dll_directory(str(p)))
                    os.environ['PATH'] = str(p) + os.pathsep + os.environ.get('PATH', '')
    return dirs


# 读取单声道或多声道 PCM WAV 文件并统一归一化到 [-1.0, 1.0] 单声道 float64 数组。
def read_wav_pcm(wav_path):
    import wave
    import numpy as np
    with wave.open(str(wav_path), 'rb') as w:
        channels = int(w.getnchannels())
        sampwidth = int(w.getsampwidth())
        sample_rate = int(w.getframerate())
        raw = w.readframes(w.getnframes())
    if not raw or channels <= 0 or sample_rate <= 0:
        return {'samples': np.zeros(0, dtype=np.float64), 'sample_rate': sample_rate, 'channels': channels, 'downmixed_to_mono': channels > 1}
    if sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        signed = np.where(signed >= 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float64) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise RuntimeError(f'[audio_energy] Unsupported WAV sample width: {sampwidth} bytes')
    if channels > 1:
        usable = (arr.size // channels) * channels
        arr = arr[:usable].reshape(-1, channels)
        mono = arr.mean(axis=1)
    else:
        mono = arr.reshape(-1)
    return {'samples': mono, 'sample_rate': sample_rate, 'channels': channels, 'downmixed_to_mono': channels > 1}


def _amp_to_dbfs(amp):
    return round(20.0 * math.log10(max(float(amp), 1e-6)), 2)


# 计算音频客观能量指标：支持立体声下混单声道、静音/已知幅值/短脉冲、末尾不足窗口、空区间及音视频起点偏移。
# 仅输出客观声压级统计与区间变化，不推断是否为平稳配乐或无人声。
def compute_audio_energy(samples, sample_rate, total_duration=None, audio_offset=0.0, window_seconds=0.5, silence_threshold_dbfs=-45.0, surge_delta_db=12.0, surge_min_dbfs=-35.0, subframe_ms=50.0, source_channels=None):
    import numpy as np
    if sample_rate is None or float(sample_rate) <= 0:
        return {'status': 'empty_audio', 'has_audio': True, 'error': 'invalid_sample_rate', 'windows': [], 'low_energy_regions': [], 'energy_surges': []}
    sr = float(sample_rate)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2:
        ch = int(arr.shape[1])
        mono = arr.mean(axis=1) if ch > 1 else (arr[:, 0] if ch == 1 else np.zeros(0, dtype=np.float64))
    else:
        mono = arr.reshape(-1)
        ch = int(source_channels or 1)
    downmixed = bool((source_channels and int(source_channels) > 1) or (arr.ndim == 2 and arr.shape[1] > 1))
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    offset = float(audio_offset or 0.0)
    audio_len_sec = len(mono) / sr
    audio_start = offset
    audio_end = offset + audio_len_sec
    span_dur = float(total_duration) if (total_duration is not None and float(total_duration) > 0) else max(0.0, audio_end)
    win_sec = max(0.05, float(window_seconds))

    if span_dur <= 0 or len(mono) == 0:
        return {
            'status': 'empty_audio',
            'has_audio': True,
            'sample_rate': int(round(sr)),
            'source_channels': int(source_channels or ch),
            'downmixed_to_mono': downmixed,
            'av_start_offset_seconds': round(offset, 6),
            'window_seconds': round(win_sec, 4),
            'silence_threshold_dbfs': float(silence_threshold_dbfs),
            'total_duration_seconds': round(max(0.0, span_dur), 4),
            'covered_duration_seconds': 0.0,
            'valid_window_count': 0,
            'empty_window_count': 0,
            'silent_window_count': 0,
            'has_partial_trailing_window': False,
            'rms': None,
            'rms_dbfs': None,
            'peak': None,
            'peak_dbfs': None,
            'window_silence_ratio': None,
            'silent_duration_ratio': None,
            'subframe_silence_ratio': None,
            'low_energy_regions': [],
            'energy_surges': [],
            'windows': [],
        }

    windows = []
    t0 = 0.0
    sub_n_base = max(1, int(round(sr * float(subframe_ms) / 1000.0)))
    covered_chunks = []
    valid_windows = []
    empty_window_count = 0
    silent_window_count = 0
    silent_covered_dur = 0.0
    total_covered_dur = 0.0
    subframe_silent_samples = 0
    subframe_total_samples = 0
    has_partial = False

    while t0 + 1e-9 < span_dur:
        t1 = min(span_dur, t0 + win_sec)
        is_partial = (t1 - t0) < (win_sec - 1e-6)
        if is_partial:
            has_partial = True
        ov0 = max(t0, audio_start)
        ov1 = min(t1, audio_end)
        if ov1 - ov0 <= 1e-9:
            seg = mono[0:0]
        else:
            idx0 = max(0, int(math.floor((ov0 - offset) * sr + 1e-9)))
            idx1 = min(len(mono), int(math.ceil((ov1 - offset) * sr - 1e-9)))
            if idx1 <= idx0 and (ov1 - ov0) * sr >= 0.5 and idx0 < len(mono):
                idx1 = min(len(mono), idx0 + 1)
            seg = mono[idx0:idx1]

        if len(seg) == 0:
            empty_window_count += 1
            windows.append({
                'start': round(t0, 4),
                'end': round(t1, 4),
                'duration': round(t1 - t0, 4),
                'is_partial': is_partial,
                'has_samples': False,
                'empty': True,
                'sample_count': 0,
                'rms': None,
                'rms_dbfs': None,
                'peak': None,
                'peak_dbfs': None,
                'is_silent': None,
                'subframe_silence_ratio': None,
            })
        else:
            covered_chunks.append(seg)
            seg_dur = len(seg) / sr
            total_covered_dur += seg_dur
            w_rms = float(np.sqrt(np.mean(np.square(seg))))
            w_peak = float(np.max(np.abs(seg)))
            w_rms_db = _amp_to_dbfs(w_rms)
            w_peak_db = _amp_to_dbfs(w_peak)
            w_silent = bool(w_rms_db <= silence_threshold_dbfs)
            if w_silent:
                silent_window_count += 1
                silent_covered_dur += seg_dur

            # 窗口内子帧静音占比（细粒度测量短脉冲窗口中的静音比例）
            w_sub_silent = 0
            for pos in range(0, len(seg), sub_n_base):
                sub = seg[pos:pos + sub_n_base]
                sub_rms_db = _amp_to_dbfs(float(np.sqrt(np.mean(np.square(sub)))))
                if sub_rms_db <= silence_threshold_dbfs:
                    w_sub_silent += len(sub)
            subframe_silent_samples += w_sub_silent
            subframe_total_samples += len(seg)
            w_sub_ratio = round(w_sub_silent / len(seg), 4)

            w_entry = {
                'start': round(t0, 4),
                'end': round(t1, 4),
                'duration': round(t1 - t0, 4),
                'is_partial': is_partial,
                'has_samples': True,
                'empty': False,
                'sample_count': int(len(seg)),
                'rms': round(w_rms, 6),
                'rms_dbfs': w_rms_db,
                'peak': round(w_peak, 6),
                'peak_dbfs': w_peak_db,
                'is_silent': w_silent,
                'subframe_silence_ratio': w_sub_ratio,
            }
            windows.append(w_entry)
            valid_windows.append(w_entry)
        t0 = t1

    if not valid_windows:
        return {
            'status': 'empty_audio',
            'has_audio': True,
            'sample_rate': int(round(sr)),
            'source_channels': int(source_channels or ch),
            'downmixed_to_mono': downmixed,
            'av_start_offset_seconds': round(offset, 6),
            'window_seconds': round(win_sec, 4),
            'silence_threshold_dbfs': float(silence_threshold_dbfs),
            'total_duration_seconds': round(span_dur, 4),
            'covered_duration_seconds': 0.0,
            'valid_window_count': 0,
            'empty_window_count': empty_window_count,
            'silent_window_count': 0,
            'has_partial_trailing_window': has_partial,
            'rms': None,
            'rms_dbfs': None,
            'peak': None,
            'peak_dbfs': None,
            'window_silence_ratio': None,
            'silent_duration_ratio': None,
            'subframe_silence_ratio': None,
            'low_energy_regions': [],
            'energy_surges': [],
            'windows': windows,
        }

    all_cov = np.concatenate(covered_chunks)
    global_rms = float(np.sqrt(np.mean(np.square(all_cov))))
    global_peak = float(np.max(np.abs(all_cov)))

    # 连续低能量区间合并（仅在相邻有效窗口均属低能量/静音时合并，不跨空区间）
    low_energy_regions = []
    cur_reg = None
    for w in windows:
        if w['has_samples'] and w['is_silent']:
            if cur_reg is None:
                cur_reg = {'start': w['start'], 'end': w['end'], '_rms_list': [w['rms_dbfs']], '_peak_list': [w['peak_dbfs']]}
            else:
                cur_reg['end'] = w['end']
                cur_reg['_rms_list'].append(w['rms_dbfs'])
                cur_reg['_peak_list'].append(w['peak_dbfs'])
        else:
            if cur_reg is not None:
                low_energy_regions.append({
                    'start': round(cur_reg['start'], 2),
                    'end': round(cur_reg['end'], 2),
                    'duration': round(cur_reg['end'] - cur_reg['start'], 2),
                    'mean_rms_dbfs': round(sum(cur_reg['_rms_list']) / len(cur_reg['_rms_list']), 2),
                    'max_peak_dbfs': round(max(cur_reg['_peak_list']), 2),
                })
                cur_reg = None
    if cur_reg is not None:
        low_energy_regions.append({
            'start': round(cur_reg['start'], 2),
            'end': round(cur_reg['end'], 2),
            'duration': round(cur_reg['end'] - cur_reg['start'], 2),
            'mean_rms_dbfs': round(sum(cur_reg['_rms_list']) / len(cur_reg['_rms_list']), 2),
            'max_peak_dbfs': round(max(cur_reg['_peak_list']), 2),
        })

    # 能量突增（基于前序有效窗口局部基线或高波峰因子短脉冲）
    energy_surges = []
    for i, w in enumerate(valid_windows):
        prev_vals = sorted(x['rms_dbfs'] for x in valid_windows[max(0, i - 4):i])
        prev_med = prev_vals[len(prev_vals) // 2] if prev_vals else None
        delta_db = round(w['rms_dbfs'] - prev_med, 2) if prev_med is not None else 0.0
        crest_db = round(w['peak_dbfs'] - w['rms_dbfs'], 2)
        if prev_med is not None and delta_db >= surge_delta_db and w['rms_dbfs'] >= surge_min_dbfs:
            energy_surges.append({
                'start': round(w['start'], 2),
                'end': round(w['end'], 2),
                'kind': 'rms_surge',
                'rms_dbfs': w['rms_dbfs'],
                'peak_dbfs': w['peak_dbfs'],
                'delta_db_vs_prev': delta_db,
            })
        elif w['peak_dbfs'] >= -20.0 and crest_db >= 10.0 and (prev_med is None or (w['peak_dbfs'] - prev_med) >= 18.0):
            energy_surges.append({
                'start': round(w['start'], 2),
                'end': round(w['end'], 2),
                'kind': 'transient_pulse',
                'rms_dbfs': w['rms_dbfs'],
                'peak_dbfs': w['peak_dbfs'],
                'delta_db_vs_prev': round(w['peak_dbfs'] - prev_med, 2) if prev_med is not None else crest_db,
            })

    return {
        'status': 'ok',
        'has_audio': True,
        'sample_rate': int(round(sr)),
        'source_channels': int(source_channels or ch),
        'downmixed_to_mono': downmixed,
        'av_start_offset_seconds': round(offset, 6),
        'window_seconds': round(win_sec, 4),
        'silence_threshold_dbfs': float(silence_threshold_dbfs),
        'total_duration_seconds': round(span_dur, 4),
        'covered_duration_seconds': round(total_covered_dur, 4),
        'valid_window_count': len(valid_windows),
        'empty_window_count': empty_window_count,
        'silent_window_count': silent_window_count,
        'has_partial_trailing_window': has_partial,
        'rms': round(global_rms, 6),
        'rms_dbfs': _amp_to_dbfs(global_rms),
        'peak': round(global_peak, 6),
        'peak_dbfs': _amp_to_dbfs(global_peak),
        'window_silence_ratio': round(silent_window_count / len(valid_windows), 4),
        'silent_duration_ratio': round(silent_covered_dur / total_covered_dur, 4) if total_covered_dur > 0 else None,
        'subframe_silence_ratio': round(subframe_silent_samples / subframe_total_samples, 4) if subframe_total_samples > 0 else None,
        'low_energy_regions': low_energy_regions,
        'energy_surges': energy_surges,
        'windows': windows,
    }


def analyze_audio_file(wav_path, media=None, window_seconds=0.5):
    pcm_info = read_wav_pcm(wav_path)
    media = media or {}
    src_ch = media.get('audio_channels') or pcm_info.get('channels') or 1
    dur = media.get('duration')
    offset = float(media.get('av_start_offset') or 0.0)
    return compute_audio_energy(
        pcm_info['samples'],
        pcm_info['sample_rate'],
        total_duration=dur,
        audio_offset=offset,
        window_seconds=window_seconds,
        source_channels=src_ch,
    )


def summarize_interval_audio_energy(audio_energy, start, end):
    if not isinstance(audio_energy, dict) or audio_energy.get('status') != 'ok':
        return None
    s0, s1 = float(start or 0.0), float(end or 0.0)
    wins = [
        w for w in (audio_energy.get('windows') or [])
        if w.get('has_samples') and w['end'] > s0 + 1e-6 and w['start'] < s1 - 1e-6
    ]
    if not wins:
        return {'has_samples': False, 'rms_dbfs': None, 'peak_dbfs': None, 'window_silence_ratio': None}
    sq_sum = 0.0
    n_sum = 0
    for w in wins:
        n = int(w.get('sample_count') or 0)
        if n > 0 and w.get('rms') is not None:
            sq_sum += (float(w['rms']) ** 2) * n
            n_sum += n
    mean_rms = math.sqrt(sq_sum / n_sum) if n_sum > 0 else 0.0
    peak_val = max(float(w.get('peak') or 0.0) for w in wins)
    sil_cnt = sum(1 for w in wins if w.get('is_silent'))
    return {
        'has_samples': True,
        'rms_dbfs': _amp_to_dbfs(mean_rms),
        'peak_dbfs': _amp_to_dbfs(peak_val),
        'window_silence_ratio': round(sil_cnt / len(wins), 4),
    }


# 输入 WAV、模型名、语言、设备选择及可追加的警告列表；返回带时间戳的简体中文转写和设备信息。
def transcribe(wav, model, language, requested, warnings):
    handles = add_cuda_paths()  # retain handles for the lifetime of CTranslate2
    from faster_whisper import WhisperModel
    import wave, numpy as np
    model_root = os.environ.get('HF_HOME') or str(ROOT / 'models')
    local = ROOT / 'models' / f'faster-whisper-{model}'
    model_ref = str(local) if (local / 'model.bin').is_file() else model
    with wave.open(str(wav), 'rb') as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    def run(device):
        compute = 'float16' if device == 'cuda' else 'int8'
        engine = WhisperModel(model_ref, device=device, compute_type=compute, download_root=model_root)
        kwargs = dict(language=language, vad_filter=True, beam_size=5, condition_on_previous_text=False)
        if language == 'zh':
            kwargs['initial_prompt'] = '以下是普通话的简体中文句子。'
        it, info = engine.transcribe(pcm, **kwargs)
        hallucination_re = re.compile(r'(优优独播剧场|YoYo\s*Television|Amara\.org|字幕由|提供字幕|感谢观看|谢谢观看|請不吝點贊|请不吝点赞|订阅转发)', re.I)
        segments = []
        filtered_segments = []
        last = 0.0
        for s in it:
            prob = float(getattr(s, 'no_speech_prob', 0.0) or 0.0)
            raw_txt = getattr(s, 'text', '') or ''
            text = to_simplified(safe_text(raw_txt.strip()))
            s_start = round(float(s.start), 2)
            s_end = max(s_start, round(float(s.end), 2))
            if prob > 0.85:
                filtered_segments.append({'start': s_start, 'end': s_end, 'no_speech_prob': round(prob, 4), 'text': text or '', 'reason': 'no_speech_prob_gt_0.85'})
                continue
            if not text:
                filtered_segments.append({'start': s_start, 'end': s_end, 'no_speech_prob': round(prob, 4), 'text': '', 'reason': 'empty_after_sanitize'})
                continue
            if hallucination_re.search(text):
                filtered_segments.append({'start': s_start, 'end': s_end, 'no_speech_prob': round(prob, 4), 'text': text, 'reason': 'hallucination_pattern'})
                continue
            start = max(last, s_start); end = max(start, s_end)
            segments.append({'id': len(segments), 'start': start, 'end': end, 'text': text})
            last = end
        return {'segments': segments, 'filtered_segments': filtered_segments, 'language': info.language, 'language_probability': info.language_probability, 'model': model, 'requested_device': requested, 'device': device, 'compute_type': compute}

    if requested == 'cpu':
        return run('cpu')
    try:
        return run('cuda')
    except (RuntimeError, OSError, ValueError) as exc:
        if requested == 'cuda':
            raise
        warnings.append(f'CUDA 加载或识别失败 ({type(exc).__name__})，已回退 CPU int8。')
        return run('cpu')


def _available_ram_mb():
    if os.name == 'nt':
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullAvailPhys / (1024 * 1024), 1)
        except Exception:
            return None
    meminfo = Path('/proc/meminfo')
    if meminfo.is_file():
        for line in meminfo.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('MemAvailable:'):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return round(int(parts[1]) / 1024.0, 1)
    return None


def _preflight_checks(a, clip_ranges):
    ffmpeg_bin = _find_media_tool('ffmpeg', getattr(a, 'ffmpeg', None))
    ffprobe_bin = _find_media_tool('ffprobe', getattr(a, 'ffprobe', None))
    append_dir = getattr(a, 'append_clips', None)
    route = None
    ytdlp_available = False
    if append_dir:
        rd = Path(append_dir).resolve()
        if not rd.is_dir() or not (rd / 'analysis.json').is_file() or not (rd / 'analysis.md').is_file():
            raise RuntimeError(f'[preflight] Invalid result directory for --append-clips (missing analysis.json or analysis.md): {append_dir}')
        if a.video and not a.video.resolve().is_file():
            raise RuntimeError(f'[preflight] Local video does not exist: {a.video}')
        free_res_mb = round(shutil.disk_usage(rd).free / (1024 * 1024), 1)
    else:
        if a.video:
            vp = a.video.resolve()
            if not vp.is_file():
                raise RuntimeError(f'[preflight] Local video does not exist: {a.video}')
        else:
            try:
                route = classify_url_route(a.url)
            except UnsupportedRouteError as ure:
                raise RuntimeError(str(ure)) from None
            if route['downloader'] == 'douyin_browser_mcp':
                fp = Path(a.fetch_python)
                if not fp.is_file():
                    raise RuntimeError(f'[preflight] Fetch python executable not found: {a.fetch_python}')
                if not (ROOT / 'douyin_fetch.py').is_file():
                    raise RuntimeError('[preflight] douyin_fetch.py missing in project root')
            elif route['downloader'] == 'yt_dlp':
                ytdlp_cmd, ytdlp_err = resolve_ytdlp_runner(getattr(a, 'ytdlp_bin', None))
                if not ytdlp_cmd:
                    raise RuntimeError(
                        f"[preflight] Optional downloader 'yt-dlp' is unavailable for external platform '{route['platform']}': {ytdlp_err}. "
                        "Local files and Douyin links remain unaffected."
                    )
                ytdlp_available = True
        if not (ROOT / 'dynamic_frames.py').is_file():
            raise RuntimeError('[preflight] dynamic_frames.py missing in project root')
        a.results_root.mkdir(parents=True, exist_ok=True)
        free_res_mb = round(shutil.disk_usage(a.results_root).free / (1024 * 1024), 1)
    a.tmp_root.mkdir(parents=True, exist_ok=True)
    free_tmp_mb = round(shutil.disk_usage(a.tmp_root).free / (1024 * 1024), 1)
    required_disk_mb = float(a.min_disk_mb)
    if getattr(a, 'save_proxy', False) and not append_dir:
        max_proxy_mb = float(getattr(a, 'max_proxy_mb', 0.0) or 0.0)
        if max_proxy_mb > 0:
            required_disk_mb += min(max_proxy_mb, 64.0)
    if min(free_res_mb, free_tmp_mb) < required_disk_mb:
        raise RuntimeError(f'[preflight] Insufficient free disk space (results={free_res_mb}MB, tmp={free_tmp_mb}MB < {required_disk_mb}MB)')
    avail_ram = _available_ram_mb()
    if avail_ram is not None and avail_ram < 256.0:
        raise RuntimeError(f'[preflight] Insufficient available RAM ({avail_ram}MB < 256MB)')
    return {
        'ffmpeg': ffmpeg_bin,
        'ffprobe': ffprobe_bin,
        'route': route,
        'ytdlp_available': ytdlp_available,
        'free_results_disk_mb': free_res_mb,
        'free_tmp_disk_mb': free_tmp_mb,
        'available_ram_mb': avail_ram,
    }


def _check_media_budgets(media, clip_ranges, a, avail_ram_mb):
    duration = float((media or {}).get('duration') or 0.0)
    if a.max_duration > 0 and duration > a.max_duration:
        raise RuntimeError(f'[preflight] Video duration ({duration:.2f}s) exceeds budget ({a.max_duration:.2f}s)')
    est_analysis_samples = max(1, int(duration * 10.0) + 16)
    est_mem_mb = round((est_analysis_samples * 480 * 216 + len(clip_ranges) * 40 * 360 * 202 * 3) / (1024 * 1024), 2)
    mem_limit = float(a.max_memory_mb)
    if avail_ram_mb is not None:
        mem_limit = min(mem_limit, float(avail_ram_mb) * 0.7)
    if a.max_memory_mb > 0 and est_mem_mb > mem_limit:
        raise RuntimeError(f'[preflight] Estimated memory ({est_mem_mb}MB) exceeds budget ({mem_limit:.1f}MB)')
    for c_idx, (t0, t1) in enumerate(clip_ranges, 1):
        eff_end = min(t1, duration) if duration > 0 else t1
        if duration > 0 and t0 < duration:
            est_clip_frames = int(max(0.0, eff_end - t0) * CLIP_TARGET_FPS) + 2
            if a.max_clip_frames > 0 and est_clip_frames > a.max_clip_frames:
                raise RuntimeError(f'[preflight] Estimated clip {c_idx} frames ({est_clip_frames}) exceed budget ({a.max_clip_frames})')


def _persist_reports(result_dir, data):
    data_clean = scrub(data)
    json_path = result_dir / 'analysis.json'
    md_path = result_dir / 'analysis.md'
    json_tmp = result_dir / f'.analysis.json.{uuid.uuid4().hex[:8]}.tmp'
    md_tmp = result_dir / f'.analysis.md.{uuid.uuid4().hex[:8]}.tmp'
    json_text = json.dumps(data_clean, ensure_ascii=False, indent=2) + '\n'
    md_text = render_markdown(data_clean)
    try:
        json_tmp.write_text(json_text, encoding='utf-8')
        md_tmp.write_text(md_text, encoding='utf-8')
        if not json_tmp.is_file() or json_tmp.stat().st_size <= 0 or not md_tmp.is_file() or md_tmp.stat().st_size <= 0:
            raise RuntimeError('[persist] Persisted temporary analysis.json or analysis.md is empty')
        json.loads(json_tmp.read_text(encoding='utf-8'))
        os.replace(json_tmp, json_path)
        os.replace(md_tmp, md_path)
    finally:
        for tmp_f in (json_tmp, md_tmp):
            if tmp_f.exists():
                try:
                    tmp_f.unlink()
                except Exception:
                    pass
    if not json_path.is_file() or json_path.stat().st_size <= 0 or not md_path.is_file() or md_path.stat().st_size <= 0:
        raise RuntimeError('[persist] Persisted analysis.json or analysis.md is empty')
    return data_clean


def build_video_proxy(
    video_path,
    tmp_dir,
    result_dir,
    orig_media,
    ffmpeg_bin,
    ffprobe_bin,
    *,
    max_dim=640,
    crf=28,
    proxy_fps=0.0,
    preset='veryfast',
    max_proxy_mb=256.0,
    max_result_mb=1024.0,
    min_disk_mb=512.0,
    source_kind='downloaded_original',
    aweme_id=None,
    source_pts=None,
    source_frame_count=None,
    timeout=300,
):
    max_dim = max(64, int(max_dim))
    crf = min(51, max(0, int(crf)))
    req_fps = float(proxy_fps or 0.0)
    if req_fps < 0 or not math.isfinite(req_fps):
        raise ValueError(f'[proxy] Invalid proxy fps: {proxy_fps}')

    vf = (
        f"scale='if(gte(iw,ih),trunc(min({max_dim},iw)/2)*2,-2)'"
        f":'if(gte(iw,ih),-2,trunc(min({max_dim},ih)/2)*2)'"
    )
    tmp_proxy = Path(tmp_dir) / 'proxy.mp4'
    cmd = [ffmpeg_bin, '-nostdin', '-v', 'error', '-y', '-i', str(video_path), '-vf', vf]
    if req_fps > 0:
        fps_mode = 'cfr'
        cmd += ['-fps_mode', 'cfr', '-r', f'{req_fps:g}']
    else:
        fps_mode = 'passthrough'
        cmd += ['-fps_mode', 'passthrough']
    cmd += ['-c:v', 'libx264', '-crf', str(crf), '-preset', str(preset), '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', str(tmp_proxy)]
    command(cmd, timeout=timeout)
    if not tmp_proxy.is_file() or tmp_proxy.stat().st_size <= 0:
        raise RuntimeError('[proxy] Generated proxy file is missing or empty')

    try:
        proxy_probe = probe(tmp_proxy, ffprobe_bin=ffprobe_bin)
        proxy_pts, _ = _probe_video_pts(tmp_proxy, ffprobe_bin=ffprobe_bin)
        proxy_bytes = int(proxy_probe['size_bytes'])
        proxy_mb = round(proxy_bytes / (1024 * 1024), 3)
        if max_proxy_mb > 0 and proxy_mb > max_proxy_mb:
            raise RuntimeError(f'[proxy] Generated proxy size ({proxy_mb:.3f}MB) exceeds max_proxy_mb budget ({max_proxy_mb}MB)')
        existing_res_bytes = _dir_size_bytes(result_dir)
        total_res_mb = round((existing_res_bytes + proxy_bytes) / (1024 * 1024), 3)
        if max_result_mb > 0 and total_res_mb > max_result_mb:
            raise RuntimeError(f'[proxy] Total result disk usage with proxy ({total_res_mb:.3f}MB) exceeds max_result_mb budget ({max_result_mb}MB)')
        free_res_mb = round(shutil.disk_usage(result_dir).free / (1024 * 1024), 1)
        if free_res_mb < min_disk_mb:
            raise RuntimeError(f'[proxy] Insufficient free disk space after proxy generation ({free_res_mb}MB < {min_disk_mb}MB)')

        if source_pts is None:
            source_pts, _ = _probe_video_pts(video_path, ffprobe_bin=ffprobe_bin)
        orig_fps = float(orig_media.get('fps') or 0.0)
        p_fps = float(proxy_probe.get('fps') or 0.0)
        orig_cnt = int(source_frame_count) if source_frame_count else len(source_pts)
        p_cnt = len(proxy_pts)
        downsampled = bool(
            (req_fps > 0 and orig_fps > 0 and req_fps < orig_fps - 0.05)
            or (orig_cnt > 0 and p_cnt < orig_cnt * 0.95)
        )

        max_abs_pts_diff = None
        if source_pts and proxy_pts:
            if len(proxy_pts) == len(source_pts):
                max_abs_pts_diff = round(max(abs(a - b) for a, b in zip(proxy_pts, source_pts)), 6)
            else:
                diffs = []
                for tp in proxy_pts:
                    pos = bisect.bisect_left(source_pts, tp)
                    cands = []
                    if pos < len(source_pts):
                        cands.append(abs(source_pts[pos] - tp))
                    if pos > 0:
                        cands.append(abs(source_pts[pos - 1] - tp))
                    if cands:
                        diffs.append(min(cands))
                if diffs:
                    max_abs_pts_diff = round(max(diffs), 6)

        orig_frame_interval = round(1.0 / orig_fps, 6) if orig_fps > 0 else None
        within_one_frame = (
            (max_abs_pts_diff <= orig_frame_interval + 1e-4)
            if (max_abs_pts_diff is not None and orig_frame_interval is not None)
            else None
        )
        orig_v_start = float(orig_media.get('video_start_time') or 0.0)
        proxy_v_start = float(proxy_probe.get('video_start_time') or 0.0)
        can_recover_motion = not downsampled
        limitation_note = (
            '降帧低码率代理（代理帧率低于原片，已丢失部分原片帧，不得承诺补回原片完整动作信息）。'
            if downsampled
            else '低码率降分辨率代理（保留源帧时间轴），画面细节/文字清晰度低于原片，不用于替代原片像素细节。'
        )

        proxy_dir = Path(result_dir) / 'proxy'
        proxy_dir.mkdir(parents=True, exist_ok=True)
        dest_proxy = proxy_dir / 'proxy.mp4'
        shutil.move(str(tmp_proxy), str(dest_proxy))
    except Exception:
        if tmp_proxy.exists():
            try:
                tmp_proxy.unlink()
            except Exception:
                pass
        raise

    return {
        'enabled': True,
        'available': True,
        'status': 'ok',
        'error': None,
        'path': 'proxy/proxy.mp4',
        'source': {
            'kind': source_kind,
            'aweme_id': aweme_id,
            'source_sha256': orig_media.get('sha256'),
            'source_size_bytes': orig_media.get('size_bytes'),
            'source_width': orig_media.get('width'),
            'source_height': orig_media.get('height'),
            'source_fps': orig_media.get('fps'),
            'source_duration': orig_media.get('duration'),
            'source_frame_count': orig_cnt,
            'source_video_start_time': orig_v_start,
        },
        'generation_params': {
            'video_codec': 'libx264',
            'crf': crf,
            'preset': str(preset),
            'pix_fmt': 'yuv420p',
            'max_dim': max_dim,
            'requested_fps': req_fps if req_fps > 0 else None,
            'fps_mode': fps_mode,
            'audio_stripped': True,
            'ffmpeg_vf': vf,
        },
        'width': proxy_probe.get('width'),
        'height': proxy_probe.get('height'),
        'fps': p_fps,
        'duration': proxy_probe.get('duration'),
        'frame_count': p_cnt,
        'size_bytes': proxy_bytes,
        'size_mb': proxy_mb,
        'sha256': proxy_probe.get('sha256'),
        'video_start_time': proxy_v_start,
        'downsampled': downsampled,
        'can_recover_original_motion': can_recover_motion,
        'limitation_note': limitation_note,
        'time_mapping': {
            'mode': 'downsampled_pts_mapped' if downsampled else 'passthrough_1to1',
            'source_start_time': orig_v_start,
            'proxy_start_time': proxy_v_start,
            'pts_offset_seconds': round(proxy_v_start - orig_v_start, 6),
            'scale_factor': 1.0,
            'max_abs_pts_diff_seconds': max_abs_pts_diff,
            'source_frame_interval_seconds': orig_frame_interval,
            'within_one_source_frame_interval': within_one_frame,
            'frame_count_ratio': round(p_cnt / orig_cnt, 4) if orig_cnt else None,
        },
    }


def _check_sampled_luma_identity(video_path, items):
    candidates = [it for it in (items or []) if isinstance(it.get('visual'), dict) and it['visual'].get('luma') is not None and it.get('frame_index') is not None]
    if not candidates:
        return
    import cv2
    picks = [candidates[0]]
    if len(candidates) >= 3:
        picks.append(candidates[len(candidates) // 2])
    if len(candidates) >= 2:
        picks.append(candidates[-1])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'[append_clips] Failed to open candidate video for visual verification: {Path(video_path).name}')
    diffs = []
    try:
        for it in picks:
            fi = int(it['frame_index'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            cand_luma = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
            diffs.append(abs(cand_luma - float(it['visual']['luma'])))
    finally:
        cap.release()
    if diffs and min(diffs) > 30.0:
        raise RuntimeError(f'[append_clips] Source identity mismatch: sampled frame luma diverges from recorded visual timeline (min_diff={min(diffs):.1f})')


def validate_clip_source(data, candidate_video, ffprobe_bin=None, expected_kind='original_reprovided'):
    cand_path = Path(candidate_video)
    if not cand_path.is_file():
        raise RuntimeError(f'[append_clips] Candidate source video does not exist: {candidate_video}')
    cand_media = probe(cand_path, ffprobe_bin=ffprobe_bin)
    cand_pts, _ = _probe_video_pts(cand_path, ffprobe_bin=ffprobe_bin)

    if expected_kind == 'proxy':
        proxy_info = data.get('proxy') or {}
        if not proxy_info.get('available'):
            raise RuntimeError('[append_clips] Proxy video is marked unavailable in analysis.json')
        rec_sha = proxy_info.get('sha256')
        if rec_sha and cand_media.get('sha256') != rec_sha:
            raise RuntimeError('[append_clips] Proxy identity mismatch: sha256 does not match recorded proxy')
        if proxy_info.get('size_bytes') is not None and cand_media['size_bytes'] != int(proxy_info['size_bytes']):
            raise RuntimeError(f"[append_clips] Proxy identity mismatch: size_bytes {cand_media['size_bytes']} != {proxy_info['size_bytes']}")
        if proxy_info.get('width') and proxy_info.get('height') and (cand_media.get('width') != proxy_info.get('width') or cand_media.get('height') != proxy_info.get('height')):
            raise RuntimeError('[append_clips] Proxy identity mismatch: resolution differs from recorded proxy')
        if proxy_info.get('frame_count') and len(cand_pts) != int(proxy_info['frame_count']):
            raise RuntimeError(f"[append_clips] Proxy timeline mismatch: frame count {len(cand_pts)} != {proxy_info['frame_count']}")
        p_dur = float(proxy_info.get('duration') or 0.0)
        if p_dur > 0 and abs(float(cand_media.get('duration') or 0.0) - p_dur) > 0.25:
            raise RuntimeError('[append_clips] Proxy timeline mismatch: duration differs from recorded proxy')
        return cand_media, cand_pts, proxy_info

    orig_media = data.get('media') or {}
    rec_sha = orig_media.get('sha256')
    if rec_sha and cand_media.get('sha256') != rec_sha:
        raise RuntimeError('[append_clips] Source identity mismatch: file sha256 does not match recorded original media')
    if orig_media.get('size_bytes') is not None and cand_media['size_bytes'] != int(orig_media['size_bytes']):
        raise RuntimeError(f"[append_clips] Source identity mismatch: size_bytes {cand_media['size_bytes']} != {orig_media['size_bytes']}")
    if orig_media.get('width') and orig_media.get('height'):
        if cand_media.get('width') != orig_media.get('width') or cand_media.get('height') != orig_media.get('height'):
            raise RuntimeError(
                f"[append_clips] Source identity mismatch: resolution {cand_media.get('width')}x{cand_media.get('height')} != {orig_media.get('width')}x{orig_media.get('height')}"
            )
    orig_dur = float(orig_media.get('duration') or 0.0)
    cand_dur = float(cand_media.get('duration') or 0.0)
    if orig_dur > 0 and abs(cand_dur - orig_dur) > max(0.15, orig_dur * 0.005):
        raise RuntimeError(f'[append_clips] Source timeline mismatch: duration {cand_dur:.3f}s != {orig_dur:.3f}s')
    if orig_media.get('fps') and cand_media.get('fps'):
        o_fps, c_fps = float(orig_media['fps']), float(cand_media['fps'])
        if abs(c_fps - o_fps) > max(0.25, o_fps * 0.02):
            raise RuntimeError(f'[append_clips] Source timeline mismatch: fps {c_fps:.3f} != {o_fps:.3f}')
    if orig_media.get('video_codec') and cand_media.get('video_codec') != orig_media.get('video_codec'):
        raise RuntimeError(f"[append_clips] Source identity mismatch: video_codec {cand_media.get('video_codec')} != {orig_media.get('video_codec')}")
    if orig_media.get('has_audio') is not None and bool(cand_media.get('has_audio')) != bool(orig_media.get('has_audio')):
        raise RuntimeError('[append_clips] Source identity mismatch: has_audio flag differs from recorded media')
    if orig_media.get('video_start_time') is not None and abs(float(cand_media.get('video_start_time') or 0.0) - float(orig_media['video_start_time'])) > 0.05:
        raise RuntimeError('[append_clips] Source timeline mismatch: video_start_time differs from recorded media')

    frames_data = data.get('frames') or {}
    rec_sfc = frames_data.get('source_frame_count')
    if rec_sfc is not None and len(cand_pts) != int(rec_sfc):
        raise RuntimeError(f'[append_clips] Source timeline mismatch: frame count {len(cand_pts)} != {rec_sfc}')

    items = frames_data.get('items') or []
    if items:
        step = max(1, len(items) // 24)
        sampled = items[::step]
        if items[-1] not in sampled:
            sampled.append(items[-1])
        for it in sampled:
            fi = it.get('frame_index')
            t_rec = it.get('t')
            if fi is None or t_rec is None:
                continue
            fi = int(fi)
            if fi < 0 or fi >= len(cand_pts):
                raise RuntimeError(f'[append_clips] Source timeline mismatch: frame_index {fi} out of bounds (0..{len(cand_pts)-1})')
            if abs(cand_pts[fi] - float(t_rec)) > 0.06:
                raise RuntimeError(f'[append_clips] Source timeline mismatch: frame_index {fi} PTS {cand_pts[fi]:.3f}s != recorded {float(t_rec):.3f}s')
    if not rec_sha:
        _check_sampled_luma_identity(cand_path, items)
    return cand_media, cand_pts, None


def _resolve_preserved_media_video(result_dir, orig_media):
    media_dir = Path(result_dir) / 'media'
    if not media_dir.is_dir():
        return None
    exts = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.flv', '.m4v'}
    cands = [p for p in sorted(media_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    target_sz = (orig_media or {}).get('size_bytes')
    for p in cands:
        if target_sz is not None and p.stat().st_size == int(target_sz):
            return p
    return cands[0]


def append_clips_to_result(a, clip_ranges):
    from PIL import Image
    started = time.perf_counter()
    result_dir = Path(a.append_clips).resolve()
    json_path = result_dir / 'analysis.json'
    md_path = result_dir / 'analysis.md'
    append_run_id = 'append-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    tmp = a.tmp_root / append_run_id
    cleanup_info = {'deleted': [], 'ok': True, 'error': None, 'skipped_reason': None}
    committed_files = []
    orig_json_bytes = None
    orig_md_bytes = None
    appended_entries = []
    source_kind = None
    batch_id = 0
    timings = {'validate_source': 0.0, 'clips': 0.0, 'persist': 0.0, 'total': 0.0}
    executed_stages = []
    skipped_stages = ['fetch', 'dynamic_frames', 'asr', 'ocr', 'storyboard', 'sheets', 'transitions']

    try:
        preflight_info = _preflight_checks(a, clip_ranges)
        executed_stages.append('preflight')
        orig_json_bytes = json_path.read_bytes()
        orig_md_bytes = md_path.read_bytes()
        data = json.loads(orig_json_bytes.decode('utf-8'))
        if data.get('status') != 'ok':
            raise RuntimeError(f"[append_clips] Target result directory status is '{data.get('status')}', refusing to append")

        t_val = time.perf_counter()
        if a.video:
            source_video = a.video.resolve()
            source_kind = 'original_reprovided'
        else:
            preserved = _resolve_preserved_media_video(result_dir, data.get('media'))
            if preserved is not None:
                source_video = preserved.resolve()
                source_kind = 'preserved_media'
            else:
                proxy_rec = data.get('proxy') or {}
                proxy_rel = proxy_rec.get('path') or 'proxy/proxy.mp4'
                proxy_file = (result_dir / proxy_rel).resolve()
                if proxy_rec.get('available') and proxy_file.is_file():
                    source_video = proxy_file
                    source_kind = 'proxy'
                else:
                    raise RuntimeError('[append_clips] No source video (--video), preserved media, or available proxy found for result directory')

        cand_media, cand_pts, proxy_meta = validate_clip_source(
            data, source_video, ffprobe_bin=preflight_info['ffprobe'], expected_kind=source_kind
        )
        timings['validate_source'] = round(time.perf_counter() - t_val, 3)
        executed_stages.append('validate_source')

        clip_media = copy.deepcopy(cand_media)
        if source_kind == 'proxy' and data.get('media'):
            # 保留代理自身的分辨率/帧率/时长供采样判定与画质标注使用
            clip_media['duration'] = float((proxy_meta or {}).get('duration') or cand_media.get('duration') or 0.0)
        _check_media_budgets(clip_media, clip_ranges, a, preflight_info.get('available_ram_mb'))

        frames_dict = data.get('frames') or {}
        existing_clips = list(frames_dict.get('exported_clips') or [])
        next_id = max((int(c.get('id') or 0) for c in existing_clips), default=0) + 1
        existing_hist = list(data.get('clip_append_history') or [])
        batch_id = len(existing_hist) + 1

        tmp.mkdir(parents=True, exist_ok=False)
        stage_out = tmp / 'stage_out'
        stage_out.mkdir(parents=True, exist_ok=True)

        t_clips = time.perf_counter()
        batch_warnings = []
        appended_entries = export_clips(
            clip_ranges,
            [],
            None,
            source_video,
            stage_out,
            media=clip_media,
            warnings=batch_warnings,
            ffprobe_bin=preflight_info['ffprobe'],
            max_clip_frames=a.max_clip_frames,
            start_id=next_id,
            pts_override=cand_pts,
            pts_source_override=('proxy_ffprobe_pts' if source_kind == 'proxy' else 'ffprobe_pts'),
            source_kind=source_kind,
            proxy_meta=proxy_meta,
            existing_clips_dir=(result_dir / 'clips'),
            appended=True,
            append_batch_id=batch_id,
        )
        staged_clips_dir = stage_out / 'clips'
        staged_files = sorted(staged_clips_dir.glob('*.jpg')) if staged_clips_dir.is_dir() else []
        for sf in staged_files:
            with Image.open(sf) as im:
                im.load()
        staged_bytes = sum(f.stat().st_size for f in staged_files)
        existing_res_bytes = _dir_size_bytes(result_dir)
        total_res_mb = round((existing_res_bytes + staged_bytes) / (1024 * 1024), 3)
        if a.max_result_mb > 0 and total_res_mb > a.max_result_mb:
            raise RuntimeError(f'[append_clips] Result directory size after append ({total_res_mb:.3f}MB) exceeds max_result_mb budget ({a.max_result_mb}MB)')
        timings['clips'] = round(time.perf_counter() - t_clips, 3)
        executed_stages.append('clips')

        for ne in appended_entries:
            ns, nend = float(ne.get('requested_start', 0.0)), float(ne.get('requested_end', 0.0))
            overlaps = []
            for ex in existing_clips:
                es, eend = float(ex.get('requested_start', ex.get('start', 0.0))), float(ex.get('requested_end', ex.get('end', 0.0)))
                if max(ns, es) < min(nend, eend) - 1e-6:
                    overlaps.append(ex.get('id'))
            if overlaps:
                ne['overlaps_clip_ids'] = overlaps

        new_data = copy.deepcopy(data)
        new_data.setdefault('initial_schema_version', data.get('schema_version'))
        new_data['schema_version'] = '1.7'
        new_data.setdefault('frames', {})['exported_clips'] = existing_clips + appended_entries
        existing_warnings = list(new_data.get('warnings') or [])
        for bw in batch_warnings:
            if bw not in existing_warnings:
                existing_warnings.append(bw)
        new_data['warnings'] = existing_warnings

        t_per = time.perf_counter()
        timings['persist'] = 0.0
        timings['total'] = round(time.perf_counter() - started, 3)
        hist_entry = {
            'batch_id': batch_id,
            'run_id': append_run_id,
            'timestamp': datetime.now().astimezone().isoformat(),
            'requested_clips': a.clips,
            'source_kind': source_kind,
            'source_name': source_video.name,
            'identity_verified': True,
            'timeline_verified': True,
            'executed_stages': executed_stages + ['persist'],
            'skipped_stages': skipped_stages,
            'appended_clip_ids': [e['id'] for e in appended_entries],
            'appended_images': [img for e in appended_entries for img in (e.get('images') or ([e['image']] if e.get('image') else []))],
            'timings': dict(timings),
        }
        new_data['clip_append_history'] = existing_hist + [hist_entry]

        dest_clips_dir = result_dir / 'clips'
        dest_clips_dir.mkdir(parents=True, exist_ok=True)
        try:
            for sf in staged_files:
                dest_file = dest_clips_dir / sf.name
                if dest_file.exists():
                    raise RuntimeError(f'[append_clips] Destination clip already exists: {dest_file.name}')
                shutil.copy2(sf, dest_file)
                committed_files.append(dest_file)
            hist_entry['timings']['persist'] = round(time.perf_counter() - t_per, 3)
            hist_entry['timings']['total'] = round(time.perf_counter() - started, 3)
            _persist_reports(result_dir, new_data)
            executed_stages.append('persist')
        except Exception:
            for cf in committed_files:
                if cf.exists():
                    try:
                        cf.unlink()
                    except Exception:
                        pass
            if orig_json_bytes is not None:
                json_path.write_bytes(orig_json_bytes)
            if orig_md_bytes is not None:
                md_path.write_bytes(orig_md_bytes)
            raise

        timings['persist'] = hist_entry['timings']['persist']
        timings['total'] = hist_entry['timings']['total']
        if tmp.exists():
            try:
                shutil.rmtree(tmp)
                cleanup_info['deleted'].append(str(tmp))
            except Exception as ce:
                cleanup_info['ok'] = False
                cleanup_info['error'] = safe_text(str(ce))
        print(json.dumps({
            'mode': 'append_clips',
            'result_dir': str(result_dir),
            'batch_id': batch_id,
            'source_kind': source_kind,
            'appended_clip_ids': [e['id'] for e in appended_entries],
            'total_exported_clips': len(new_data['frames']['exported_clips']),
            'executed_stages': executed_stages,
            'skipped_stages': skipped_stages,
            'timings': timings,
            'status': 'ok',
            'cleanup_ok': cleanup_info['ok'],
        }, ensure_ascii=False))
        return 0 if cleanup_info['ok'] else 1
    except Exception as e:
        err_msg = safe_text(str(e))
        if tmp.exists():
            try:
                shutil.rmtree(tmp)
                cleanup_info['deleted'].append(str(tmp))
            except Exception as ce:
                cleanup_info['ok'] = False
                cleanup_info['error'] = safe_text(str(ce))
        print(json.dumps({
            'mode': 'append_clips',
            'result_dir': str(result_dir),
            'source_kind': source_kind,
            'executed_stages': executed_stages,
            'skipped_stages': skipped_stages,
            'timings': {'total': round(time.perf_counter() - started, 3)},
            'status': 'error',
            'error': err_msg,
            'cleanup_ok': cleanup_info['ok'],
        }, ensure_ascii=False))
        return 1


def main():
    default_fetch_py = os.environ.get('DOUYIN_FETCH_PYTHON') or str(Path.home() / 'douyin-browser-mcp' / '.venv' / 'Scripts' / 'python.exe')
    parser = argparse.ArgumentParser()
    parser.add_argument('url', nargs='?'); parser.add_argument('--video', type=Path); parser.add_argument('--keep-media', action='store_true')
    parser.add_argument('--clips', default='', help='按时间段导出局部密集动作拼图，格式如 00:50-01:05,02:19-02:30')
    parser.add_argument('--append-clips', type=Path, default=None, help='基于已有结果目录增量追加 --clips（不重跑全片抽帧/下载/ASR/OCR）')
    parser.add_argument('--save-proxy', action='store_true', help='保存低码率视频代理至结果目录 proxy/proxy.mp4 并记录时间映射')
    parser.add_argument('--proxy-max-dim', type=int, default=640)
    parser.add_argument('--proxy-crf', type=int, default=28)
    parser.add_argument('--proxy-fps', type=float, default=0.0, help='可选代理帧率；0 表示保留源帧时间轴 (passthrough)')
    parser.add_argument('--model', default='large-v3'); parser.add_argument('--language', default='zh'); parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results')
    parser.add_argument('--tmp-root', type=Path, default=ROOT / 'tmp')
    parser.add_argument('--fetch-python', default=default_fetch_py)
    parser.add_argument('--ytdlp-bin', default=None, help='可选 yt-dlp 可执行路径或覆盖标记')
    parser.add_argument('--batch-file', type=Path, default=None, help='串行批量分析清单 (.txt / .json)')
    parser.add_argument('--resume-batch', type=Path, default=None, help='从既有批次目录断点续跑/重试失败任务（绝不覆盖已有成果）')
    parser.add_argument('--retry-degraded', action='store_true', help='配合 --resume-batch 时也重跑此前降级的任务')
    parser.add_argument('--stop-on-error', action='store_true', help='批量模式下遇到首条失败即停止，其余记为 not_executed')
    parser.add_argument('--batch-timeout', type=int, default=1800, help='批量模式单条子任务超时（秒）')
    parser.add_argument('--ffmpeg', default=None); parser.add_argument('--ffprobe', default=None)
    parser.add_argument('--max-duration', type=float, default=1800.0)
    parser.add_argument('--min-disk-mb', type=float, default=512.0)
    parser.add_argument('--max-memory-mb', type=float, default=2048.0)
    parser.add_argument('--max-frames', type=int, default=6000)
    parser.add_argument('--max-clip-frames', type=int, default=1200)
    parser.add_argument('--max-ocr-frames', type=int, default=96)
    parser.add_argument('--max-proxy-mb', type=float, default=256.0)
    parser.add_argument('--max-result-mb', type=float, default=1024.0)
    a = parser.parse_args()
    if a.batch_file or a.resume_batch:
        if a.url or a.video or a.append_clips:
            parser.error('--batch-file / --resume-batch cannot be combined with URL, --video, or --append-clips')
        import batch_analyze
        _, _, batch_exit = batch_analyze.run_batch(a)
        return batch_exit
    if a.append_clips:
        if a.url:
            parser.error('URL and --append-clips are mutually exclusive')
        if not a.clips:
            parser.error('--append-clips requires --clips')
        clip_ranges = parse_clip_ranges(a.clips)
        if not clip_ranges:
            parser.error('--append-clips requires at least one valid --clips range')
        return append_clips_to_result(a, clip_ranges)
    if a.url and a.video:
        parser.error('URL and --video are mutually exclusive')
    if not a.url and not a.video:
        parser.error('URL or --video is required')
    clip_ranges = parse_clip_ranges(a.clips)
    started = time.perf_counter(); run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    a.results_root.mkdir(parents=True, exist_ok=True)
    tmp = a.tmp_root / run_id
    initial_route = None
    if a.url:
        try:
            initial_route = classify_url_route(a.url)
        except UnsupportedRouteError:
            initial_route = None
    if a.video:
        label = re.sub(r'[^\w.-]+', '_', a.video.stem, flags=re.UNICODE).strip('._')[:80] or 'video'
    elif initial_route:
        label = initial_route['platform']
    else:
        label = 'url'
    result_dir = a.results_root / f'{label}-{run_id}'; result_dir.mkdir()
    budgets = {
        'max_duration_seconds': a.max_duration,
        'min_free_disk_mb': a.min_disk_mb,
        'max_memory_mb': a.max_memory_mb,
        'max_selected_frames': a.max_frames,
        'max_clip_frames': a.max_clip_frames,
        'max_ocr_frames': a.max_ocr_frames,
        'max_proxy_mb': a.max_proxy_mb,
        'max_result_mb': a.max_result_mb,
        'stage_timeouts_seconds': {'fetch': 300, 'probe': 60, 'frames': 600, 'audio_extract': 300, 'ffprobe_pts': 120, 'proxy': 300},
    }
    data = {
        'schema_version': '1.8',
        'status': 'error',
        'failed_stage': None,
        'run_id': run_id,
        'input': {
            'url': safe_text(a.url) if a.url else None,
            'platform': ('local' if a.video else (initial_route['platform'] if initial_route else 'unsupported')),
            'platform_item_id': None,
            'aweme_id': None,
            'downloader': ('local_file' if a.video else (initial_route['downloader'] if initial_route else None)),
            'route_kind': ('local_file' if a.video else (initial_route['route_kind'] if initial_route else 'unsupported')),
            'task_unique_key': None,
            'local_video': str(a.video) if a.video else None,
        },
        'budgets': budgets,
        'preflight': {},
        'stages': {},
        'metadata': None,
        'metadata_sources': {},
        'missing_metadata': {},
        'media': {},
        'proxy': {'enabled': bool(a.save_proxy), 'available': False},
        'style_profile': {},
        'frames': {},
        'audio_energy': {},
        'transcript': {'segments': [], 'filtered_segments': [], 'model': a.model, 'requested_device': a.device, 'device': None, 'compute_type': None, 'language': None, 'language_probability': None},
        'timeline': [],
        'timings': {k: 0.0 for k in ('fetch', 'probe', 'frames', 'asr', 'align', 'proxy', 'total')},
        'warnings': [],
        'cleanup': {'deleted': [], 'ok': True, 'error': None, 'skipped_reason': None, 'policy': 'cleanup_tmp_after_proxy_and_report_persist'},
    }
    video = None; error = None; current_stage = 'preflight'; persisted = False
    try:
        preflight_info = _preflight_checks(a, clip_ranges)
        data['preflight'] = preflight_info
        data['stages']['preflight'] = 'ok'
        tmp.mkdir(parents=True, exist_ok=False)

        current_stage = 'fetch'
        t = time.perf_counter()
        if a.video:
            video = a.video.resolve()
            data['missing_metadata'] = {'reason': 'local file'}
        else:
            route = preflight_info.get('route') or classify_url_route(a.url)
            data['input']['platform'] = route['platform']
            data['input']['downloader'] = route['downloader']
            data['input']['route_kind'] = route['route_kind']
            if route['downloader'] == 'douyin_browser_mcp':
                fetch_file = tmp / 'fetch.json'
                try:
                    command([a.fetch_python, ROOT / 'douyin_fetch.py', a.url, '--out-dir', tmp, '--result-json', fetch_file], timeout=budgets['stage_timeouts_seconds']['fetch'])
                except RuntimeError:
                    if fetch_file.exists():
                        fetch_error = json.loads(fetch_file.read_text(encoding='utf-8'))
                        raise RuntimeError('Fetch failed: ' + safe_text(fetch_error.get('error', {}).get('code', 'download error'))) from None
                    raise
                fetched_raw = json.loads(fetch_file.read_text(encoding='utf-8'))
                if fetched_raw.get('status') != 'completed':
                    raise RuntimeError('Fetch failed')
                fetched = normalize_media_result(fetched_raw, route)
            elif route['downloader'] == 'yt_dlp':
                ytdlp_cmd, ytdlp_err = resolve_ytdlp_runner(getattr(a, 'ytdlp_bin', None))
                if not ytdlp_cmd:
                    raise RuntimeError(f"[fetch] Optional downloader 'yt-dlp' unavailable: {ytdlp_err}")
                fetched = fetch_external_with_ytdlp(
                    a.url,
                    tmp,
                    route,
                    ytdlp_cmd,
                    preflight_info.get('ffmpeg'),
                    command_runner=command,
                    timeout=budgets['stage_timeouts_seconds']['fetch'],
                )
            else:
                raise RuntimeError(f"[fetch] Unsupported downloader route: {route.get('downloader')}")
            video = Path(fetched['video_path'])
            data['metadata'] = fetched.get('metadata')
            data['metadata_sources'] = fetched.get('metadata_sources') or {}
            data['missing_metadata'] = fetched.get('missing') or {}
            data['warnings'].extend((fetched.get('download') or {}).get('warnings') or [])
            plat = fetched.get('platform') or route['platform']
            pid = fetched.get('platform_item_id')
            data['input']['platform'] = plat
            data['input']['platform_item_id'] = pid
            data['input']['task_unique_key'] = fetched.get('task_unique_key')
            ident = (data['metadata'] or {}).get('aweme_id') if plat == 'douyin' else None
            data['input']['aweme_id'] = ident
            if plat == 'douyin' and ident and re.fullmatch(r'\d+', str(ident)):
                renamed = a.results_root / f'{ident}-{run_id}'
                result_dir.rename(renamed)
                result_dir = renamed
            elif pid:
                safe_pid = re.sub(r'[^\w.-]+', '_', str(pid), flags=re.UNICODE).strip('._')[:64]
                if safe_pid:
                    renamed = a.results_root / f'{plat}_{safe_pid}-{run_id}'
                    result_dir.rename(renamed)
                    result_dir = renamed
        data['timings']['fetch'] = round(time.perf_counter()-t, 3)
        data['stages']['fetch'] = 'ok'

        current_stage = 'probe'
        t = time.perf_counter()
        data['media'] = probe(video, ffprobe_bin=preflight_info['ffprobe'])
        if a.video:
            local_sha = (data['media'].get('sha256') or '')[:12]
            if local_sha:
                data['input']['platform_item_id'] = local_sha
                data['input']['task_unique_key'] = f'local:{local_sha}'
        _check_media_budgets(data['media'], clip_ranges, a, preflight_info.get('available_ram_mb'))
        data['timings']['probe'] = round(time.perf_counter()-t, 3)
        data['stages']['probe'] = 'ok'

        current_stage = 'frames'
        t = time.perf_counter(); frame_dir = tmp / 'frames'; frame_json = tmp / 'frames.json'
        command([
            sys.executable, ROOT / 'dynamic_frames.py', video,
            '--out-dir', frame_dir, '--json', frame_json, '--keep-frames',
            '--cmd-timeout', str(budgets['stage_timeouts_seconds']['frames']),
            '--max-selected-frames', str(a.max_frames),
        ], timeout=budgets['stage_timeouts_seconds']['frames'])
        sampled = json.loads(frame_json.read_text(encoding='utf-8'))
        items = [{'index': i, 't': f['t'], 'frame_index': f['frame_index'], 'reasons': f.get('reasons', []), 'scores': f.get('scores', {})} for i, f in enumerate(sampled['frames'])]
        baseline = round(data['media']['duration'] * 10)
        current_stage = 'visual_features'
        visual_features = analyze_visual_features(items, frame_dir, data['media'])
        lighting_shifts = detect_lighting_transitions(items)
        current_stage = 'sheets'
        sheets, sheet_map = make_sheets(items, frame_dir, result_dir, data['media'])
        current_stage = 'clips'
        exported_clips = export_clips(clip_ranges, items, frame_dir, video, result_dir, data['media'], warnings=data['warnings'], ffprobe_bin=preflight_info['ffprobe'], max_clip_frames=a.max_clip_frames)
        data['frames'] = {'count': len(items), 'baseline_10fps': baseline, 'saving_ratio': round(1-len(items)/baseline, 4) if baseline else None, 'scene_cuts': sampled.get('scene_cuts', []), 'lighting_transitions': sorted(lighting_shifts), 'timeline_sheet_count': len(sheets), 'exported_clips': exported_clips, 'items': items, 'sampler_parameters': sampled.get('parameters', {}), 'source_frame_count': sampled.get('source_frame_count'), 'analysis_sample_count': sampled.get('analysis_sample_count')}
        data['timings']['frames'] = round(time.perf_counter()-t, 3)
        data['stages']['frames'] = 'ok'

        current_stage = 'asr'
        t = time.perf_counter()
        if data['media']['has_audio']:
            wav = tmp / 'audio.wav'
            command([preflight_info['ffmpeg'], '-nostdin', '-v', 'error', '-y', '-i', video, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav], timeout=budgets['stage_timeouts_seconds']['audio_extract'])
            data['transcript'] = transcribe(wav, a.model, a.language, a.device, data['warnings'])
            data['audio_energy'] = analyze_audio_file(wav, media=data['media'])
            av_offset = float(data['media'].get('av_start_offset') or 0.0)
            if abs(av_offset) >= 0.02:
                data['warnings'].append(f'检测到音视频起点偏移 {av_offset:+.3f}s，音频能量窗口已按视频时间轴对齐。')
        else:
            data['audio_energy'] = {'status': 'no_audio_track', 'has_audio': False}
            data['warnings'].append('视频没有音轨，未进行语音识别。')
        data['timings']['asr'] = round(time.perf_counter()-t, 3)
        data['stages']['asr'] = 'ok'

        current_stage = 'align'
        t = time.perf_counter()
        data['timeline'] = align_timeline(items, data['transcript']['segments'])
        preliminary_shots = partition_preliminary_shots(data['timeline'], lighting_shifts=lighting_shifts)
        sb_items = select_storyboard_frames(items, sampled.get('scene_cuts', []), data['transcript']['segments'], data['media'].get('duration'), lighting_shifts=lighting_shifts)
        ocr_targets, ocr_coverage_audit = select_ocr_frames(sb_items, preliminary_shots, max_ocr_frames=a.max_ocr_frames)
        ocr_by_index, recurring_ocr, ocr_status = extract_keyframe_ocr(ocr_targets, frame_dir, data['media'], coverage_audit=ocr_coverage_audit)
        ocr_frames_audit = ocr_status.pop('frames', [])
        data['frames']['ocr_status'] = ocr_status
        data['frames']['ocr_frames'] = ocr_frames_audit
        if ocr_status.get('status') in ('unavailable', 'error', 'partial_error'):
            err_suffix = f" ({ocr_status['error']})" if ocr_status.get('error') else ''
            data['warnings'].append(f"关键帧 OCR 状态异常 ({ocr_status['status']}: {ocr_status['failed_frames']}/{ocr_status['total_frames']} 帧失败){err_suffix}。")
        if ocr_status.get('budget_exceeded'):
            data['warnings'].append(
                f"OCR 帧数超出预算上限 ({ocr_status.get('max_ocr_frames_budget')} 帧)，仅覆盖 {ocr_status.get('covered_shot_count')}/{ocr_status.get('preliminary_shots_with_frames_count')} 个有帧初步分镜，不得视为全分镜已扫描。"
            )
        for idx_k, kf in enumerate(sb_items):
            if kf['index'] in ocr_by_index:
                kf['ocr_texts'] = ocr_by_index[kf['index']]
            prev_kf = sb_items[idx_k - 1] if idx_k > 0 else None
            sc = (kf.get('scores') or {}).get('scene_score') if 'scene_after' in (kf.get('reasons') or []) else None
            kf['transition_desc'] = describe_frame_delta(prev_kf, kf, sc, ocr_by_index)
        current_stage = 'storyboard'
        make_storyboard(sb_items, frame_dir, result_dir, data['media'])
        current_stage = 'transitions'
        trans_strips = select_transition_strips(items, sampled.get('scene_cuts', []), lighting_shifts=lighting_shifts, ocr_by_index=ocr_by_index)
        make_transitions_sheet(trans_strips, frame_dir, result_dir, data['media'])
        data['frames']['storyboard_frames'] = [{'index': f['index'], 't': f['t'], 'frame_index': f['frame_index'], 'highlight_reason': f.get('highlight_reason'), 'transition_desc': f.get('transition_desc'), 'ocr_texts': f.get('ocr_texts', []), 'visual': f.get('visual')} for f in sb_items]
        data['style_profile'] = build_style_profile(data['media'], data['frames'], data['transcript'], data['timeline'], visual_features, sb_items, ocr_by_index=ocr_by_index, recurring_ocr=recurring_ocr, lighting_shifts=lighting_shifts, sheet_map=sheet_map, transition_strips=trans_strips, audio_energy=data['audio_energy'])
        data['timings']['align'] = round(time.perf_counter()-t, 3)
        data['stages']['align'] = 'ok'

        if a.save_proxy and video and video.is_file():
            current_stage = 'proxy'
            t_px = time.perf_counter()
            data['proxy'] = build_video_proxy(
                video,
                tmp,
                result_dir,
                data['media'],
                preflight_info['ffmpeg'],
                preflight_info['ffprobe'],
                max_dim=a.proxy_max_dim,
                crf=a.proxy_crf,
                proxy_fps=a.proxy_fps,
                max_proxy_mb=a.max_proxy_mb,
                max_result_mb=a.max_result_mb,
                min_disk_mb=a.min_disk_mb,
                source_kind=('local_video' if a.video else 'downloaded_original'),
                aweme_id=data['input'].get('aweme_id'),
                source_frame_count=data['frames'].get('source_frame_count'),
                timeout=budgets['stage_timeouts_seconds']['proxy'],
            )
            data['timings']['proxy'] = round(time.perf_counter() - t_px, 3)
            data['stages']['proxy'] = 'ok'
            if data['proxy'].get('downsampled'):
                data['warnings'].append('已生成降帧低码率代理（代理帧率低于原片，不可用于补回原片丢失的动作帧）。')

        if a.keep_media and video and video.is_file():
            current_stage = 'keep_media'
            dest = result_dir / 'media'; dest.mkdir(exist_ok=True)
            shutil.copy2(video, dest / video.name)
            for manifest in tmp.glob('*.json'):
                if manifest.name not in ('frames.json', 'fetch.json'):
                    content = scrub(json.loads(manifest.read_text(encoding='utf-8')))
                    if isinstance(content, dict):
                        content['output_path'] = str(dest / video.name)
                    (dest / manifest.name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
            data['stages']['keep_media'] = 'ok'

        total_res_mb = round(_dir_size_bytes(result_dir) / (1024 * 1024), 3)
        if a.max_result_mb > 0 and total_res_mb > a.max_result_mb:
            raise RuntimeError(f'[budget] Result directory size ({total_res_mb:.3f}MB) exceeds max_result_mb ({a.max_result_mb}MB)')

        data['status'] = 'ok'
    except Exception as e:
        error = safe_text(str(e))
        data['status'] = 'error'
        data['failed_stage'] = current_stage
        data['stages'][current_stage] = 'error'
        data['error'] = error
        if current_stage == 'proxy' and isinstance(data.get('proxy'), dict):
            data['proxy']['available'] = False
            data['proxy']['status'] = 'error'
            data['proxy']['error'] = error
        partial = sorted(str(p.relative_to(result_dir)) for p in result_dir.rglob('*') if p.is_file() and p.name not in ('analysis.json', 'analysis.md'))
        if partial:
            data['partial_outputs'] = partial
    finally:
        data['timings']['total'] = round(time.perf_counter()-started, 3)
        # 必须先完成代理索引与报告持久化，再清理临时目录（含下载原片）；写入失败时保留 tmp 作为恢复依据。
        try:
            data = _persist_reports(result_dir, data)
            persisted = True
        except Exception as pe:
            persist_err = safe_text(str(pe))
            error = error or persist_err
            data['status'] = 'error'
            data['failed_stage'] = data.get('failed_stage') or 'persist'
            data['stages']['persist'] = 'error'
            data['error'] = error
            data['cleanup']['ok'] = False
            data['cleanup']['skipped_reason'] = 'persist_failed_preserved_tmp'
            if tmp.exists():
                data['cleanup']['preserved_tmp'] = str(tmp)
        if persisted and tmp.exists():
            try:
                shutil.rmtree(tmp)
                data['cleanup']['deleted'].append(str(tmp))
            except Exception as ce:
                data['cleanup']['ok'] = False
                data['cleanup']['error'] = safe_text(str(ce))
                data['warnings'].append('临时目录清理失败，产物已持久化但未完整清理。')
                if data['status'] == 'ok':
                    data['status'] = 'completed_with_cleanup_failure'
                error = error or f'Cleanup failed: {data["cleanup"]["error"]}'
                try:
                    data = _persist_reports(result_dir, data)
                except Exception:
                    pass
    print(json.dumps({'result_dir': str(result_dir), 'frame_count': data['frames'].get('count', 0), 'storyboard_count': len(data['frames'].get('storyboard_frames', [])), 'transition_strips': len(data.get('style_profile', {}).get('transition_strips', [])), 'timeline_sheets': data['frames'].get('timeline_sheet_count', 0), 'segment_count': len(data['transcript'].get('segments', [])), 'proxy_available': bool((data.get('proxy') or {}).get('available')), 'timings': data['timings'], 'status': data['status'], 'cleanup_ok': data['cleanup']['ok']}, ensure_ascii=False))
    return 1 if (error or data['status'] != 'ok') else 0


if __name__ == '__main__': sys.exit(main())
