# 串行批量编排与可追溯极简摘要：严格串行调度分析任务，隔离单条故障，区分成功/降级/失败/未执行并生成零覆盖批次索引。
"""Serial batch runner and traceable summary generator for douyin-analyzer."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from media_router import UnsupportedRouteError, classify_url_route, safe_text, sanitize_replay_url

ROOT = Path(__file__).resolve().parent


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def _short_hash(text):
    return hashlib.sha1(str(text).encode('utf-8', errors='replace')).hexdigest()[:10]


def _parse_txt_line(line):
    parts = [p.strip() for p in line.split('|')]
    target = parts[0]
    spec = {}
    if re.match(r'^https?://', target, re.I):
        spec['url'] = target
    else:
        spec['video'] = target
    for opt in parts[1:]:
        if '=' not in opt:
            continue
        k, v = opt.split('=', 1)
        k = k.strip().lower()
        v = v.strip()
        if k == 'clips':
            spec['clips'] = v
        elif k == 'label':
            spec['label'] = v
        elif k == 'save_proxy':
            spec['save_proxy'] = v.lower() in ('1', 'true', 'yes', 'on')
        elif k == 'keep_media':
            spec['keep_media'] = v.lower() in ('1', 'true', 'yes', 'on')
        elif k == 'max_ocr_frames':
            try:
                spec['max_ocr_frames'] = int(v)
            except ValueError:
                spec['spec_error'] = f'Invalid max_ocr_frames: {v}'
    return spec


# 读取 .txt 或 .json 批量清单，返回规范化任务规格列表。
def load_batch_specs(batch_file):
    batch_path = Path(batch_file)
    if not batch_path.is_file():
        raise RuntimeError(f'[batch] Batch file not found: {batch_file}')
    raw = batch_path.read_text(encoding='utf-8-sig')
    specs = []
    if batch_path.suffix.lower() == '.json':
        parsed = json.loads(raw)
        items = parsed.get('tasks') if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            raise RuntimeError('[batch] JSON batch file must be a list or an object containing a "tasks" list')
        for idx, item in enumerate(items, 1):
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    specs.append({'task_index': idx, 'spec_error': 'Empty task string'})
                else:
                    entry = _parse_txt_line(s)
                    entry['task_index'] = idx
                    specs.append(entry)
            elif isinstance(item, dict):
                entry = dict(item)
                entry['task_index'] = idx
                specs.append(entry)
            else:
                specs.append({'task_index': idx, 'spec_error': f'Unsupported task item type: {type(item).__name__}'})
    else:
        idx = 0
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            idx += 1
            entry = _parse_txt_line(line)
            entry['task_index'] = idx
            specs.append(entry)
    if not specs:
        raise RuntimeError(f'[batch] No tasks found in batch file: {batch_file}')
    return specs


# 生成跨平台唯一的复合任务标识，避免只依赖跨平台可能重复的数字 ID。
def build_initial_task_identity(spec, batch_base_dir=None):
    idx = spec.get('task_index', 0)
    url = (spec.get('url') or '').strip() if isinstance(spec.get('url'), str) else ''
    video = (spec.get('video') or '').strip() if isinstance(spec.get('video'), str) else ''
    if url and video:
        return {
            'task_uid': f'invalid:conflict_{idx}_{_short_hash(url + "|" + video)}',
            'platform': 'invalid',
            'route_kind': 'conflict',
            'input_kind': 'invalid',
            'input_display': safe_text(url) or video,
            'spec_error': 'Task cannot specify both url and video',
        }
    if url:
        cleaned_url = safe_text(url) or '[URL]'
        try:
            route = classify_url_route(url)
            platform = route['platform']
            route_kind = route['route_kind']
        except UnsupportedRouteError:
            platform = 'unsupported'
            route_kind = 'unsupported'
        return {
            'task_uid': f'{platform}:url_{_short_hash(url)}',
            'platform': platform,
            'route_kind': route_kind,
            'input_kind': 'url',
            'input_display': cleaned_url,
            'spec_error': spec.get('spec_error'),
        }
    if video:
        p = Path(video)
        if not p.is_absolute() and batch_base_dir is not None:
            p = (Path(batch_base_dir) / p)
        resolved = str(p.resolve()) if p else video
        return {
            'task_uid': f'local:{p.stem[:32]}_{_short_hash(resolved)}',
            'platform': 'local',
            'route_kind': 'local_file',
            'input_kind': 'video',
            'input_display': str(p),
            'resolved_video_path': resolved,
            'spec_error': spec.get('spec_error'),
        }
    return {
        'task_uid': f'invalid:empty_{idx}',
        'platform': 'invalid',
        'route_kind': 'missing_input',
        'input_kind': 'invalid',
        'input_display': '(missing)',
        'spec_error': spec.get('spec_error') or 'Task missing url or video',
    }


def _extract_sampling_limits(data):
    limits = []
    frames = data.get('frames') or {}
    budgets = data.get('budgets') or {}
    max_frames = budgets.get('max_selected_frames')
    if max_frames and int(frames.get('count') or 0) >= int(max_frames):
        limits.append(f'selected_frames_reached_budget:{max_frames}')
    ocr_status = frames.get('ocr_status') or {}
    if ocr_status.get('budget_exceeded'):
        limits.append(
            f"ocr_budget_exceeded:{ocr_status.get('covered_shot_count')}/{ocr_status.get('preliminary_shots_with_frames_count')}_shots(max={ocr_status.get('max_ocr_frames_budget')})"
        )
    if ocr_status.get('single_frame_shot_count'):
        limits.append(f"ocr_single_frame_shots:{ocr_status.get('single_frame_shot_count')}")
    for c in (frames.get('exported_clips') or []):
        for r in (c.get('coverage_insufficient_reasons') or []):
            tag = f"clip#{c.get('clip_id')}:{r}"
            if tag not in limits:
                limits.append(tag)
    proxy = data.get('proxy') or {}
    if proxy.get('downsampled'):
        limits.append(f"proxy_downsampled:{proxy.get('fps')}fps(cannot_recover_original_motion)")
    return limits


def _extract_experimental_flags(data):
    flags = {'vlm_experimental': False, 'stage5_vlm': 'skipped_by_plan'}
    if data.get('schema_version'):
        flags['schema_version'] = data['schema_version']
    proxy = data.get('proxy') or {}
    if proxy.get('enabled'):
        flags['proxy_enabled'] = True
        if proxy.get('downsampled'):
            flags['proxy_downsampled'] = True
    return flags


# 依据退出码与落盘 analysis.json 准确归类：success / degraded / failed。
def classify_task_outcome(exit_code, result_dir, data, stdout_err=None):
    degradations = []
    if exit_code != 0 or not isinstance(data, dict):
        err_msg = None
        if isinstance(data, dict):
            err_msg = data.get('error')
        if not err_msg:
            err_msg = safe_text(stdout_err) or f'Process exited with code {exit_code}'
        return 'failed', (data.get('status') if isinstance(data, dict) else 'error'), degradations, err_msg

    raw_status = data.get('status')
    if raw_status == 'error':
        return 'failed', raw_status, degradations, data.get('error') or 'Analysis status is error'

    if raw_status == 'completed_with_cleanup_failure':
        degradations.append('cleanup_failure:tmp_preserved_or_uncleaned')

    frames = data.get('frames') or {}
    ocr_status = frames.get('ocr_status') or {}
    ocr_state = ocr_status.get('status')
    if ocr_state in ('unavailable', 'error', 'partial_error'):
        degradations.append(f"ocr_{ocr_state}:{ocr_status.get('failed_frames')}/{ocr_status.get('total_frames')}")
    if ocr_status.get('budget_exceeded'):
        degradations.append(
            f"ocr_budget_exceeded:{ocr_status.get('covered_shot_count')}/{ocr_status.get('preliminary_shots_with_frames_count')}"
        )

    transcript = data.get('transcript') or {}
    if transcript.get('requested_device') == 'cuda' and transcript.get('device') == 'cpu':
        degradations.append('asr_cuda_fallback_to_cpu')
    for w in (data.get('warnings') or []):
        wl = str(w).lower()
        if ('cuda' in wl and 'cpu' in wl) or '语音识别失败' in wl or 'asr 不可用' in wl:
            if 'asr_degraded' not in degradations:
                degradations.append('asr_degraded')

    category = 'degraded' if degradations else 'success'
    return category, raw_status, degradations, None


def _parse_json_line(stdout_text):
    for line in reversed((stdout_text or '').splitlines()):
        s = line.strip()
        if s.startswith('{') and s.endswith('}'):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
    return None


def _build_counts(tasks):
    counts = {'total': len(tasks), 'success': 0, 'degraded': 0, 'failed': 0, 'not_executed': 0}
    for t in tasks:
        cat = t.get('status_category')
        if cat in counts:
            counts[cat] += 1
        else:
            counts['failed'] += 1
    return counts


# 生成极简但可完整追溯到单条结果的 Markdown 摘要，保留警告、采样限制和实验标识。
def render_batch_summary(index_data):
    counts = index_data.get('counts') or {}
    lines = [
        '# 批量视频分析总摘要（串行可追溯版）',
        '',
        f"- **批次 ID**：`{index_data.get('batch_id')}`",
        f"- **执行模式**：`{index_data.get('execution_mode')}`（默认严格串行，避免显存竞争；单条失败不掩盖其余结果）",
        f"- **Schema 版本**：`{index_data.get('schema_version')}`",
        f"- **创建时间**：{index_data.get('created_at')}",
    ]
    if index_data.get('resumed_from_batch'):
        lines.append(f"- **断点/重试继承来源**：`{index_data.get('resumed_from_batch')}`（已成功产物直接引用，零覆盖）")
    lines += [
        f"- **任务总数与状态统计**：总计 **{counts.get('total', 0)}** | 成功 **{counts.get('success', 0)}** | 降级 **{counts.get('degraded', 0)}** | 失败 **{counts.get('failed', 0)}** | 未执行 **{counts.get('not_executed', 0)}**",
        '',
        '| # | 复合任务标识 | 平台 / 作品ID | 状态分类 (原始状态) | 耗时(s) | 关键警告与降级 | 采样限制 | 实验标识 | 完整结果追溯 |',
        '| --- | --- | --- | --- | ---: | --- | --- | --- | --- |',
    ]

    def _cell(v):
        if v is None or v == '':
            return '—'
        return str(v).replace('|', '\\|').replace('\n', ' ')

    for t in index_data.get('tasks') or []:
        cid = t.get('canonical_task_id') or t.get('task_uid')
        plat = t.get('platform') or 'unknown'
        pid = t.get('platform_item_id') or '—'
        cat = t.get('status_category')
        raw_st = t.get('raw_status') or '—'
        state_str = f"**{cat}** ({raw_st})"
        if t.get('reused_existing_result'):
            state_str += ' [复用旧成果]'
        warn_parts = []
        if t.get('error'):
            warn_parts.append(f"错误: {t['error']}")
        for d in (t.get('degradation_reasons') or []):
            warn_parts.append(f"降级: {d}")
        for w in (t.get('warnings') or [])[:3]:
            warn_parts.append(w)
        if len(t.get('warnings') or []) > 3:
            warn_parts.append(f"等共{len(t['warnings'])}条警告")
        warn_str = '；'.join(warn_parts) or '无'

        lim_str = '；'.join(t.get('sampling_limits') or []) or '无限制触发'
        exp = t.get('experimental_flags') or {}
        exp_str = f"VLM={exp.get('vlm_experimental', False)}" + (', 代理降帧' if exp.get('proxy_downsampled') else '')

        links = []
        if t.get('analysis_md') and Path(t['analysis_md']).is_file():
            links.append(f"`{t['analysis_md']}`")
        if t.get('analysis_json') and Path(t['analysis_json']).is_file():
            links.append(f"`{t['analysis_json']}`")
        if not links and t.get('result_dir'):
            links.append(f"`{t['result_dir']}`")
        trace_str = ' / '.join(links) or '未产出目录'

        lines.append(
            f"| {t.get('task_index')} | `{_cell(cid)}` | `{_cell(plat)}` / `{_cell(pid)}` | {_cell(state_str)} | {t.get('elapsed_seconds', 0.0)} | {_cell(warn_str)} | {_cell(lim_str)} | {_cell(exp_str)} | {_cell(trace_str)} |"
        )
    lines.append('')
    return '\n'.join(lines)


def _persist_batch_outputs(batch_dir, index_data):
    index_data['counts'] = _build_counts(index_data['tasks'])
    summary_md = render_batch_summary(index_data)
    _atomic_write_text(batch_dir / 'batch_index.json', json.dumps(index_data, ensure_ascii=False, indent=2))
    _atomic_write_text(batch_dir / 'batch_summary.md', summary_md)


def _load_resumable_map(resume_batch_dir, retry_degraded=False):
    p = Path(resume_batch_dir)
    idx_file = p / 'batch_index.json' if p.is_dir() else p
    if not idx_file.is_file():
        raise RuntimeError(f'[batch] Resume batch index not found: {resume_batch_dir}')
    old = json.loads(idx_file.read_text(encoding='utf-8'))
    reusable = {}
    for t in old.get('tasks') or []:
        cat = t.get('status_category')
        keep = (cat == 'success') or (cat == 'degraded' and not retry_degraded)
        res_dir = t.get('result_dir')
        if keep and res_dir and (Path(res_dir) / 'analysis.json').is_file() and (Path(res_dir) / 'analysis.md').is_file():
            reusable[t.get('task_uid')] = t
    return old.get('batch_id'), reusable


def _run_single_subtask(spec, identity, args, env_overrides=None):
    cmd = [
        sys.executable,
        str(ROOT / 'analyze.py'),
        '--results-root', str(args.results_root),
        '--tmp-root', str(args.tmp_root),
        '--fetch-python', str(args.fetch_python),
        '--model', str(args.model),
        '--language', str(args.language),
        '--device', str(args.device),
        '--max-duration', str(args.max_duration),
        '--min-disk-mb', str(args.min_disk_mb),
        '--max-memory-mb', str(args.max_memory_mb),
        '--max-frames', str(args.max_frames),
        '--max-clip-frames', str(args.max_clip_frames),
        '--max-ocr-frames', str(spec.get('max_ocr_frames') or args.max_ocr_frames),
        '--max-proxy-mb', str(args.max_proxy_mb),
        '--max-result-mb', str(args.max_result_mb),
    ]
    if getattr(args, 'ffmpeg', None):
        cmd += ['--ffmpeg', str(args.ffmpeg)]
    if getattr(args, 'ffprobe', None):
        cmd += ['--ffprobe', str(args.ffprobe)]
    if getattr(args, 'ytdlp_bin', None):
        cmd += ['--ytdlp-bin', str(args.ytdlp_bin)]
    if spec.get('keep_media') or getattr(args, 'keep_media', False):
        cmd.append('--keep-media')
    if spec.get('save_proxy') or getattr(args, 'save_proxy', False):
        cmd.append('--save-proxy')
    clips_val = spec.get('clips') if spec.get('clips') is not None else getattr(args, 'clips', '')
    if clips_val:
        cmd += ['--clips', str(clips_val)]
    if identity['input_kind'] == 'url':
        cmd.append(spec['url'].strip())
    else:
        cmd += ['--video', identity['resolved_video_path']]

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    if env_overrides:
        env.update(env_overrides)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=int(getattr(args, 'batch_timeout', 1800) or 1800),
            env=env,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        parsed = _parse_json_line(proc.stdout)
        result_dir = Path(parsed['result_dir']) if isinstance(parsed, dict) and parsed.get('result_dir') else None
        data = None
        if result_dir and (result_dir / 'analysis.json').is_file():
            try:
                data = json.loads((result_dir / 'analysis.json').read_text(encoding='utf-8'))
            except Exception:
                data = None
        err_tail = (proc.stderr or '')[-500:] if proc.returncode != 0 else None
        return proc.returncode, elapsed, result_dir, data, err_tail
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        return 124, elapsed, None, None, f'Subtask timed out after {exc.timeout}s'


# 串行执行批量任务或基于旧批次断点续跑；绝不覆盖已有成功成果，返回 (batch_dir, index_data, exit_code)。
def run_batch(args):
    started = time.perf_counter()
    batch_run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    batch_dir = results_root / f'batch-{batch_run_id}'
    batch_dir.mkdir(parents=True, exist_ok=False)

    resumed_from_id = None
    reusable_map = {}
    if getattr(args, 'resume_batch', None):
        resumed_from_id, reusable_map = _load_resumable_map(
            args.resume_batch,
            retry_degraded=bool(getattr(args, 'retry_degraded', False)),
        )

    batch_base_dir = Path(args.batch_file).resolve().parent if getattr(args, 'batch_file', None) else None
    if getattr(args, 'batch_file', None):
        specs = load_batch_specs(args.batch_file)
    elif getattr(args, 'resume_batch', None):
        p = Path(args.resume_batch)
        idx_file = p / 'batch_index.json' if p.is_dir() else p
        old_data = json.loads(idx_file.read_text(encoding='utf-8'))
        specs = []
        for t in old_data.get('tasks') or []:
            s = dict(t.get('spec') or {})
            s['task_index'] = t.get('task_index')
            specs.append(s)
    else:
        raise RuntimeError('[batch] Either --batch-file or --resume-batch is required')

    index_data = {
        'schema_version': '1.8',
        'batch_id': f'batch-{batch_run_id}',
        'execution_mode': 'serial',
        'created_at': datetime.now().isoformat(),
        'batch_file': str(args.batch_file) if getattr(args, 'batch_file', None) else None,
        'resumed_from_batch': resumed_from_id,
        'stop_on_error': bool(getattr(args, 'stop_on_error', False)),
        'counts': {'total': len(specs), 'success': 0, 'degraded': 0, 'failed': 0, 'not_executed': 0},
        'tasks': [],
        'elapsed_seconds': 0.0,
    }

    stop_triggered = False
    for spec in specs:
        identity = build_initial_task_identity(spec, batch_base_dir=batch_base_dir)
        uid = identity['task_uid']
        sanitized_spec = {
            k: (sanitize_replay_url(v) or safe_text(v) if k == 'url' else v)
            for k, v in spec.items()
        }

        if uid in reusable_map:
            prev = dict(reusable_map[uid])
            prev['task_index'] = spec.get('task_index')
            prev['reused_existing_result'] = True
            prev['inherited_from_batch'] = resumed_from_id
            prev['elapsed_seconds'] = 0.0
            index_data['tasks'].append(prev)
            _persist_batch_outputs(batch_dir, index_data)
            continue

        if stop_triggered:
            index_data['tasks'].append({
                'task_index': spec.get('task_index'),
                'task_uid': uid,
                'canonical_task_id': uid,
                'platform': identity['platform'],
                'platform_item_id': None,
                'route_kind': identity['route_kind'],
                'input_display': identity['input_display'],
                'spec': sanitized_spec,
                'status_category': 'not_executed',
                'raw_status': 'not_executed',
                'exit_code': None,
                'elapsed_seconds': 0.0,
                'result_dir': None,
                'analysis_json': None,
                'analysis_md': None,
                'warnings': ['因开启 stop_on_error 且前置任务失败，本条未执行。'],
                'degradation_reasons': [],
                'sampling_limits': [],
                'experimental_flags': {'vlm_experimental': False, 'stage5_vlm': 'skipped_by_plan'},
                'error': 'Skipped due to stop_on_error',
                'reused_existing_result': False,
            })
            _persist_batch_outputs(batch_dir, index_data)
            continue

        if identity.get('spec_error'):
            index_data['tasks'].append({
                'task_index': spec.get('task_index'),
                'task_uid': uid,
                'canonical_task_id': uid,
                'platform': identity['platform'],
                'platform_item_id': None,
                'route_kind': identity['route_kind'],
                'input_display': identity['input_display'],
                'spec': sanitized_spec,
                'status_category': 'failed',
                'raw_status': 'error',
                'exit_code': 2,
                'elapsed_seconds': 0.0,
                'result_dir': None,
                'analysis_json': None,
                'analysis_md': None,
                'warnings': [],
                'degradation_reasons': [],
                'sampling_limits': [],
                'experimental_flags': {'vlm_experimental': False, 'stage5_vlm': 'skipped_by_plan'},
                'error': identity['spec_error'],
                'reused_existing_result': False,
            })
            if getattr(args, 'stop_on_error', False):
                stop_triggered = True
            _persist_batch_outputs(batch_dir, index_data)
            continue

        exit_code, elapsed, result_dir, data, err_tail = _run_single_subtask(spec, identity, args)
        cat, raw_status, degradations, err_msg = classify_task_outcome(exit_code, result_dir, data, stdout_err=err_tail)

        inp = (data or {}).get('input') or {}
        platform = inp.get('platform') or identity['platform']
        item_id = inp.get('platform_item_id') or inp.get('aweme_id')
        if not item_id and platform == 'local' and isinstance(data, dict):
            sha = ((data.get('media') or {}).get('sha256') or '')
            if sha:
                item_id = sha[:12]
        canonical_id = f'{platform}:{item_id}' if item_id else uid
        warnings = list((data or {}).get('warnings') or [])
        sampling_limits = _extract_sampling_limits(data or {}) if isinstance(data, dict) else []
        exp_flags = _extract_experimental_flags(data or {}) if isinstance(data, dict) else {'vlm_experimental': False, 'stage5_vlm': 'skipped_by_plan'}

        task_record = {
            'task_index': spec.get('task_index'),
            'task_uid': uid,
            'canonical_task_id': canonical_id,
            'platform': platform,
            'platform_item_id': item_id,
            'route_kind': inp.get('route_kind') or identity['route_kind'],
            'downloader': inp.get('downloader'),
            'input_display': identity['input_display'],
            'spec': sanitized_spec,
            'status_category': cat,
            'raw_status': raw_status,
            'exit_code': exit_code,
            'elapsed_seconds': elapsed,
            'result_dir': str(result_dir) if result_dir else None,
            'analysis_json': str(result_dir / 'analysis.json') if result_dir and (result_dir / 'analysis.json').is_file() else None,
            'analysis_md': str(result_dir / 'analysis.md') if result_dir and (result_dir / 'analysis.md').is_file() else None,
            'warnings': warnings,
            'degradation_reasons': degradations,
            'sampling_limits': sampling_limits,
            'experimental_flags': exp_flags,
            'error': err_msg,
            'reused_existing_result': False,
        }
        index_data['tasks'].append(task_record)
        if cat == 'failed' and getattr(args, 'stop_on_error', False):
            stop_triggered = True
        _persist_batch_outputs(batch_dir, index_data)

    index_data['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    _persist_batch_outputs(batch_dir, index_data)

    counts = index_data['counts']
    summary_out = {
        'mode': 'batch',
        'batch_id': index_data['batch_id'],
        'batch_dir': str(batch_dir),
        'batch_index': str(batch_dir / 'batch_index.json'),
        'batch_summary': str(batch_dir / 'batch_summary.md'),
        'counts': counts,
        'elapsed_seconds': index_data['elapsed_seconds'],
    }
    print(json.dumps(summary_out, ensure_ascii=False))
    # 存在 failed 或 not_executed 时返回非 0；全部为 success/degraded 时返回 0。
    overall_exit = 1 if (counts.get('failed', 0) > 0 or counts.get('not_executed', 0) > 0) else 0
    return batch_dir, index_data, overall_exit
