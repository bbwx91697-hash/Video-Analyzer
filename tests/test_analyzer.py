# 流水线对齐、脱敏、输出及模拟集成测试。
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import (
    _clip_coverage_metrics,
    _coverage_sort_key,
    _extract_clip_frames_from_video,
    _find_media_tool,
    _probe_video_pts,
    align_timeline,
    analyze_visual_features,
    build_smart_shots,
    build_style_profile,
    command,
    compute_audio_energy,
    describe_frame_delta,
    export_clips,
    extract_keyframe_ocr,
    make_sheets,
    make_storyboard,
    make_transitions_sheet,
    parse_clip_ranges,
    partition_preliminary_shots,
    read_wav_pcm,
    render_markdown,
    select_ocr_frames,
    select_storyboard_frames,
    select_transition_strips,
    transcribe,
)
from douyin_fetch import sanitize_metadata, create_time_iso


# 无输入；验证重叠语音区间的帧唯一归属和排序，无返回值。
def test_alignment():
    frames = [{'index': i, 't': t, 'frame_index': i, 'reasons': []} for i, t in enumerate([0, 1, 2, 3, 4, 5])]
    segments = [{'id': 0, 'start': 1, 'end': 3, 'text': 'a'}, {'id': 1, 'start': 2, 'end': 4, 'text': 'b'}]
    timeline = align_timeline(frames, segments)
    got = [f['index'] for row in timeline for f in row['frames']]
    assert sorted(got) == list(range(6)) and len(set(got)) == 6
    assert [row['start'] for row in timeline] == sorted(row['start'] for row in timeline)
    assert next(r for r in timeline if r.get('segment_id') == 0)['frames'][0]['index'] == 1


# 无输入；验证元数据过滤与东八区时间转换，无返回值。
def test_security_and_time():
    result = sanitize_metadata({'title': 'safe', 'canonical_url': 'https://www.douyin.com/video/123?secret=x', 'cookie': 'x', 'token': 'x', 'sign': 'x', 'author_nickname': 'https://a/b?x=y'})
    assert not any('cookie' in k or 'token' in k or 'sign' in k for k in result)
    assert not result.get('canonical_url') and not result.get('author_nickname')
    assert create_time_iso(0) == '1970-01-01T08:00:00+08:00'


# 无输入；验证缺失元数据时仍能生成 Markdown，无返回值。
def test_markdown_nulls():
    assert isinstance(render_markdown({'metadata': None, 'media': {}, 'frames': {}, 'transcript': {}, 'timeline': []}), str)

# 无输入；通过模拟下载与抽帧验证结果、脱敏、媒体保留和临时目录清理，无返回值。
def test_fetch_pipeline_integration():
    import json
    import tempfile
    from unittest.mock import patch
    from PIL import Image
    import analyze
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        # 输入模拟命令参数；写入测试夹具并返回空输出。
        def fake_command(args, timeout=None):
            if str(args[1]).endswith('douyin_fetch.py'):
                out = Path(args[args.index('--out-dir') + 1]); video = out / 'douyin-123.mp4'; video.write_bytes(b'video')
                (out / 'douyin-123.mp4.manifest.json').write_text(json.dumps({'output_path': str(video), 'token': 'secret'}))
                result = {'status': 'completed', 'video_path': str(video), 'metadata': {'aweme_id': '123'}, 'download': {'warnings': ['no audio']}}
                Path(args[args.index('--result-json') + 1]).write_text(json.dumps(result), encoding='utf-8')
            else:
                out = Path(args[args.index('--out-dir') + 1]); out.mkdir()
                Image.new('RGB', (200, 100), 'red').save(out / 'frame_00000000.jpg')
                sample = {'frames': [{'t': 0, 'frame_index': 0, 'reasons': ['first']}], 'parameters': {}, 'scene_cuts': []}
                Path(args[args.index('--json') + 1]).write_text(json.dumps(sample))
            return ''
        probe = {'duration': 1, 'width': 200, 'height': 100, 'fps': 10, 'video_codec': 'h264', 'has_audio': False, 'audio_codec': None, 'size_bytes': 5}
        argv = ['analyze.py', 'https://www.douyin.com/video/123?token=secret', '--results-root', str(root/'results'), '--tmp-root', str(root/'tmp'), '--keep-media']
        with patch.object(sys, 'argv', argv), patch.object(analyze, 'command', fake_command), patch.object(analyze, 'probe', return_value=probe):
            assert analyze.main() == 0
        result_dir = next((root/'results').iterdir())
        data = json.loads((result_dir/'analysis.json').read_text(encoding='utf-8'))
        assert data['status'] == 'ok' and data['frames']['count'] == 1 and data['input']['aweme_id'] == '123'
        assert 'secret' not in (result_dir/'analysis.json').read_text(encoding='utf-8')
        assert (result_dir/'timeline_001.jpg').is_file() and (result_dir/'media'/'douyin-123.mp4').is_file()
        manifest = json.loads((result_dir/'media'/'douyin-123.mp4.manifest.json').read_text())
        assert 'token' not in manifest and Path(manifest['output_path']).is_file()
        assert not list((root/'tmp').iterdir())


# 无输入；验证无帧语音段两侧的无语音帧分组，无返回值。
def test_no_speech_gaps_split_at_empty_speech():
    frames = [{'index': i, 't': t} for i, t in enumerate([0, 5])]
    rows = align_timeline(frames, [{'id': 0, 'start': 1, 'end': 4, 'text': 'x'}])
    assert [r['type'] for r in rows] == ['no-speech', 'speech', 'no-speech']


# 无输入；验证全量备查拼图零丢帧、sheet_map 索引与 transitions.jpg 连拍板生成，无返回值。
def test_zero_drop_sheets_and_transitions():
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        fdir = root / 'frames'; fdir.mkdir()
        out = root / 'out'; out.mkdir()
        items = []
        for i in range(40):
            Image.new('RGB', (160, 90), (220 if i == 14 else 40, 80, 120)).save(fdir / f'frame_{i:08d}.jpg')
            reasons = ['scene_before'] if i == 14 else (['scene_after'] if i == 15 else ['local_motion'])
            scores = {'scene_score': 0.52 if i == 15 else None, 'local_max_diff': 32.0 if i in (15, 25, 26, 27, 28) else 12.0}
            items.append({'index': i, 't': i * 0.25, 'frame_index': i, 'reasons': reasons, 'scores': scores,
                          'visual': {'luma': 180.0 if i == 14 else 45.0, 'contrast': 40.0, 'sat': 30.0, 'warm': 0.0}})
        paths, smap = make_sheets(items, fdir, out, {'width': 1920, 'height': 1080})
        assert len(paths) == 2 and all(p.is_file() for p in paths)
        assert len(smap) == 40 and smap[0] == 'timeline_001.jpg' and smap[39] == 'timeline_002.jpg'
        strips = select_transition_strips(items, [{'frame_index': 15, 't': 3.75, 'score': 0.52}], max_strips=4)
        assert strips and '高亮转暗' in strips[0]['summary'] and '强光冲白' not in strips[0]['summary']
        t_path = make_transitions_sheet(strips, fdir, out, {'width': 1920, 'height': 1080})
        assert t_path is not None and t_path.is_file()


# 无输入；验证时间区间参数解析及非有限数拒绝（P1-4 回归），无返回值。
def test_parse_clip_ranges():
    assert parse_clip_ranges('00:50-01:05, 139-145.5') == [(50.0, 65.0), (139.0, 145.5)]
    for bad in ('10-5', 'inf-10', '0-inf', '-inf-5', 'nan-5', '0-nan', '0:inf-10'):
        try:
            parse_clip_ranges(bad)
            assert False, f'Expected ValueError for {bad}'
        except ValueError:
            pass


# 无输入；验证短无语音片段并入前段时完整重算时间、帧范围、拼图与关键帧（P1-1 回归），无返回值。
def test_build_smart_shots_merge_recomputes_fields():
    f490 = {'index': 490, 't': 19.60, 'frame_index': 490, 'reasons': ['local_motion'], 'scores': {'local_max_diff': 10.0},
            'visual': {'luma': 100.0, 'contrast': 30.0, 'sat': 20.0, 'warm': 0.0}}
    f491 = {'index': 491, 't': 19.75, 'frame_index': 491, 'reasons': ['local_motion'], 'scores': {'local_max_diff': 15.0},
            'visual': {'luma': 110.0, 'contrast': 30.0, 'sat': 20.0, 'warm': 0.0}}
    f492 = {'index': 492, 't': 19.85, 'frame_index': 492, 'reasons': ['scene_before'], 'scores': {'local_max_diff': 18.0},
            'visual': {'luma': 190.0, 'contrast': 35.0, 'sat': 20.0, 'warm': 0.0}}
    f493 = {'index': 493, 't': 20.00, 'frame_index': 493, 'reasons': ['scene_after'], 'scores': {'scene_score': 0.5, 'local_max_diff': 35.0},
            'visual': {'luma': 40.0, 'contrast': 35.0, 'sat': 20.0, 'warm': 0.0}}
    timeline = [
        {'type': 'speech', 'start': 18.0, 'end': 19.6, 'text': '前句台词', 'frames': [f490]},
        {'type': 'no-speech', 'start': 19.7, 'end': 19.85, 'text': '', 'frames': [f491, f492]},
        {'type': 'speech', 'start': 19.9, 'end': 21.0, 'text': '后句台词', 'frames': [f493]},
    ]
    sheet_map = {490: 'timeline_015.jpg', 491: 'timeline_016.jpg', 492: 'timeline_016.jpg', 493: 'timeline_016.jpg'}
    shots = build_smart_shots(timeline, {491}, sheet_map=sheet_map)
    assert len(shots) == 2
    first, second = shots[0], shots[1]
    assert first['end'] == 19.85 and first['duration'] == 1.85
    assert first['frame_count'] == 3
    assert first['frame_range'] == '#490–#492 (3帧)'
    assert first['timeline_sheets'] == ['timeline_015.jpg', 'timeline_016.jpg']
    assert any('#491(' in kf for kf in first['keyframes'])
    assert '_frames' not in first and '_head_frame' not in first and '_tail_frame' not in first
    # 下一段的转场描述必须基于合并后的真实尾帧 #492(luma=190) -> #493(luma=40)
    assert '高亮转暗' in second['transition_desc']


# 无输入；验证 missing frame JPEG 时拼图严格报错而不是生成黑格（P1-3 回归），无返回值。
def test_missing_frame_image_fails_explicitly():
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        fdir = root / 'frames'; fdir.mkdir()
        out = root / 'out'; out.mkdir()
        Image.new('RGB', (160, 90), 'red').save(fdir / 'frame_00000000.jpg')
        items = [
            {'index': 0, 't': 0.0, 'frame_index': 0, 'reasons': ['first'], 'scores': {}},
            {'index': 1, 't': 0.5, 'frame_index': 1, 'reasons': ['last'], 'scores': {}},
        ]
        for fn, args in (
            (make_sheets, (items, fdir, out)),
            (make_storyboard, (items, fdir, out)),
            (make_transitions_sheet, ([{'id': 1, 'kind': 'test', 't0': 0.0, 't1': 0.5, 'summary': 's', 'frames': items}], fdir, out)),
            (export_clips, ([(0.0, 0.6)], items, fdir, None, out)),
        ):
            try:
                fn(*args)
                assert False, f'{fn.__name__} should fail on missing frame image'
            except RuntimeError as e:
                assert 'Missing frame image' in str(e)


# 无输入；验证 --clips 在 >=6 稀帧时仍补抽、按真实 PTS 且不超过 10fps、并分页不丢弃 >60 帧（P1-2 回归），无返回值。
def test_export_clips_pts_rate_limit_supplement_and_pagination():
    import subprocess
    import tempfile
    from PIL import Image
    ffmpeg_bin = _find_media_tool('ffmpeg')
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        vid = root / 'sample_24fps.mp4'
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc=size=160x90:rate=24:duration=7.0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid),
        ], check=True)
        pts, pts_src = _probe_video_pts(vid)
        assert pts_src == 'ffprobe_pts' and len(pts) >= 165 and abs(pts[24] - 1.0) < 0.05

        fdir = root / 'frames'; fdir.mkdir()
        out = root / 'out'; out.mkdir()
        # 故意放 7 帧但覆盖 0-7s 且跨度>1s，验证不会因为 >=6 帧而跳过补抽
        sparse_items = []
        for idx, fi in enumerate((0, 24, 48, 72, 96, 120, 144)):
            Image.new('RGB', (160, 90), 'blue').save(fdir / f'frame_{fi:08d}.jpg')
            sparse_items.append({'index': idx, 't': float(idx), 'frame_index': fi, 'reasons': ['local_motion'], 'scores': {}})

        exported = export_clips([(0.0, 7.0)], sparse_items, fdir, vid, out, {'width': 160, 'height': 90, 'fps': 24.0})
        assert len(exported) == 1
        info = exported[0]
        # 7 秒 @ <=10fps -> 最多 71 帧（若错误用 round(24/10)=2 则是 12fps ≈ 84 帧），且应 >60 帧证明未截断
        assert 65 <= info['frame_count'] <= 71, f"Unexpected clip frame_count: {info['frame_count']}"
        assert len(info.get('images') or []) == 2
        for rel in info['images']:
            assert (out / rel).is_file() and (out / rel).stat().st_size > 0

        # 同时直接验证 VFR 非均匀 PTS 抽样率不超过 10fps 且不重复帧
        resample = getattr(Image, 'Resampling', Image).LANCZOS
        vfr_pts = [0.0, 0.03, 0.09, 0.10, 0.14, 0.21, 0.50, 0.90]
        from unittest.mock import patch
        import cv2
        import numpy as np
        class FakeCap:
            def __init__(self): self.i = -1
            def isOpened(self): return True
            def grab(self): self.i += 1; return self.i < len(vfr_pts)
            def retrieve(self): return True, np.zeros((90, 160, 3), dtype=np.uint8)
            def release(self): pass
        with patch.object(cv2, 'VideoCapture', return_value=FakeCap()):
            got, ext_meta = _extract_clip_frames_from_video(vid, 0.0, 0.95, 160, 90, resample, pts=vfr_pts)
        # 0.0(keep,next=0.1), 0.10(keep,next=0.2), 0.21(keep,next=0.3), 0.50(keep,next=0.6), 0.90(keep)
        assert [round(x['t'], 2) for x in got] == [0.0, 0.10, 0.21, 0.50, 0.90]
        assert not ext_meta['decode_ended_early']

        # 低帧率（2fps）与近结尾/完全越界区间、以及审计字段验证
        vid_low = root / 'low_2fps.mp4'
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc=size=160x90:rate=2:duration=3.0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid_low),
        ], check=True)
        out_low = root / 'out_low'; out_low.mkdir()
        low_warnings = []
        low_exported = export_clips(
            [(1.5, 4.0), (5.0, 6.0)], [], fdir, vid_low, out_low,
            {'width': 160, 'height': 90, 'fps': 2.0, 'duration': 3.0}, warnings=low_warnings
        )
        assert len(low_exported) == 2
        c1, c2 = low_exported
        assert c1['requested_start'] == 1.5 and c1['requested_end'] == 4.0
        assert c1['actual_first_pts'] is not None and c1['actual_last_pts'] is not None
        assert c1['time_source'] == 'ffprobe_pts'
        assert 'clamped_to_video_end' in c1['coverage_insufficient_reasons']
        assert 'source_fps_below_10fps' in c1['coverage_insufficient_reasons']
        assert c2['frame_count'] == 0 and c2['time_source'] == 'out_of_range'
        assert 'out_of_video_duration' in c2['coverage_insufficient_reasons']
        assert any('超出视频总时长' in w for w in low_warnings)

        # 覆盖质量择优（同为 5 帧，但全覆盖 0..4s 比全部挤在 0..0.4s 质量更高）
        clustered = [{'t': i * 0.1} for i in range(5)]
        spread = [{'t': float(i)} for i in range(5)]
        assert _coverage_sort_key(_clip_coverage_metrics(spread, 0.0, 4.0)) > _coverage_sort_key(_clip_coverage_metrics(clustered, 0.0, 4.0))

        # PTS 失败不得静默
        with patch('analyze.command', side_effect=RuntimeError('ffprobe failed')):
            try:
                _probe_video_pts(vid_low, ffprobe_bin='ffprobe')
                assert False, 'Expected RuntimeError when ffprobe PTS fails'
            except RuntimeError as e:
                assert '[clips]' in str(e)


# 无输入；验证阶段1：损坏图像/分析缺图、持久化优先于清理、清理失败独立状态、预算拒绝、Windows 中文空格路径及无效入口（非零退出码），无返回值。
def test_stage1_corrupt_persist_cleanup_budgets_and_windows_paths():
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    from PIL import Image
    import analyze

    # 1. 损坏图片与 analyze_visual_features 缺图不得静默跳过
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        fdir = root / 'frames'; fdir.mkdir()
        out = root / 'out'; out.mkdir()
        corrupt = fdir / 'frame_00000000.jpg'
        corrupt.write_bytes(b'not-a-valid-jpeg')
        bad_items = [{'index': 0, 't': 0.0, 'frame_index': 0, 'reasons': ['first'], 'scores': {}}]
        try:
            analyze_visual_features(bad_items, fdir, {'width': 160, 'height': 90})
            assert False, 'analyze_visual_features should reject corrupt JPEG'
        except RuntimeError as e:
            assert '[visual_features]' in str(e)
        try:
            make_sheets(bad_items, fdir, out)
            assert False, 'make_sheets should reject corrupt JPEG'
        except RuntimeError as e:
            assert '[sheets]' in str(e) and 'frame #0' in str(e)

    # 2. 持久化和清理顺序：写入失败时保留 tmp；清理失败时独立状态且返回非零
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        vid = root / 'v.mp4'; vid.write_bytes(b'123')
        probe_data = {'duration': 1.0, 'width': 160, 'height': 90, 'fps': 10.0, 'video_codec': 'h264', 'has_audio': False, 'audio_codec': None, 'size_bytes': 3}
        def fake_cmd(args, timeout=None):
            out = Path(args[args.index('--out-dir') + 1]); out.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (160, 90), 'red').save(out / 'frame_00000000.jpg')
            Path(args[args.index('--json') + 1]).write_text(json.dumps({'frames': [{'t': 0.0, 'frame_index': 0, 'reasons': ['first']}], 'parameters': {}, 'scene_cuts': []}))
            return ''

        # 2a. 写入失败 -> tmp 不得删除
        argv = ['analyze.py', '--video', str(vid), '--results-root', str(root/'res_fail'), '--tmp-root', str(root/'tmp_fail')]
        with patch.object(sys, 'argv', argv), patch.object(analyze, 'command', fake_cmd), patch.object(analyze, 'probe', return_value=probe_data), patch.object(analyze, '_persist_reports', side_effect=OSError('disk full')):
            code = analyze.main()
            assert code == 1
        assert list((root/'tmp_fail').iterdir()), 'tmp must be preserved when persisting reports fails'

        # 2b. 清理失败 -> 状态不为 ok 且返回非零
        argv2 = ['analyze.py', '--video', str(vid), '--results-root', str(root/'res_cleanfail'), '--tmp-root', str(root/'tmp_cleanfail')]
        with patch.object(sys, 'argv', argv2), patch.object(analyze, 'command', fake_cmd), patch.object(analyze, 'probe', return_value=probe_data), patch.object(shutil, 'rmtree', side_effect=PermissionError('locked')):
            code = analyze.main()
            assert code == 1
        rdir = next((root/'res_cleanfail').iterdir())
        saved = json.loads((rdir / 'analysis.json').read_text(encoding='utf-8'))
        assert saved['status'] == 'completed_with_cleanup_failure' and saved['cleanup']['ok'] is False

        # 2c. 超出时长预算直接拒绝
        long_probe = dict(probe_data, duration=9999.0)
        argv3 = ['analyze.py', '--video', str(vid), '--results-root', str(root/'res_budget'), '--tmp-root', str(root/'tmp_budget'), '--max-duration', '10']
        with patch.object(sys, 'argv', argv3), patch.object(analyze, 'probe', return_value=long_probe):
            code = analyze.main()
            assert code == 1
        rdir3 = next((root/'res_budget').iterdir())
        saved3 = json.loads((rdir3 / 'analysis.json').read_text(encoding='utf-8'))
        assert saved3['status'] == 'error' and saved3['failed_stage'] == 'probe' and 'exceeds budget' in saved3['error']

    # 3. 命令行互斥输入、缺 ffmpeg、无效下载器 -> 非零退出码
    res_both = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'analyze.py'), 'https://v.douyin.com/x', '--video', 'x.mp4'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res_both.returncode != 0
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        vid = root / 'x.mp4'; vid.write_bytes(b'x')
        res_noff = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'analyze.py'), '--video', str(vid), '--ffmpeg', str(root/'missing_ffmpeg.exe'), '--results-root', str(root/'r'), '--tmp-root', str(root/'t')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res_noff.returncode == 1
        res_badfetch = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'analyze.py'), 'https://v.douyin.com/x', '--fetch-python', str(root/'missing_py.exe'), '--results-root', str(root/'r2'), '--tmp-root', str(root/'t2')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res_badfetch.returncode == 1

    # 4. Windows 中文+空格路径 + 非默认项目目录真实合成媒体端到端抽帧测试
    proj_root = Path(__file__).resolve().parents[1]
    ffmpeg_bin = _find_media_tool('ffmpeg')
    with tempfile.TemporaryDirectory(dir=proj_root) as temp:
        cn_proj = Path(temp) / '非默认 目录'
        cn_proj.mkdir()
        for fn in ('analyze.py', 'dynamic_frames.py'):
            shutil.copy2(proj_root / fn, cn_proj / fn)
        media_dir = Path(temp) / '中文 素材'
        media_dir.mkdir()
        vid = media_dir / '测试 视频.mp4'
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc=size=160x90:rate=10:duration=1.2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vid),
        ], check=True)
        # 在复制出来的非默认中文空格目录运行 dynamic_frames + sheets/storyboard 验证不误写回原目录
        out_frames = media_dir / '输出 帧'
        fjson = media_dir / 'frames.json'
        run_df = subprocess.run([sys.executable, str(cn_proj / 'dynamic_frames.py'), str(vid), '--out-dir', str(out_frames), '--json', str(fjson), '--keep-frames'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        assert run_df.returncode == 0, run_df.stderr
        parsed_df = json.loads(fjson.read_text(encoding='utf-8'))
        assert parsed_df['selected_count'] >= 2
        for fr in parsed_df['frames']:
            assert (out_frames / f"frame_{fr['frame_index']:08d}.jpg").is_file()

    # 5. ASR 被过滤片段保留过滤理由
    with tempfile.TemporaryDirectory(dir=proj_root) as temp:
        import wave
        import numpy as np
        wav = Path(temp) / 'a.wav'
        with wave.open(str(wav), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
        fake_segs = [
            SimpleNamespace(start=0.0, end=1.0, text='杂音', no_speech_prob=0.95),
            SimpleNamespace(start=1.0, end=2.0, text='感谢观看订阅转发', no_speech_prob=0.10),
            SimpleNamespace(start=2.0, end=3.0, text='   ', no_speech_prob=0.10),
            SimpleNamespace(start=3.0, end=4.0, text='正常口播内容', no_speech_prob=0.05),
        ]
        class FakeWhisper:
            def __init__(self, *args, **kwargs): pass
            def transcribe(self, *args, **kwargs):
                return iter(fake_segs), SimpleNamespace(language='zh', language_probability=0.99)
        fake_fw = SimpleNamespace(WhisperModel=FakeWhisper)
        with patch.dict(sys.modules, {'faster_whisper': fake_fw}):
            tr = transcribe(wav, 'large-v3', 'zh', 'cpu', [])
        assert len(tr['segments']) == 1 and tr['segments'][0]['text'] == '正常口播内容'
        reasons = [x['reason'] for x in tr['filtered_segments']]
        assert reasons == ['no_speech_prob_gt_0.85', 'hallucination_pattern', 'empty_after_sanitize']
        md = render_markdown({'metadata': None, 'media': {'duration': 4.0}, 'frames': {'count': 0}, 'transcript': tr, 'timeline': []})
        assert 'ASR 过滤追溯：共过滤 3 条候选片段' in md


# 无输入；验证 P2 修复：结尾/少帧连拍、OCR 状态记录、command 超时与脱敏 stderr、以及严谨措辞，无返回值。
def test_p2_fixes_end_frames_ocr_timeout_and_markdown():
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        fdir = root / 'frames'; fdir.mkdir()
        out = root / 'out'; out.mkdir()

        # 1. 结尾不足 4 帧的光影突变回拉到 4 帧 + 总数不足 4 帧的短视频正常生成连拍条
        items_10 = [{'index': i, 't': i * 0.5, 'frame_index': i, 'reasons': ['local_motion'],
                     'visual': {'luma': 180.0 if i < 9 else 40.0, 'contrast': 30.0, 'sat': 20.0, 'warm': 0.0}}
                    for i in range(10)]
        strips_10 = select_transition_strips(items_10, [], lighting_shifts={9})
        assert len(strips_10) == 1 and len(strips_10[0]['frames']) == 4
        assert [f['index'] for f in strips_10[0]['frames']] == [6, 7, 8, 9]

        for i in range(3):
            Image.new('RGB', (160, 90), 'green').save(fdir / f'frame_{i:08d}.jpg')
        items_3 = [
            {'index': 0, 't': 0.0, 'frame_index': 0, 'reasons': ['first'], 'visual': {'luma': 10.0}},
            {'index': 1, 't': 0.5, 'frame_index': 1, 'reasons': ['scene_after'], 'scores': {'scene_score': 0.6}, 'visual': {'luma': 90.0}},
            {'index': 2, 't': 1.0, 'frame_index': 2, 'reasons': ['last'], 'visual': {'luma': 95.0}},
        ]
        strips_3 = select_transition_strips(items_3, [{'frame_index': 1, 't': 0.5, 'score': 0.6}])
        assert len(strips_3) == 1 and len(strips_3[0]['frames']) == 3
        t_path = make_transitions_sheet(strips_3, fdir, out)
        assert t_path is not None and t_path.is_file()

        # 2. OCR 状态记录（跳过 / 失败计数）
        _, _, st_skip = extract_keyframe_ocr([], fdir)
        assert st_skip['status'] == 'skipped'
        _, _, st_err = extract_keyframe_ocr([{'index': 99, 'frame_index': 9999}], fdir)
        assert st_err['status'] in ('unavailable', 'error') and st_err['failed_frames'] == 1

    # 3. command 超时与脱敏 stderr
    try:
        command([sys.executable, '-c', 'import time; time.sleep(2)'], timeout=0.1)
        assert False, 'Expected timeout RuntimeError'
    except RuntimeError as e:
        assert 'timed out' in str(e)

    try:
        command([sys.executable, '-c', "import sys; sys.stderr.write('err https://a.b/x?token=TOPSECRET cookie=ABC'); sys.exit(2)"])
        assert False, 'Expected CalledProcessError RuntimeError'
    except RuntimeError as e:
        msg = str(e)
        assert 'TOPSECRET' not in msg and 'ABC' not in msg and 'exit 2' in msg

    # 4. 无语音措辞不夸大为“全程无对白”
    style = build_style_profile({'duration': 5.0}, {'scene_cuts': []}, {'segments': []}, [], {}, [])
    md = render_markdown({'metadata': None, 'media': {'duration': 5.0}, 'frames': {'count': 0}, 'transcript': {'segments': []}, 'timeline': [], 'style_profile': style})
    assert '未检测到语音片段' in style['audio_narrative']['audio_style']
    assert '未检测到语音片段。' in md and '全程无人声台词' not in md and '无转写片段。' not in md

    # 5. 客观转场措辞检查
    desc_dark = describe_frame_delta({'visual': {'luma': 10.0}}, {'visual': {'luma': 80.0}}, cut_score=0.45)
    assert '极暗转亮' in desc_dark and '黑屏' not in desc_dark
    desc_drop = describe_frame_delta({'visual': {'luma': 80.0}}, {'visual': {'luma': 10.0}})
    assert '亮度骤降至极暗' in desc_drop and '收束' not in desc_drop


# 无输入；验证真实 VFR MP4 抽帧、dynamic_frames 命令超时与显式编码、以及 analyze.ps1 输入互斥返回非零，无返回值。
def test_real_vfr_dynamic_frames_timeout_and_ps1():
    import subprocess
    import tempfile
    import dynamic_frames as df_mod
    ffmpeg_bin = _find_media_tool('ffmpeg')
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        # 1. 生成真实 VFR MP4（前 10 帧 0.05s 间隔=20fps，后 10 帧 0.25s 间隔=4fps）
        vfr = root / 'real_vfr.mp4'
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc2=size=160x90:rate=10:duration=2.0',
            '-vf', "settb=expr=1/1000,setpts='if(lt(N,10),N*0.05/TB,(0.5+(N-10)*0.25)/TB)'",
            '-fps_mode', 'vfr', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(vfr),
        ], check=True)
        pts, src = _probe_video_pts(vfr)
        assert src == 'ffprobe_pts' and len(pts) >= 18
        deltas = {round(b - a, 2) for a, b in zip(pts, pts[1:])}
        assert 0.05 in deltas and 0.25 in deltas, f'Expected 0.05 and 0.25 VFR deltas, got {deltas}'
        out_vfr = root / 'out_vfr'; out_vfr.mkdir()
        res_vfr = export_clips([(0.0, pts[-1] + 0.05)], [], root, vfr, out_vfr, {'width': 160, 'height': 90, 'fps': 7.27, 'duration': pts[-1] + 0.05})
        assert len(res_vfr) == 1 and res_vfr[0]['time_source'] == 'ffprobe_pts'
        # 前 0.5s 按 <=10fps 取 5 帧(0.0,0.1,0.2,0.3,0.4)，后半 4fps 取全部 -> 共约 13-15 帧
        assert 12 <= res_vfr[0]['frame_count'] <= 16, f"Unexpected VFR frame count: {res_vfr[0]['frame_count']}"

    # 2. dynamic_frames.command 超时保护
    try:
        df_mod.command([sys.executable, '-c', 'import time; time.sleep(2)'], timeout=0.1)
        assert False, 'dynamic_frames.command should raise on timeout'
    except RuntimeError as e:
        assert 'timed out' in str(e)

    # 3. analyze.ps1 URL 与 -Video 同传必须非零退出
    ps1 = Path(__file__).resolve().parents[1] / 'analyze.ps1'
    ps_res = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ps1), 'https://v.douyin.com/x', '-Video', 'x.mp4'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
    )
    assert ps_res.returncode != 0


# 无输入；验证 Markdown 报告清楚区分无音轨、有音轨无语音、OCR 无字与 OCR 识别器不可用，无返回值。
def test_stage2_no_audio_notext_and_recognizer_unavailable_distinction():
    style_no_audio = build_style_profile({'duration': 5.0, 'has_audio': False}, {'scene_cuts': []}, {'segments': []}, [], {}, [])
    assert '无音轨' in style_no_audio['audio_narrative']['audio_style']
    md_no_audio = render_markdown({
        'metadata': None,
        'media': {'duration': 5.0, 'has_audio': False},
        'frames': {'count': 0, 'ocr_status': {'status': 'unavailable', 'total_frames': 2, 'processed_frames': 0, 'failed_frames': 2, 'error': 'RapidOCR missing'}},
        'transcript': {'segments': []},
        'timeline': [],
        'style_profile': style_no_audio,
        'warnings': ['视频没有音轨，未进行语音识别。', '关键帧 OCR 状态异常 (unavailable: 2/2 帧失败)。'],
    })
    assert '无音轨（未进行语音识别）' in md_no_audio
    assert '视频没有音轨，未进行语音识别。' in md_no_audio
    assert 'OCR 识别器不可用' in md_no_audio
    assert '## 运行提示与告警' in md_no_audio

    style_with_audio = build_style_profile({'duration': 5.0, 'has_audio': True}, {'scene_cuts': []}, {'segments': []}, [], {}, [])
    assert '纯配乐/环境音展示' in style_with_audio['audio_narrative']['audio_style']
    md_with_audio = render_markdown({
        'metadata': None,
        'media': {'duration': 5.0, 'has_audio': True},
        'frames': {'count': 0, 'ocr_status': {'status': 'ok', 'total_frames': 2, 'processed_frames': 2, 'failed_frames': 0, 'error': None}},
        'transcript': {'segments': []},
        'timeline': [],
        'style_profile': style_with_audio,
        'warnings': [],
    })
    assert '纯配乐/环境音展示（未检测到语音片段）' in md_with_audio
    assert '未检测到语音片段。' in md_with_audio
    assert '未检出明显高频常驻字' in md_with_audio
    assert 'OCR 识别器不可用' not in md_with_audio


# 无输入；验证阶段3扩展OCR：去重不重复识别、初步分镜1-2帧覆盖与单帧记录、预算超限标记、以及时间去偏常驻文字（含boundary_vfr短片回归），无返回值。
def test_15_stage3_extended_ocr_dedup_budget_audit_and_recurring_debias():
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    from PIL import Image
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        fdir = root / 'frames'; fdir.mkdir()
        # 构造 3 个初步分镜：
        # shot 1 (0.0-1.5s): 4 帧（前段），含 1 帧在 storyboard 中（验证重合不重复识别）
        # shot 2 (1.6-2.8s): 仅 1 个可用源帧（验证不足两帧如实记录）
        # shot 3 (3.0-3.9s): 5 帧（结尾 VFR 密帧，模拟 boundary_vfr_endcut_noaudio）
        frames = [
            {'index': 0, 't': 0.0, 'frame_index': 0, 'reasons': ['first'], 'scores': {'local_max_diff': 5.0}},
            {'index': 1, 't': 0.5, 'frame_index': 5, 'reasons': ['normal_motion'], 'scores': {'local_max_diff': 12.0}},
            {'index': 2, 't': 1.0, 'frame_index': 10, 'reasons': ['normal_motion'], 'scores': {'local_max_diff': 8.0}},
            {'index': 3, 't': 1.5, 'frame_index': 15, 'reasons': ['normal_motion'], 'scores': {'local_max_diff': 10.0}},
            {'index': 4, 't': 2.0, 'frame_index': 20, 'reasons': ['scene_after'], 'scores': {'scene_score': 0.5, 'local_max_diff': 30.0}},
            {'index': 5, 't': 3.0, 'frame_index': 30, 'reasons': ['scene_after'], 'scores': {'scene_score': 0.5, 'local_max_diff': 25.0}},
            {'index': 6, 't': 3.2, 'frame_index': 32, 'reasons': ['local_motion'], 'scores': {'local_max_diff': 20.0}},
            {'index': 7, 't': 3.4, 'frame_index': 34, 'reasons': ['local_motion'], 'scores': {'local_max_diff': 22.0}},
            {'index': 8, 't': 3.6, 'frame_index': 36, 'reasons': ['local_motion'], 'scores': {'local_max_diff': 18.0}},
            {'index': 9, 't': 3.85, 'frame_index': 39, 'reasons': ['last'], 'scores': {'local_max_diff': 15.0}},
        ]
        for fr in frames:
            Image.new('RGB', (320, 180), 'black').save(fdir / f"frame_{fr['frame_index']:08d}.jpg")
        timeline = [
            {'type': 'speech', 'start': 0.0, 'end': 1.55, 'text': '第一段', 'frames': frames[:4]},
            {'type': 'speech', 'start': 1.6, 'end': 2.8, 'text': '单帧段', 'frames': frames[4:5]},
            {'type': 'speech', 'start': 2.9, 'end': 3.9, 'text': '结尾密帧段', 'frames': frames[5:]},
        ]
        prelim = partition_preliminary_shots(timeline)
        assert len(prelim) == 3
        sb_items = [dict(frames[0], highlight_reason='首帧'), dict(frames[9], highlight_reason='末帧')]

        targets, audit = select_ocr_frames(sb_items, prelim, max_ocr_frames=96)
        # shot 1: 已有 frame_index 0, 补 1 帧 -> 2帧；shot 2: 仅 1 帧 -> 1帧；shot 3: 已有 39, 补 1 帧 -> 2帧；总去重目标 5 帧
        assert len(targets) == 5
        assert audit['budget_exceeded'] is False
        assert audit['coverage_sufficient'] is True
        assert audit['covered_shot_count'] == 3
        assert audit['fully_covered_shot_count'] == 3
        assert audit['single_frame_shot_count'] == 1
        assert audit['deduplicated_overlap_count'] >= 2
        assert any(s['shot_id'] == 2 and s['reason'] == 'fewer_than_2_available_frames' for s in audit['insufficient_shots'])

        # 验证超出预算标记与严禁宣称“每段已扫”
        targets_small, audit_small = select_ocr_frames(sb_items, prelim, max_ocr_frames=3)
        assert len(targets_small) == 3
        assert audit_small['budget_exceeded'] is True
        assert audit_small['coverage_sufficient'] is False
        assert any(s['reason'] == 'budget_exceeded' for s in audit_small['insufficient_shots'])

        # 构造伪 RapidOCR：验证仅调用 len(targets) 次、保留原始识别结果（含低分项）、
        # 且结尾密帧文字（占 4/5 帧）不被短片 0.25 误判为常驻，而全片固定角标被准确归入常驻
        call_counter = {'n': 0}
        # 故意把 targets 加上所有 10 帧以精确测试短片去偏回归（含 boundary_vfr_endcut_noaudio 场景）
        all_targets, all_audit = select_ocr_frames(frames, prelim, max_ocr_frames=96)
        assert len(all_targets) == 10

        class FakeRapidOCR:
            def __init__(self, **kwargs): pass
            def __call__(self, img, use_cls=False):
                idx = call_counter['n']
                call_counter['n'] += 1
                fi = all_targets[idx]['frame_index']
                box = [[20, 20], [220, 20], [220, 60], [20, 60]]
                items_out = [
                    (box, '全片固定角标', 0.95),
                    (box, '123', 0.99),  # 纯数字会被过滤但必须保留在 raw_results
                    (box, '低分噪声', 0.40),  # 低分会被过滤但必须保留在 raw_results
                ]
                if fi >= 30:
                    items_out.append((box, 'END CUT 3.5S', 0.94))
                if fi in (20, 30, 32):
                    items_out.append((box, 'SLOW5FPSSEG2', 0.93))
                return items_out, None

        fake_mod = SimpleNamespace(RapidOCR=FakeRapidOCR)
        # 注入 1 条重复 frame_index 验证 extract_keyframe_ocr 自身也会去重不重复调用引擎
        dup_input = list(all_targets) + [dict(all_targets[0], index=99)]
        with patch.dict(sys.modules, {'rapidocr_onnxruntime': fake_mod}):
            ocr_by_index, recurring, st = extract_keyframe_ocr(dup_input, fdir, {'duration': 3.9}, coverage_audit=all_audit)
        assert call_counter['n'] == 10, f"Expected 10 deduplicated OCR calls, got {call_counter['n']}"
        assert st['status'] == 'ok' and st['processed_frames'] == 10 and st['failed_frames'] == 0
        assert '全片固定角标' in recurring
        # 核心回归：短片后半段密集文字（5/10 帧 = 50%）不应被误归为全片常驻
        assert 'END CUT 3.5S' not in recurring
        assert 'SLOW5FPSSEG2' not in recurring
        assert 'END CUT 3.5S' in ocr_by_index[5]
        # 校验逐帧审计留痕字段完整性
        rec0 = st['frames'][0]
        assert rec0['frame_index'] == 0 and rec0['pts'] == 0.0 and rec0['status'] == 'ok'
        assert any(r['filter_reason'] == 'punctuation_or_digits_only' for r in rec0['raw_results'])
        assert any(r['filter_reason'] == 'low_confidence' for r in rec0['raw_results'])

        # 验证不均匀采样（20s 视频中 16-18s 连拍 10 帧出现同一阶段字幕）不误判为常驻
        skew_frames = (
            [{'index': i, 't': float(i * 4), 'frame_index': i, 'shot_ids': [1]} for i in range(4)]
            + [{'index': 4 + j, 't': 16.0 + j * 0.15, 'frame_index': 4 + j, 'shot_ids': [2]} for j in range(10)]
        )
        for sf in skew_frames:
            Image.new('RGB', (320, 180), 'black').save(fdir / f"frame_{sf['frame_index']:08d}.jpg")
        skew_idx = {'n': 0}
        class SkewOCR:
            def __init__(self, **kwargs): pass
            def __call__(self, img, use_cls=False):
                cur = skew_frames[skew_idx['n']]; skew_idx['n'] += 1
                box = [[20, 20], [220, 20], [220, 60], [20, 60]]
                out = [(box, '全片左上水印', 0.96)]
                if cur['t'] >= 16.0:
                    out.append((box, '爆燃高能瞬间', 0.95))
                return out, None
        with patch.dict(sys.modules, {'rapidocr_onnxruntime': SimpleNamespace(RapidOCR=SkewOCR)}):
            skew_ocr, skew_rec, _ = extract_keyframe_ocr(skew_frames, fdir, {'duration': 20.0})
        assert '全片左上水印' in skew_rec
        assert '爆燃高能瞬间' not in skew_rec, f"Skewed burst text wrongly marked recurring: {skew_rec}"
        assert '爆燃高能瞬间' in skew_ocr[4]

        # 验证超预算渲染绝不宣称“每段已扫”
        st_budget_md = dict(st, budget_exceeded=True, max_ocr_frames_budget=3, covered_shot_count=2, preliminary_shots_with_frames_count=3)
        st_budget_md.pop('frames', None)
        md_budget = render_markdown({
            'metadata': None, 'media': {'duration': 3.9, 'has_audio': False},
            'frames': {'count': 10, 'ocr_status': st_budget_md},
            'transcript': {'segments': []}, 'timeline': timeline,
            'style_profile': build_style_profile({'duration': 3.9, 'has_audio': False}, {'scene_cuts': []}, {'segments': []}, timeline, {}, sb_items),
        })
        assert '覆盖不足' in md_budget and '每段已扫' not in md_budget and '全分镜已扫' not in md_budget


# 无输入；建立带人工真值标注的短字幕样本，使用真实 RapidOCR 对比扩展前（仅 storyboard 24 帧）与扩展后召回、误识与耗时，无返回值。
def test_16_stage3_ocr_short_subtitle_recall_benchmark():
    import tempfile
    import time as _time
    from PIL import Image, ImageDraw
    from analyze import load_font
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        fdir = root / 'frames'; fdir.mkdir()
        # 构造 32 个硬切分镜（0..31s，每段 1 帧，共 32 帧 > storyboard 上限 24 帧）
        # select_storyboard_frames 在 32 个均匀硬切帧里会因 max_frames=24 必然丢弃若干中间镜头。
        items = []
        timeline = []
        font = load_font(42)
        gt_events = {}
        for i in range(32):
            img = Image.new('RGB', (960, 540), (20, 24, 32))
            draw = ImageDraw.Draw(img)
            subtitle = f"第{i+1:02d}幕测试字幕"
            draw.rectangle([120, 200, 840, 320], fill=(245, 245, 245))
            draw.text((220, 232), subtitle, fill=(10, 10, 10), font=font)
            img.save(fdir / f"frame_{i:08d}.jpg", quality=95)
            gt_events[i] = subtitle
            fr = {'index': i, 't': float(i) + 0.2, 'frame_index': i, 'reasons': ['scene_after'] if i > 0 else ['first'], 'scores': {'local_max_diff': 10.0, 'scene_score': 0.5 if i > 0 else None}}
            items.append(fr)
            timeline.append({'type': 'no-speech', 'start': float(i), 'end': float(i) + 0.9, 'text': '', 'frames': [fr]})

        sb_items = select_storyboard_frames(items, [{'frame_index': i, 't': float(i) + 0.2, 'score': 0.5} for i in range(1, 32)], [], 32.0, max_frames=24)
        sb_indices = {f['index'] for f in sb_items}
        missed_by_sb = sorted(set(range(32)) - sb_indices)
        assert len(missed_by_sb) >= 4, f"Expected storyboard(24) to drop >=4 shots from 32, got {missed_by_sb}"

        # 扩展前（仅 storyboard 24 帧）
        t0 = _time.perf_counter()
        ocr_before, _, st_before = extract_keyframe_ocr(sb_items, fdir, {'duration': 32.0})
        dur_before = _time.perf_counter() - t0
        assert st_before['status'] == 'ok'

        # 扩展后（preliminary shots 扩展覆盖全 32 个分镜）
        prelim = partition_preliminary_shots(timeline)
        ocr_targets, audit = select_ocr_frames(sb_items, prelim, max_ocr_frames=96)
        t1 = _time.perf_counter()
        ocr_after, _, st_after = extract_keyframe_ocr(ocr_targets, fdir, {'duration': 32.0}, coverage_audit=audit)
        dur_after = _time.perf_counter() - t1
        assert st_after['status'] == 'ok'

        recalled_before = sum(1 for idx, gt in gt_events.items() if any(gt in txt for txt in ocr_before.get(idx, [])))
        recalled_after = sum(1 for idx, gt in gt_events.items() if any(gt in txt for txt in ocr_after.get(idx, [])))
        assert recalled_before < len(gt_events), "Baseline storyboard OCR should miss dropped shots"
        assert recalled_after > recalled_before, f"Extended OCR should improve recall: before={recalled_before}, after={recalled_after}"
        # 所有被 storyboard 漏掉的片段都应在扩展后被召回
        newly_recalled = [idx for idx in missed_by_sb if any(gt_events[idx] in txt for txt in ocr_after.get(idx, []))]
        assert len(newly_recalled) >= len(missed_by_sb) - 1, f"Expected missed shots recalled, got {newly_recalled} vs {missed_by_sb}"
        assert dur_before > 0 and dur_after > 0


# 无输入；严格验证音频能量指标：静音/已知幅值/短脉冲夹具、立体声转单声道、末尾不足窗口、空区间、音视频起点偏移及客观报告措辞，无返回值。
def test_17_stage3_audio_energy_fixtures_and_boundaries():
    import math as _math
    import tempfile
    import wave
    import numpy as np

    sr = 16000
    # 1. 静音夹具
    silence = np.zeros(sr * 2, dtype=np.float64)
    res_sil = compute_audio_energy(silence, sr, total_duration=2.0, window_seconds=0.5)
    assert res_sil['status'] == 'ok' and res_sil['rms'] == 0.0 and res_sil['rms_dbfs'] == -120.0
    assert res_sil['peak'] == 0.0 and res_sil['window_silence_ratio'] == 1.0
    assert len(res_sil['low_energy_regions']) == 1 and res_sil['low_energy_regions'][0]['duration'] == 2.0
    assert res_sil['energy_surges'] == []

    # 2. 已知幅值正弦波夹具（A=0.5, RMS=0.5/sqrt(2)≈0.353553, -9.03 dBFS）
    t_ax = np.arange(sr * 2, dtype=np.float64) / sr
    sine = 0.5 * np.sin(2.0 * _math.pi * 1000.0 * t_ax)
    res_sine = compute_audio_energy(sine, sr, total_duration=2.0, window_seconds=0.5)
    assert abs(res_sine['rms'] - 0.5 / _math.sqrt(2.0)) < 0.002
    assert abs(res_sine['rms_dbfs'] - (-9.03)) < 0.06
    assert abs(res_sine['peak'] - 0.5) < 0.002
    assert res_sine['window_silence_ratio'] == 0.0
    assert res_sine['low_energy_regions'] == [] and res_sine['energy_surges'] == []

    # 3. 短脉冲夹具（2.0s 静音中 1.10-1.14s 含 A=0.9 正弦脉冲）
    pulse = np.zeros(sr * 2, dtype=np.float64)
    p_s, p_e = int(1.10 * sr), int(1.14 * sr)
    pulse[p_s:p_e] = 0.9 * np.sin(2.0 * _math.pi * 1000.0 * np.arange(p_e - p_s) / sr)
    res_pulse = compute_audio_energy(pulse, sr, total_duration=2.0, window_seconds=0.5)
    assert res_pulse['window_silence_ratio'] == 0.75
    assert abs(res_pulse['peak'] - 0.9) < 0.005
    assert len(res_pulse['energy_surges']) == 1 and res_pulse['energy_surges'][0]['start'] == 1.0
    pulse_win = res_pulse['windows'][2]
    assert pulse_win['subframe_silence_ratio'] >= 0.80

    # 4. 立体声转单声道 WAV（L=0.6, R=0.0 -> 单声道 0.3）
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        wav_stereo = Path(temp) / 'stereo.wav'
        n_st = sr
        left = np.full(n_st, int(round(0.6 * 32767)), dtype=np.int16)
        right = np.zeros(n_st, dtype=np.int16)
        interleaved = np.column_stack([left, right]).reshape(-1)
        with wave.open(str(wav_stereo), 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(interleaved.tobytes())
        pcm_st = read_wav_pcm(wav_stereo)
        assert pcm_st['channels'] == 2 and pcm_st['downmixed_to_mono'] is True
        res_st = compute_audio_energy(pcm_st['samples'], pcm_st['sample_rate'], total_duration=1.0, source_channels=pcm_st['channels'])
        assert res_st['downmixed_to_mono'] is True and abs(res_st['rms'] - 0.3) < 0.002
        stereo_2d = np.column_stack([np.full(sr, 0.6), np.zeros(sr)])
        res_st_2d = compute_audio_energy(stereo_2d, sr, total_duration=1.0)
        assert res_st_2d['downmixed_to_mono'] is True and abs(res_st_2d['rms'] - 0.3) < 0.002

    # 5. 末尾不足窗口（1.3s 音频按 0.5s 切窗 -> 第 3 窗长 0.3s 且按实际样本算 RMS 不掉幅度）
    trailing = np.full(int(round(1.3 * sr)), 0.5, dtype=np.float64)
    res_tr = compute_audio_energy(trailing, sr, total_duration=1.3, window_seconds=0.5)
    assert res_tr['has_partial_trailing_window'] is True
    assert len(res_tr['windows']) == 3
    last_w = res_tr['windows'][-1]
    assert last_w['is_partial'] is True and abs(last_w['duration'] - 0.3) < 1e-4
    assert abs(last_w['rms'] - 0.5) < 0.002

    # 6. 空区间与音视频起点偏移（+0.75s 正偏移与 -0.3s 负偏移、及完全空样本）
    pos_arr = np.full(int(round(1.25 * sr)), 0.4, dtype=np.float64)
    res_pos_off = compute_audio_energy(pos_arr, sr, total_duration=2.0, audio_offset=0.75, window_seconds=0.5)
    assert res_pos_off['av_start_offset_seconds'] == 0.75
    assert res_pos_off['empty_window_count'] == 1 and res_pos_off['valid_window_count'] == 3
    assert res_pos_off['windows'][0]['empty'] is True and res_pos_off['windows'][0]['rms'] is None
    assert res_pos_off['windows'][1]['has_samples'] is True and res_pos_off['windows'][1]['sample_count'] == int(0.25 * sr)
    assert abs(res_pos_off['windows'][1]['rms'] - 0.4) < 0.002

    neg_arr = np.full(int(round(1.3 * sr)), 0.25, dtype=np.float64)
    res_neg_off = compute_audio_energy(neg_arr, sr, total_duration=1.0, audio_offset=-0.3, window_seconds=0.5)
    assert res_neg_off['av_start_offset_seconds'] == -0.3
    assert abs(res_neg_off['covered_duration_seconds'] - 1.0) < 0.01

    res_empty = compute_audio_energy(np.zeros(0, dtype=np.float64), sr, total_duration=1.0)
    assert res_empty['status'] == 'empty_audio' and res_empty['valid_window_count'] == 0

    # 7. 报告只作有证据的客观描述，不凭能量推断平稳配乐或无人声
    style = build_style_profile({'duration': 2.0, 'has_audio': True}, {'scene_cuts': []}, {'segments': []}, [], {}, [], audio_energy=res_pulse)
    md = render_markdown({
        'metadata': None, 'media': {'duration': 2.0, 'has_audio': True},
        'frames': {'count': 0}, 'transcript': {'segments': []}, 'timeline': [],
        'audio_energy': res_pulse, 'style_profile': style,
    })
    assert '音频能量指标（客观波形统计）' in md
    assert '低能量区间' in md and '能量突增' in md and '窗口静音占比' in md
    assert '不单独判定配乐/人声' in md


# 无输入；验证阶段 4 结果目录增量追加 Clips：不重跑全片流程、校验身份与时间轴拒错片、支持两次/重叠追加与中途失败回滚，无返回值。
def test_18_stage4_append_clips_from_reprovided_original_and_validation():
    import json
    import subprocess
    import tempfile
    from unittest.mock import patch
    import analyze

    ffmpeg_bin = _find_media_tool('ffmpeg')
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        orig_vid = root / 'orig.mp4'
        wrong_dur_vid = root / 'wrong_dur.mp4'
        wrong_content_vid = root / 'wrong_content.mp4'
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=24:duration=3.0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(orig_vid),
        ], check=True)
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=24:duration=4.0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(wrong_dur_vid),
        ], check=True)
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'color=c=black:size=320x180:rate=24:duration=3.0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(wrong_content_vid),
        ], check=True)

        argv_init = [
            'analyze.py', '--video', str(orig_vid), '--clips', '00:00-00:01',
            '--results-root', str(root / 'results'), '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_init), patch.object(analyze, 'extract_keyframe_ocr', side_effect=lambda *args, **kwargs: ({}, [], {'status': 'unavailable', 'total_frames': 0, 'processed_frames': 0, 'failed_frames': 0, 'error': 'mocked'})):
            assert analyze.main() == 0
        assert not list((root / 'tmp').iterdir())
        result_dir = next((root / 'results').iterdir())
        init_json = json.loads((result_dir / 'analysis.json').read_text(encoding='utf-8'))
        assert init_json['schema_version'] == '1.8'
        assert len(init_json['frames']['exported_clips']) == 1
        clip1_path = result_dir / init_json['frames']['exported_clips'][0]['image']
        clip1_bytes = clip1_path.read_bytes()

        json_before = (result_dir / 'analysis.json').read_bytes()
        md_before = (result_dir / 'analysis.md').read_bytes()
        for bad_vid in (wrong_dur_vid, wrong_content_vid):
            argv_bad = [
                'analyze.py', '--append-clips', str(result_dir), '--video', str(bad_vid),
                '--clips', '00:01-00:02', '--tmp-root', str(root / 'tmp'),
            ]
            with patch.object(sys, 'argv', argv_bad):
                assert analyze.main() == 1
            assert (result_dir / 'analysis.json').read_bytes() == json_before
            assert (result_dir / 'analysis.md').read_bytes() == md_before
            assert clip1_path.read_bytes() == clip1_bytes

        data_s3 = json.loads(json_before.decode('utf-8'))
        data_s3['schema_version'] = '1.6'
        data_s3['media'].pop('sha256', None)
        (result_dir / 'analysis.json').write_text(json.dumps(data_s3, ensure_ascii=False, indent=2), encoding='utf-8')
        argv_bad_s3 = [
            'analyze.py', '--append-clips', str(result_dir), '--video', str(wrong_content_vid),
            '--clips', '00:01-00:02', '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_bad_s3):
            assert analyze.main() == 1
        (result_dir / 'analysis.json').write_bytes(json_before)

        def _forbid(*args, **kwargs):
            raise AssertionError('Full-video stage should NOT run during --append-clips')

        argv_ap1 = [
            'analyze.py', '--append-clips', str(result_dir), '--video', str(orig_vid),
            '--clips', '00:00-00:02', '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_ap1), \
             patch.object(analyze, 'transcribe', _forbid), \
             patch.object(analyze, 'extract_keyframe_ocr', _forbid), \
             patch.object(analyze, 'make_storyboard', _forbid), \
             patch.object(analyze, 'make_sheets', _forbid), \
             patch.object(analyze, 'make_transitions_sheet', _forbid):
            assert analyze.main() == 0

        argv_ap2 = [
            'analyze.py', '--append-clips', str(result_dir), '--video', str(orig_vid),
            '--clips', '00:01-00:03', '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_ap2), \
             patch.object(analyze, 'transcribe', _forbid), \
             patch.object(analyze, 'extract_keyframe_ocr', _forbid):
            assert analyze.main() == 0

        after2 = json.loads((result_dir / 'analysis.json').read_text(encoding='utf-8'))
        clips_all = after2['frames']['exported_clips']
        assert len(clips_all) == 3
        assert [c['id'] for c in clips_all] == [1, 2, 3]
        assert clips_all[1]['appended'] is True and clips_all[1]['append_batch_id'] == 1
        assert 1 in (clips_all[1].get('overlaps_clip_ids') or [])
        assert clips_all[2]['appended'] is True and clips_all[2]['append_batch_id'] == 2
        assert clip1_path.read_bytes() == clip1_bytes
        for c in clips_all:
            assert (result_dir / c['image']).is_file()
        hist = after2['clip_append_history']
        assert len(hist) == 2
        assert 'fetch' in hist[0]['skipped_stages'] and 'dynamic_frames' in hist[0]['skipped_stages']
        assert 'asr' in hist[0]['skipped_stages'] and 'ocr' in hist[0]['skipped_stages']
        assert after2['timings'] == init_json['timings']

        pre_fail_json = (result_dir / 'analysis.json').read_bytes()
        pre_fail_md = (result_dir / 'analysis.md').read_bytes()
        pre_fail_files = sorted(p.name for p in (result_dir / 'clips').iterdir())
        argv_fail = [
            'analyze.py', '--append-clips', str(result_dir), '--video', str(orig_vid),
            '--clips', '00:00-00:01,00:01-00:02', '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_fail), patch.object(analyze, '_persist_reports', side_effect=RuntimeError('simulated persist boom')):
            assert analyze.main() == 1
        assert sorted(p.name for p in (result_dir / 'clips').iterdir()) == pre_fail_files
        assert (result_dir / 'analysis.json').read_bytes() == pre_fail_json
        assert (result_dir / 'analysis.md').read_bytes() == pre_fail_md
        assert not list((root / 'tmp').iterdir())


# 无输入；验证阶段 4 可选低码率代理：记录参数与时间映射、PTS 偏差<=1帧、图面/字段标识画质来源、降帧代理不承诺补回原片、磁盘预算与清理顺序，无返回值。
def test_19_stage4_optional_proxy_pts_labeling_budget_and_cleanup_order():
    import json
    import shutil as _shutil
    import subprocess
    import tempfile
    from unittest.mock import patch
    from PIL import Image
    import analyze

    ffmpeg_bin = _find_media_tool('ffmpeg')
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        hd_vid = root / 'hd_24fps.mp4'
        subprocess.run([
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc=size=1280x720:rate=24:duration=3.0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(hd_vid),
        ], check=True)

        events = []
        orig_persist = analyze._persist_reports
        orig_rmtree = _shutil.rmtree
        orig_cmd = analyze.command

        def spy_persist(rd, d):
            events.append(('persist', bool((rd / 'proxy' / 'proxy.mp4').is_file()), bool((d.get('proxy') or {}).get('available'))))
            return orig_persist(rd, d)

        def spy_rmtree(p, *args, **kwargs):
            events.append(('cleanup_tmp', Path(p).exists()))
            return orig_rmtree(p, *args, **kwargs)

        def fake_fetch_cmd(args, timeout=None):
            if str(args[1]).endswith('douyin_fetch.py'):
                out = Path(args[args.index('--out-dir') + 1])
                dl_vid = out / 'downloaded_orig.mp4'
                _shutil.copy2(hd_vid, dl_vid)
                res = {'status': 'completed', 'video_path': str(dl_vid), 'metadata': {'aweme_id': '999888'}, 'download': {'warnings': []}}
                Path(args[args.index('--result-json') + 1]).write_text(json.dumps(res), encoding='utf-8')
                return ''
            return orig_cmd(args, timeout=timeout)

        argv_proxy = [
            'analyze.py', 'https://v.douyin.com/fake999/', '--save-proxy', '--clips', '00:00-00:02',
            '--results-root', str(root / 'results'), '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_proxy), \
             patch.object(analyze, 'command', fake_fetch_cmd), \
             patch.object(analyze, 'extract_keyframe_ocr', side_effect=lambda *args, **kwargs: ({}, [], {'status': 'unavailable', 'total_frames': 0, 'processed_frames': 0, 'failed_frames': 0, 'error': 'mocked'})), \
             patch.object(analyze, '_persist_reports', side_effect=spy_persist), \
             patch.object(analyze.shutil, 'rmtree', side_effect=spy_rmtree):
            assert analyze.main() == 0

        p_idx = next(i for i, e in enumerate(events) if e[0] == 'persist')
        c_idx = next(i for i, e in enumerate(events) if e[0] == 'cleanup_tmp')
        assert events[p_idx] == ('persist', True, True) and p_idx < c_idx

        res_dir = next((root / 'results').iterdir())
        data = json.loads((res_dir / 'analysis.json').read_text(encoding='utf-8'))
        px = data['proxy']
        assert px['available'] is True and px['downsampled'] is False and px['can_recover_original_motion'] is True
        assert px['width'] == 640 and px['height'] == 360
        assert px['generation_params']['fps_mode'] == 'passthrough'
        assert px['time_mapping']['within_one_source_frame_interval'] is True
        assert px['time_mapping']['max_abs_pts_diff_seconds'] <= (1.0 / 24.0 + 1e-4)

        argv_ap_px = [
            'analyze.py', '--append-clips', str(res_dir), '--clips', '00:00-00:02',
            '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_ap_px):
            assert analyze.main() == 0
        after_px = json.loads((res_dir / 'analysis.json').read_text(encoding='utf-8'))
        c_orig, c_px = after_px['frames']['exported_clips'][0], after_px['frames']['exported_clips'][1]
        assert c_px['source_kind'] == 'proxy' and c_px['time_source'] == 'proxy_ffprobe_pts'
        assert '低码率代理抽帧' in c_px['source_quality_label']
        assert abs(c_px['actual_first_pts'] - c_orig['actual_first_pts']) <= (1.0 / 24.0 + 1e-4)
        assert abs(c_px['actual_last_pts'] - c_orig['actual_last_pts']) <= (1.0 / 24.0 + 1e-4)
        px_img = res_dir / c_px['image']
        orig_img = res_dir / c_orig['image']
        with Image.open(px_img) as im_px, Image.open(orig_img) as im_orig:
            assert im_px.height == im_orig.height + 34

        argv_ds = [
            'analyze.py', '--video', str(hd_vid), '--save-proxy', '--proxy-fps', '4',
            '--results-root', str(root / 'results_ds'), '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_ds), \
             patch.object(analyze, 'extract_keyframe_ocr', side_effect=lambda *args, **kwargs: ({}, [], {'status': 'unavailable', 'total_frames': 0, 'processed_frames': 0, 'failed_frames': 0, 'error': 'mocked'})):
            assert analyze.main() == 0
        ds_dir = next((root / 'results_ds').iterdir())
        ds_data = json.loads((ds_dir / 'analysis.json').read_text(encoding='utf-8'))
        assert ds_data['proxy']['downsampled'] is True and ds_data['proxy']['can_recover_original_motion'] is False
        argv_ap_ds = ['analyze.py', '--append-clips', str(ds_dir), '--clips', '00:00-00:02', '--tmp-root', str(root / 'tmp')]
        with patch.object(sys, 'argv', argv_ap_ds):
            assert analyze.main() == 0
        ds_after = json.loads((ds_dir / 'analysis.json').read_text(encoding='utf-8'))
        ds_clip = ds_after['frames']['exported_clips'][0]
        assert ds_clip['can_recover_original_motion'] is False
        assert 'proxy_downsampled_cannot_recover_original_motion' in ds_clip['coverage_insufficient_reasons']
        assert '不可补回原片' in ds_clip['source_quality_label']

        argv_over_budget = [
            'analyze.py', '--video', str(hd_vid), '--save-proxy', '--max-proxy-mb', '0.001',
            '--results-root', str(root / 'results_budget'), '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_over_budget), \
             patch.object(analyze, 'extract_keyframe_ocr', side_effect=lambda *args, **kwargs: ({}, [], {'status': 'unavailable', 'total_frames': 0, 'processed_frames': 0, 'failed_frames': 0, 'error': 'mocked'})):
            assert analyze.main() == 1
        b_dir = next((root / 'results_budget').iterdir())
        b_data = json.loads((b_dir / 'analysis.json').read_text(encoding='utf-8'))
        assert b_data['status'] == 'error' and b_data['failed_stage'] == 'proxy'
        assert not (b_dir / 'proxy' / 'proxy.mp4').exists()

        ps1 = Path(__file__).resolve().parents[1] / 'analyze.ps1'
        p_no_clips = subprocess.run([
            'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ps1),
            '-AppendClips', str(res_dir),
        ], capture_output=True, text=True)
        assert p_no_clips.returncode == 2
        p_both = subprocess.run([
            'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ps1),
            'https://v.douyin.com/test/', '-AppendClips', str(res_dir), '-Clips', '00:00-00:01',
        ], capture_output=True, text=True)
        assert p_both.returncode == 2


# 无输入；验证严格域名解析路由、欺骗绕过拦截、未知域名拒绝与抖音失败禁止静默切换 yt-dlp，无返回值。
def test_20_stage6_url_routing_strict_host_and_no_silent_fallback():
    import json
    import tempfile
    from unittest.mock import patch
    import analyze
    import media_router

    r_dy = media_router.classify_url_route('https://www.douyin.com/video/7699064323272802491')
    assert r_dy['platform'] == 'douyin' and r_dy['downloader'] == 'douyin_browser_mcp' and r_dy['route_kind'] == 'douyin_standard'
    r_dy_s = media_router.classify_url_route('https://v.douyin.com/uvhJnYkkvmo/')
    assert r_dy_s['platform'] == 'douyin' and r_dy_s['downloader'] == 'douyin_browser_mcp' and r_dy_s['route_kind'] == 'douyin_short'
    r_bi = media_router.classify_url_route('https://www.bilibili.com/video/BV1xx411c7mD')
    assert r_bi['platform'] == 'bilibili' and r_bi['downloader'] == 'yt_dlp' and r_bi['route_kind'] == 'bilibili_standard'
    r_bi_s = media_router.classify_url_route('https://b23.tv/abc123')
    assert r_bi_s['platform'] == 'bilibili' and r_bi_s['route_kind'] == 'bilibili_short'
    r_yt = media_router.classify_url_route('https://www.youtube.com/watch?v=dQw4w9WgXcQ&token=SECRET')
    assert r_yt['platform'] == 'youtube' and r_yt['downloader'] == 'yt_dlp' and r_yt['route_kind'] == 'youtube_standard'
    r_yt_s = media_router.classify_url_route('https://youtu.be/dQw4w9WgXcQ')
    assert r_yt_s['platform'] == 'youtube' and r_yt_s['route_kind'] == 'youtube_short'

    assert media_router.sanitize_replay_url('https://www.youtube.com/watch?v=dQw4w9WgXcQ&token=SECRET&cookie=abc') == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

    for bad in (
        'https://evil.com/bilibili.com/video/BV1xx',
        'https://evilbilibili.com/video/BV1xx',
        'https://douyin.com.evil.org/video/123',
        'https://example.com/video/123',
        'file:///C:/test.mp4',
        'http://localhost:8080/video',
        'not_a_url',
    ):
        try:
            media_router.classify_url_route(bad)
            raise AssertionError(f'Expected UnsupportedRouteError for {bad}')
        except media_router.UnsupportedRouteError:
            pass

    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        called = []

        def spy_command(args, timeout=None):
            called.append(list(args))
            raise RuntimeError('douyin fetch boom')

        argv_unknown = [
            'analyze.py', 'https://evil.com/bilibili.com/video/123',
            '--results-root', str(root / 'res_u'), '--tmp-root', str(root / 'tmp_u'),
        ]
        with patch.object(sys, 'argv', argv_unknown), patch.object(analyze, 'command', spy_command):
            assert analyze.main() == 1
        assert called == []
        u_dir = next((root / 'res_u').iterdir())
        u_data = json.loads((u_dir / 'analysis.json').read_text(encoding='utf-8'))
        assert u_data['status'] == 'error' and u_data['failed_stage'] == 'preflight'
        assert 'Unsupported URL host' in (u_data.get('error') or '')

        fake_fetch_py = root / 'fake_fetch_py.exe'
        fake_fetch_py.write_text('stub', encoding='utf-8')
        argv_dy_fail = [
            'analyze.py', 'https://v.douyin.com/uvhJnYkkvmo/',
            '--fetch-python', str(fake_fetch_py),
            '--results-root', str(root / 'res_dy'), '--tmp-root', str(root / 'tmp_dy'),
        ]
        with patch.object(sys, 'argv', argv_dy_fail), patch.object(analyze, 'command', spy_command):
            assert analyze.main() == 1
        assert len(called) == 1 and str(called[0][1]).endswith('douyin_fetch.py')


# 无输入；验证 yt-dlp 可选缺失隔离、统一媒体接口、元数据缺失原因、超时/故障脱敏与中文路径处理，无返回值。
def test_21_stage6_optional_ytdlp_unified_schema_missing_metadata_and_no_affect():
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    from unittest.mock import patch
    import analyze
    import media_router

    ffmpeg = shutil.which('ffmpeg') or r'D:\ffmpeg\bin\ffmpeg.exe'
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp) / '中文 路径_测试'
        root.mkdir(parents=True)
        src_vid = root / '测试 样片.mp4'
        subprocess.run(
            [ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=320x240:r=10:d=1.2', '-pix_fmt', 'yuv420p', str(src_vid)],
            check=True,
        )

        # 1. 显式模拟 yt-dlp 缺失时，外站拦截明确，而本地入口完全不受影响。
        old_env = os.environ.get('DOUYIN_ANALYZER_DISABLE_YTDLP')
        os.environ['DOUYIN_ANALYZER_DISABLE_YTDLP'] = '1'
        try:
            argv_miss = [
                'analyze.py', 'https://www.bilibili.com/video/BV1test6',
                '--results-root', str(root / 'res_miss'), '--tmp-root', str(root / 'tmp_miss'),
            ]
            with patch.object(sys, 'argv', argv_miss):
                assert analyze.main() == 1
            m_dir = next((root / 'res_miss').iterdir())
            m_data = json.loads((m_dir / 'analysis.json').read_text(encoding='utf-8'))
            assert m_data['status'] == 'error' and 'yt-dlp' in (m_data.get('error') or '').lower()

            argv_local = [
                'analyze.py', '--video', str(src_vid),
                '--results-root', str(root / 'res_local'), '--tmp-root', str(root / 'tmp_local'),
            ]
            with patch.object(sys, 'argv', argv_local):
                assert analyze.main() == 0
            l_dir = next((root / 'res_local').iterdir())
            l_data = json.loads((l_dir / 'analysis.json').read_text(encoding='utf-8'))
            assert l_data['status'] == 'ok' and l_data['input']['platform'] == 'local'
            assert l_data['input']['platform_item_id'] and l_data['input']['task_unique_key'].startswith('local:')
        finally:
            if old_env is None:
                os.environ.pop('DOUYIN_ANALYZER_DISABLE_YTDLP', None)
            else:
                os.environ['DOUYIN_ANALYZER_DISABLE_YTDLP'] = old_env

        # 2. 模拟 yt-dlp 下载成功（含部分元数据缺失与中文路径）。
        fake_ytdlp = root / 'yt-dlp.exe'
        fake_ytdlp.write_text('stub', encoding='utf-8')
        orig_command = analyze.command

        def fake_ext_command(args, timeout=None):
            if str(args[0]) == str(fake_ytdlp.resolve()):
                out_tpl = args[args.index('-o') + 1]
                out_dir = Path(out_tpl).parent
                shutil.copy2(src_vid, out_dir / 'ytdlp_BV1test6.mp4')
                info = {
                    'id': 'BV1test6',
                    'title': 'B站外站测试视频',
                    'uploader': None,
                    'duration': 1.2,
                    'width': 320,
                    'height': 240,
                    'upload_date': '20261114',
                    'webpage_url': 'https://www.bilibili.com/video/BV1test6?spm_id_from=333.1007&token=SECRET',
                }
                (out_dir / 'ytdlp_BV1test6.info.json').write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
                return ''
            return orig_command(args, timeout=timeout)

        argv_ext = [
            'analyze.py', 'https://b23.tv/short6?token=SECRET',
            '--ytdlp-bin', str(fake_ytdlp),
            '--results-root', str(root / 'res_ext'), '--tmp-root', str(root / 'tmp_ext'),
        ]
        with patch.object(sys, 'argv', argv_ext), patch.object(analyze, 'command', fake_ext_command):
            assert analyze.main() == 0
        e_dir = next((root / 'res_ext').iterdir())
        assert e_dir.name.startswith('bilibili_BV1test6-')
        e_json_txt = (e_dir / 'analysis.json').read_text(encoding='utf-8')
        assert 'SECRET' not in e_json_txt
        e_data = json.loads(e_json_txt)
        assert e_data['schema_version'] == '1.8'
        assert e_data['input']['platform'] == 'bilibili'
        assert e_data['input']['platform_item_id'] == 'BV1test6'
        assert e_data['input']['downloader'] == 'yt_dlp'
        assert e_data['input']['task_unique_key'] == 'bilibili:BV1test6'
        assert e_data['metadata']['title'] == 'B站外站测试视频'
        assert e_data['metadata_sources']['title'] == 'yt_dlp'
        assert 'author_nickname' in e_data['missing_metadata']
        e_md = (e_dir / 'analysis.md').read_text(encoding='utf-8')
        assert 'bilibili' in e_md and 'BV1test6' in e_md and '元数据缺失说明' in e_md

        # 3. 验证 yt-dlp 超时故障不切换抖音下载器并脱敏落盘。
        def fake_ext_timeout(args, timeout=None):
            if str(args[0]) == str(fake_ytdlp.resolve()):
                raise RuntimeError('yt-dlp.exe timed out after 300s')
            return orig_command(args, timeout=timeout)

        argv_to = [
            'analyze.py', 'https://youtu.be/dQw4w9WgXcQ',
            '--ytdlp-bin', str(fake_ytdlp),
            '--results-root', str(root / 'res_to'), '--tmp-root', str(root / 'tmp_to'),
        ]
        with patch.object(sys, 'argv', argv_to), patch.object(analyze, 'command', fake_ext_timeout):
            assert analyze.main() == 1
        to_dir = next((root / 'res_to').iterdir())
        to_data = json.loads((to_dir / 'analysis.json').read_text(encoding='utf-8'))
        assert to_data['status'] == 'error' and to_data['failed_stage'] == 'fetch'
        assert 'timed out' in (to_data.get('error') or '')


# 无输入；验证串行批量、单条失败隔离、四态区分、跨平台标识防重、零覆盖断点续跑与极简可追溯摘要，无返回值。
def test_22_stage6_serial_batch_isolation_degraded_resume_and_summary():
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    from unittest.mock import patch
    import analyze
    import batch_analyze

    ffmpeg = shutil.which('ffmpeg') or r'D:\ffmpeg\bin\ffmpeg.exe'
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        dir_a = root / '目录 甲'
        dir_b = root / '目录 乙'
        dir_a.mkdir()
        dir_b.mkdir()
        # 故意起相同数字文件名 123.mp4，测试跨目录/跨来源任务标识防冲突。
        vid_a = dir_a / '123.mp4'
        vid_b = dir_b / '123.mp4'
        subprocess.run(
            [ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=red:s=240x180:r=10:d=1.0', '-pix_fmt', 'yuv420p', str(vid_a)],
            check=True,
        )
        subprocess.run(
            [ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=green:s=240x180:r=10:d=1.0', '-pix_fmt', 'yuv420p', str(vid_b)],
            check=True,
        )

        # 用真实的本地子进程跑一个包含：成功、失败（未知域名）、中文路径成功的串行批量清单。
        batch_txt = root / 'tasks.txt'
        batch_txt.write_text(
            f"# 批量清单\n{vid_a} | clips=00:00-00:01\nhttps://unknown-site.invalid/video/999\n{vid_b}\n",
            encoding='utf-8',
        )
        argv_batch = [
            'analyze.py', '--batch-file', str(batch_txt),
            '--results-root', str(root / 'results'), '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_batch):
            code = analyze.main()
        assert code == 1  # 第2条失败，总退出码非 0，但不中断第3条运行

        batch_dirs = [p for p in (root / 'results').iterdir() if p.is_dir() and p.name.startswith('batch-')]
        assert len(batch_dirs) == 1
        bdir1 = batch_dirs[0]
        idx1 = json.loads((bdir1 / 'batch_index.json').read_text(encoding='utf-8'))
        assert idx1['execution_mode'] == 'serial'
        assert idx1['counts'] == {'total': 3, 'success': 2, 'degraded': 0, 'failed': 1, 'not_executed': 0}
        t1, t2, t3 = idx1['tasks']
        assert t1['status_category'] == 'success' and t3['status_category'] == 'success'
        assert t2['status_category'] == 'failed'
        assert t1['task_uid'] != t3['task_uid'] and t1['canonical_task_id'] != t3['canonical_task_id']
        assert Path(t1['analysis_md']).is_file() and Path(t3['analysis_json']).is_file()

        sum_md1 = (bdir1 / 'batch_summary.md').read_text(encoding='utf-8')
        assert 'VLM=False' in sum_md1 and '123' in sum_md1 and 'Unsupported URL host' in sum_md1
        t1_mtime_before = Path(t1['analysis_json']).stat().st_mtime_ns

        # 验证 classify_task_outcome 精确区分 degraded（例如 OCR 不可用 / cleanup 失败）
        deg_cat, _, deg_reasons, _ = batch_analyze.classify_task_outcome(
            0, Path(t1['result_dir']),
            {'status': 'completed_with_cleanup_failure', 'frames': {'ocr_status': {'status': 'unavailable', 'failed_frames': 1, 'total_frames': 1}}, 'warnings': []},
        )
        assert deg_cat == 'degraded' and len(deg_reasons) >= 2

        # 修改旧批次索引里的第2条为可成功本地视频，用 --resume-batch 验证：
        # 1) 不重跑、不覆盖 t1 / t3 已有成果；2) 生成全新 batch- 目录并重跑 t2。
        vid_c = root / '补救视频.mp4'
        shutil.copy2(vid_a, vid_c)
        idx1['tasks'][1]['spec'] = {'task_index': 2, 'video': str(vid_c)}
        (bdir1 / 'batch_index.json').write_text(json.dumps(idx1, ensure_ascii=False, indent=2), encoding='utf-8')

        argv_resume = [
            'analyze.py', '--resume-batch', str(bdir1),
            '--results-root', str(root / 'results'), '--tmp-root', str(root / 'tmp'),
        ]
        with patch.object(sys, 'argv', argv_resume):
            code2 = analyze.main()
        assert code2 == 0
        batch_dirs2 = sorted(p for p in (root / 'results').iterdir() if p.is_dir() and p.name.startswith('batch-'))
        assert len(batch_dirs2) == 2
        bdir2 = [p for p in batch_dirs2 if p != bdir1][0]
        idx2 = json.loads((bdir2 / 'batch_index.json').read_text(encoding='utf-8'))
        assert idx2['counts'] == {'total': 3, 'success': 3, 'degraded': 0, 'failed': 0, 'not_executed': 0}
        assert idx2['tasks'][0]['reused_existing_result'] is True and idx2['tasks'][2]['reused_existing_result'] is True
        assert idx2['tasks'][1]['reused_existing_result'] is False
        assert Path(t1['analysis_json']).stat().st_mtime_ns == t1_mtime_before

        # 验证 --stop-on-error 会将未执行条目准确标记为 not_executed
        stop_json = root / 'stop.json'
        stop_json.write_text(json.dumps([
            {'url': 'https://unknown-site.invalid/fail1'},
            {'video': str(vid_a)},
        ], ensure_ascii=False), encoding='utf-8')
        argv_stop = [
            'analyze.py', '--batch-file', str(stop_json), '--stop-on-error',
            '--results-root', str(root / 'results_stop'), '--tmp-root', str(root / 'tmp_stop'),
        ]
        with patch.object(sys, 'argv', argv_stop):
            assert analyze.main() == 1
        s_bdir = next((root / 'results_stop').iterdir())
        s_idx = json.loads((s_bdir / 'batch_index.json').read_text(encoding='utf-8'))
        assert s_idx['counts'] == {'total': 2, 'success': 0, 'degraded': 0, 'failed': 1, 'not_executed': 1}
        assert s_idx['tasks'][1]['status_category'] == 'not_executed'

        # 验证 analyze.ps1 批量互斥参数校验
        ps1 = Path(__file__).resolve().parents[1] / 'analyze.ps1'
        p_mix = subprocess.run([
            'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ps1),
            'https://v.douyin.com/test/', '-BatchFile', str(batch_txt),
        ], capture_output=True, text=True)
        assert p_mix.returncode == 2


class AnalyzerTests(unittest.TestCase):
    def test_01_alignment(self): test_alignment()
    def test_02_security_and_time(self): test_security_and_time()
    def test_03_markdown_nulls(self): test_markdown_nulls()
    def test_04_fetch_pipeline_integration(self): test_fetch_pipeline_integration()
    def test_05_no_speech_gaps_split_at_empty_speech(self): test_no_speech_gaps_split_at_empty_speech()
    def test_06_zero_drop_sheets_and_transitions(self): test_zero_drop_sheets_and_transitions()
    def test_07_parse_clip_ranges(self): test_parse_clip_ranges()
    def test_08_build_smart_shots_merge_recomputes_fields(self): test_build_smart_shots_merge_recomputes_fields()
    def test_09_missing_frame_image_fails_explicitly(self): test_missing_frame_image_fails_explicitly()
    def test_10_export_clips_pts_rate_limit_supplement_and_pagination(self): test_export_clips_pts_rate_limit_supplement_and_pagination()
    def test_11_p2_fixes_end_frames_ocr_timeout_and_markdown(self): test_p2_fixes_end_frames_ocr_timeout_and_markdown()
    def test_12_stage1_corrupt_persist_cleanup_budgets_and_windows_paths(self): test_stage1_corrupt_persist_cleanup_budgets_and_windows_paths()
    def test_13_real_vfr_dynamic_frames_timeout_and_ps1(self): test_real_vfr_dynamic_frames_timeout_and_ps1()
    def test_14_stage2_no_audio_notext_and_recognizer_unavailable_distinction(self): test_stage2_no_audio_notext_and_recognizer_unavailable_distinction()
    def test_15_stage3_extended_ocr_dedup_budget_audit_and_recurring_debias(self): test_15_stage3_extended_ocr_dedup_budget_audit_and_recurring_debias()
    def test_16_stage3_ocr_short_subtitle_recall_benchmark(self): test_16_stage3_ocr_short_subtitle_recall_benchmark()
    def test_17_stage3_audio_energy_fixtures_and_boundaries(self): test_17_stage3_audio_energy_fixtures_and_boundaries()
    def test_18_stage4_append_clips_from_reprovided_original_and_validation(self): test_18_stage4_append_clips_from_reprovided_original_and_validation()
    def test_19_stage4_optional_proxy_pts_labeling_budget_and_cleanup_order(self): test_19_stage4_optional_proxy_pts_labeling_budget_and_cleanup_order()
    def test_20_stage6_url_routing_strict_host_and_no_silent_fallback(self): test_20_stage6_url_routing_strict_host_and_no_silent_fallback()
    def test_21_stage6_optional_ytdlp_unified_schema_missing_metadata_and_no_affect(self): test_21_stage6_optional_ytdlp_unified_schema_missing_metadata_and_no_affect()
    def test_22_stage6_serial_batch_isolation_degraded_resume_and_summary(self): test_22_stage6_serial_batch_isolation_degraded_resume_and_summary()


if __name__ == '__main__':
    test_alignment(); test_security_and_time(); test_markdown_nulls()
    test_fetch_pipeline_integration(); test_no_speech_gaps_split_at_empty_speech()
    test_zero_drop_sheets_and_transitions(); test_parse_clip_ranges()
    test_build_smart_shots_merge_recomputes_fields()
    test_missing_frame_image_fails_explicitly()
    test_export_clips_pts_rate_limit_supplement_and_pagination()
    test_p2_fixes_end_frames_ocr_timeout_and_markdown()
    test_stage1_corrupt_persist_cleanup_budgets_and_windows_paths()
    test_real_vfr_dynamic_frames_timeout_and_ps1()
    test_stage2_no_audio_notext_and_recognizer_unavailable_distinction()
    test_15_stage3_extended_ocr_dedup_budget_audit_and_recurring_debias()
    test_16_stage3_ocr_short_subtitle_recall_benchmark()
    test_17_stage3_audio_energy_fixtures_and_boundaries()
    test_18_stage4_append_clips_from_reprovided_original_and_validation()
    test_19_stage4_optional_proxy_pts_labeling_budget_and_cleanup_order()
    test_20_stage6_url_routing_strict_host_and_no_silent_fallback()
    test_21_stage6_optional_ytdlp_unified_schema_missing_metadata_and_no_affect()
    test_22_stage6_serial_batch_isolation_degraded_resume_and_summary()
    print('22 tests passed')
