# 下载器桥接：使用下载器自身 venv，在进程内捕获公开元数据并安全输出。
"""Download through the installed MCP package and collect public page metadata."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import unquote

FIELDS = ('aweme_id', 'canonical_url', 'title', 'desc', 'hashtags', 'author_nickname',
          'author_unique_id', 'author_sec_uid', 'create_time', 'create_time_iso',
          'duration_seconds', 'digg_count', 'comment_count', 'collect_count', 'share_count',
          'play_count', 'music_title', 'music_author', 'video_width', 'video_height', 'page_title')
MAX_BODY = 2 * 1024 * 1024
URL_RE = re.compile(r'https?://[^\s\"\'<>]+', re.I)
SECRET_RE = re.compile(r'(?i)(cookie|token|authorization|bearer|signature|x-bogus|msToken)\s*[:=]')


# 输入任意值；返回通过安全规则处理的文本，拒绝可能包含敏感信息的内容。
def safe_text(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or SECRET_RE.search(value):
        return None
    # Even a URL without a query can carry a signature in its path. The only
    # exported URL is the canonical URL constructed from a numeric ID below.
    if URL_RE.search(value):
        return None
    return value[:10000]


# 输入 Unix 时间戳；返回东八区 ISO 时间，无效时返回空值。
def create_time_iso(value):
    try:
        return datetime.fromtimestamp(int(value), timezone(timedelta(hours=8))).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


# 输入候选元数据字典；仅返回白名单内且通过类型和内容过滤的字段。
def sanitize_metadata(values):
    """Allow only the documented public fields and safe scalar values."""
    result = {}
    for key, value in values.items():
        if key not in FIELDS or any(s in key.lower() for s in ('cookie', 'token', 'sign')):
            continue
        if key == 'hashtags':
            result[key] = [v for item in value if (v := safe_text(item))] if isinstance(value, list) else None
        elif isinstance(value, str):
            result[key] = safe_text(value)
        elif value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)):
            result[key] = value
        else:
            result[key] = None
    return result


# 输入嵌套对象和路径；安全取值，路径不存在时返回空值。
def _get(obj, *path):
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


# 输入嵌套对象和目标作品 ID；遍历并产出匹配作品对象。
def _aweme_objects(obj, target):
    """Find only objects explicitly identifying the requested video."""
    stack = [obj]
    visited = 0
    while stack and visited < 100000:
        node = stack.pop()
        visited += 1
        if isinstance(node, dict):
            if str(node.get('aweme_id', node.get('awemeId', node.get('item_id', '')))) == target:
                yield node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


# 输入作品对象；返回提取出的候选元数据字典。
def _extract(obj):
    author = obj.get('author') or {}
    stats = obj.get('statistics') or obj.get('stats') or {}
    video = obj.get('video') or {}
    music = obj.get('music') or {}
    desc = obj.get('desc') or obj.get('description')
    hashtags = []
    for item in (obj.get('text_extra') or []):
        if isinstance(item, dict) and item.get('hashtag_name'):
            hashtags.append(str(item['hashtag_name']))
    if not hashtags and isinstance(desc, str):
        hashtags = re.findall(r'#([^\s#]+)', desc)
    duration = video.get('duration') if isinstance(video, dict) else None
    if isinstance(duration, (int, float)):
        duration /= 1000
    return {'aweme_id': str(obj.get('aweme_id') or obj.get('awemeId') or obj.get('item_id')),
            'title': desc, 'desc': desc, 'hashtags': hashtags or None,
            'author_nickname': _get(author, 'nickname'), 'author_unique_id': _get(author, 'unique_id') or _get(author, 'short_id'),
            'author_sec_uid': _get(author, 'sec_uid'), 'create_time': obj.get('create_time'),
            'duration_seconds': duration, 'digg_count': _get(stats, 'digg_count'),
            'comment_count': _get(stats, 'comment_count'), 'collect_count': _get(stats, 'collect_count'),
            'share_count': _get(stats, 'share_count'), 'play_count': _get(stats, 'play_count'),
            'music_title': _get(music, 'title'), 'music_author': _get(music, 'author'),
            'video_width': _get(video, 'width'), 'video_height': _get(video, 'height')}


# 输入页面计数字符串；返回解析后的数值或空值。
def _parse_count(value):
    if not isinstance(value, str):
        return None
    s = value.strip().replace(',', '')
    m = re.search(r'(\d+(?:\.\d+)?)\s*([万亿]?)', s)
    if not m:
        return None
    return int(float(m.group(1)) * {'': 1, '万': 10000, '亿': 100000000}[m.group(2)])


# 从匹配目标 aweme_id 的作品 JSON 对象中提取专属高清视频流候选（杜绝背景推荐流或测试视频串台）。
def _extract_target_candidates(obj):
    from douyin_media_mcp.models import MediaCandidate
    video = obj.get('video') or {}
    if not isinstance(video, dict):
        return []
    out = []
    seen = set()
    sources = list(video.get('bit_rate') or [])
    if video.get('play_addr'):
        sources.append({'play_addr': video.get('play_addr'), 'bit_rate': video.get('bit_rate_val') or 0})
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        addr = entry.get('play_addr') or {}
        if not isinstance(addr, dict):
            continue
        w = addr.get('width') or video.get('width')
        h = addr.get('height') or video.get('height')
        size = addr.get('data_size')
        br = entry.get('bit_rate')
        for u in (addr.get('url_list') or []):
            if not isinstance(u, str) or not u.startswith('http') or u in seen:
                continue
            low = u.lower()
            if any(x in low for x in ('douyinstatic.com', 'uuu_265.mp4', 'byteimg.com')):
                continue
            seen.add(u)
            out.append(MediaCandidate(
                kind='video',
                url=u,
                mime_type='video/mp4',
                width=int(w) if isinstance(w, (int, float)) and w > 0 else None,
                height=int(h) if isinstance(h, (int, float)) and h > 0 else None,
                bitrate=int(br) if isinstance(br, (int, float)) and br > 0 else None,
                content_length=int(size) if isinstance(size, (int, float)) and size > 0 else None,
                request_headers={},
            ))
    return out


# 输入 BrowserResolver 类；只在当前进程替换方法并返回捕获状态，不修改下载器包文件。
# 由 %USERPROFILE%\douyin-browser-mcp\.venv 的 Python 运行；monkeypatch 仅存在于当前进程，不修改该包。
def install_capture(BrowserResolver):
    """Patch only this process; attach before navigation, filter static UI probe videos, and bind target aweme_id streams."""
    try:
        from douyin_media_mcp import candidates as cand_mod
        from douyin_media_mcp.resolvers import browser as browser_mod
        from douyin_media_mcp.models import MediaCandidate
        orig_cand_fn = cand_mod.candidate_from_response

        def filtered_candidate_from_response(url, **kwargs):
            low = str(url).lower()
            if any(x in low for x in ('douyinstatic.com', 'uuu_265.mp4', '/obj/douyin-pc-web/', 'login-guide', 'byteimg.com')):
                return None
            return orig_cand_fn(url, **kwargs)

        cand_mod.candidate_from_response = filtered_candidate_from_response
        browser_mod.candidate_from_response = filtered_candidate_from_response
    except Exception:
        pass
    capture = {'target': None, 'api': [], 'embedded': [], 'dom': {}, 'tasks': set(), 'target_candidates': [], 'ua': None}
    orig_resolve = BrowserResolver.resolve
    orig_open = BrowserResolver._open_context
    orig_metadata = BrowserResolver._metadata

    # 输入 resolver 实例、规范 URL 和作品 ID；记录目标后调用原方法并用目标专属流覆盖候选。
    async def resolve(self, canonical_url, aweme_id):
        capture['target'] = str(aweme_id)
        resolved = await orig_resolve(self, canonical_url, aweme_id)
        if capture['target_candidates']:
            headers_base = {'Referer': canonical_url}
            if capture.get('ua'):
                headers_base['User-Agent'] = capture['ua']
            elif resolved.candidates and resolved.candidates[0].request_headers:
                headers_base.update(resolved.candidates[0].request_headers)
            bound = [
                MediaCandidate(
                    kind=c.kind, url=c.url, mime_type=c.mime_type,
                    width=c.width, height=c.height, bitrate=c.bitrate,
                    content_length=c.content_length, request_headers=dict(headers_base)
                )
                for c in capture['target_candidates']
            ]
            resolved.candidates = bound + [c for c in resolved.candidates if c.kind != 'video']
        return resolved

    # 输入 resolver 实例与 Playwright；调用原方法、注册响应监听并返回上下文结果。
    async def open_context(self, playwright):
        result = await orig_open(self, playwright)
        context = result[1]
        # 输入浏览器响应；限制内容类型和体积后提取候选 API 元数据，无返回值。
        async def process(response):
            try:
                url = response.url.lower()
                if '/aweme/v1/web/aweme/detail' not in url and '/aweme/' not in url:
                    return
                headers = await response.all_headers()
                if 'json' not in headers.get('content-type', '').lower() and '/aweme/v1/web/aweme/detail' not in url:
                    return
                length = headers.get('content-length')
                if length and int(length) > MAX_BODY:
                    return
                body = await response.body()
                if len(body) > MAX_BODY:
                    return
                data = json.loads(body)
                for item in _aweme_objects(data, capture['target']):
                    capture['api'].append(_extract(item))
                    capture['target_candidates'].extend(_extract_target_candidates(item))
            except Exception:
                pass
        # 输入浏览器响应；调度异步解析任务并跟踪其生命周期，无返回值。
        def on_response(response):
            try:
                task = asyncio.create_task(process(response))
                capture['tasks'].add(task)
                task.add_done_callback(capture['tasks'].discard)
            except Exception:
                pass
        context.on('response', on_response)
        return result

    # 输入页面；等待目标 API 响应就绪，保留原元数据结果并解析嵌入数据和 DOM。
    async def metadata(page):
        for _ in range(24):
            if capture['api']:
                break
            if capture['tasks']:
                await asyncio.gather(*list(capture['tasks']), return_exceptions=True)
            if capture['api']:
                break
            await asyncio.sleep(0.25)
        base = await orig_metadata(page)
        if capture['tasks']:
            await asyncio.gather(*list(capture['tasks']), return_exceptions=True)
        try:
            capture['ua'] = await page.evaluate('navigator.userAgent')
            data = await page.evaluate('''() => {
              const selectors = ['detail-video-info','video-author-title','video-player-digg',
                'video-player-comment','video-player-collect','video-player-share'];
              const dom = {}; for (const s of selectors) dom[s] = document.querySelector(`[data-e2e="${s}"]`)?.innerText?.slice(0, 10000) || null;
              return {scripts: [...document.scripts].map(s => s.textContent || '').filter(s => s.length <= 2097152 && (s.includes('aweme_id') || s.includes('awemeId') || s.includes('%7B'))).slice(0, 30), dom,
                body: (document.body?.innerText || '').slice(0, 100000)};
            }''')
            for raw in data.get('scripts', []):
                for candidate in (raw, unquote(raw)):
                    with suppress(Exception):
                        parsed = json.loads(candidate)
                        for item in _aweme_objects(parsed, capture['target']):
                            capture['embedded'].append(_extract(item))
                            if not capture['target_candidates']:
                                capture['target_candidates'].extend(_extract_target_candidates(item))
                        break
            dom = data.get('dom', {})
            capture['dom'] = {'page_title': base.get('title'), 'duration_seconds': base.get('duration_seconds'),
                              'desc': dom.get('detail-video-info'), 'title': dom.get('detail-video-info'),
                              'author_nickname': dom.get('video-author-title'),
                              **{field: _parse_count(dom.get(selector)) for field, selector in (
                                  ('digg_count', 'video-player-digg'), ('comment_count', 'video-player-comment'),
                                  ('collect_count', 'video-player-collect'), ('share_count', 'video-player-share'))}}
            m = re.search(r'发布时间\s*[:：]?\s*(\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)', data.get('body', ''))
            if m:
                with suppress(ValueError, OverflowError):
                    date = re.sub(r'[年月/.]', '-', m.group(1)).replace('日', '')
                    dt = datetime.fromisoformat(date).replace(tzinfo=timezone(timedelta(hours=8)))
                    capture['dom']['create_time'] = int(dt.timestamp())
        except Exception:
            capture['dom'] = {'page_title': base.get('title'), 'duration_seconds': base.get('duration_seconds')}
        return base
    BrowserResolver.resolve = resolve
    BrowserResolver._open_context = open_context
    BrowserResolver._metadata = staticmethod(metadata)
    return capture


# 输入捕获数据和下载结果；返回合并元数据、字段来源、缺失原因三个字典。
# 字段按 api > embedded > dom 优先级填充，再使用下载结果补缺；白名单及 sanitizer 限制输出。
# 网页 API 返回的 play_count 为 0 按缺失处理，不当作真实播放量。
def build_metadata(capture, download):
    metadata = {field: None for field in FIELDS}
    sources = {field: None for field in FIELDS}
    for source in ('api', 'embedded', 'dom'):
        records = capture.get(source) or []
        if isinstance(records, dict):
            records = [records]
        for record in records:
            for field, value in sanitize_metadata(record).items():
                if metadata[field] is None and value is not None:
                    metadata[field], sources[field] = value, source
    aweme_id = str(download.get('aweme_id') or capture.get('target') or '')
    if aweme_id.isdigit():
        if metadata['aweme_id'] is None:
            metadata['aweme_id'], sources['aweme_id'] = aweme_id, 'download'
        metadata['canonical_url'], sources['canonical_url'] = f'https://www.douyin.com/video/{aweme_id}', 'download'
    for field, download_field in (('duration_seconds','duration_seconds'), ('video_width','width'), ('video_height','height')):
        if metadata[field] is None:
            value = sanitize_metadata({field: download.get(download_field)}).get(field)
            if value is not None:
                metadata[field], sources[field] = value, 'download'
    if metadata['create_time'] is not None:
        try:
            metadata['create_time'] = int(metadata['create_time'])
        except (ValueError, TypeError, OverflowError):
            metadata['create_time'] = None
            sources['create_time'] = None
        metadata['create_time_iso'] = create_time_iso(metadata['create_time'])
        if metadata['create_time_iso']:
            sources['create_time_iso'] = sources['create_time']
    play_zero = metadata['play_count'] == 0 and sources['play_count'] == 'api'
    if play_zero:
        metadata['play_count'], sources['play_count'] = None, None
    missing = {field: '公开来源未提供或未能安全提取' for field in FIELDS if metadata[field] is None}
    if play_zero:
        missing['play_count'] = '抖音网页接口不公开播放量（statistics.play_count 恒为 0），按缺失处理'
    return metadata, sources, missing


# 输入命令行参数；异步下载并返回安全整理的下载结果及元数据。
async def run(args):
    from douyin_media_mcp.service import DouyinMediaService
    from douyin_media_mcp.config import Settings
    from douyin_media_mcp.resolvers.browser import BrowserResolver
    capture = install_capture(BrowserResolver)
    start = time.perf_counter()
    download = await DouyinMediaService(Settings.from_env()).download(
        url=args.url, output_dir=args.out_dir, quality='best', overwrite=True)
    if capture.get('target') and not capture['api'] and not capture['embedded'] and not capture['target_candidates']:
        raise RuntimeError(f"Target aweme_id {capture['target']} did not return API, embedded metadata, or target stream candidates (possible page block); refusing unbound stream")
    metadata, sources, missing = build_metadata(capture, download)
    subset = {k: download.get(k) for k in ('output_path', 'duration_seconds', 'width', 'height', 'fps', 'audio_codec', 'media_mode')}
    subset['warnings'] = [s for w in download.get('warnings', []) if (s := safe_text(w))]
    return {'status': 'completed', 'video_path': download['output_path'], 'download': subset,
            'metadata': metadata, 'metadata_sources': sources, 'missing': missing,
            'timings': {'download': round(time.perf_counter()-start, 3)}}


# 读取 URL 和输出路径；写成功或脱敏错误 JSON，返回退出码。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--result-json', required=True)
    args = parser.parse_args()
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(run(args))
        code = 0
    except Exception as exc:
        # Never persist arbitrary exception strings: dependencies can include signed URLs or secrets.
        try:
            from douyin_media_mcp.errors import MediaMCPError
        except ImportError:
            MediaMCPError = ()
        if isinstance(exc, MediaMCPError):
            error_code = safe_text(exc.code) or 'DOWNLOAD_FAILED'
            message = safe_text(exc.message) or '下载失败；详情已隐藏。'
        elif 'refusing unbound stream' in str(exc):
            error_code, message = 'TARGET_STREAM_UNBOUND', '未捕获到目标作品专属 API 或视频流，已拦截未绑定流防止串台。'
        else:
            error_code, message = 'FETCH_FAILED', '下载失败；详情已隐藏。'
        result = {'status': 'error', 'error': {'code': error_code, 'message': message}}
        code = 1
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
