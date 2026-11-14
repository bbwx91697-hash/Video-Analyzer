# 媒体路由与外站适配：严格按解析主机名分发抖音与支持外站，屏蔽下载器差异并统一安全元数据输出。
"""Strict host-based URL routing, optional yt-dlp execution, and unified sanitized media schema."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SAFE_REPLAY_QUERY_KEYS = {'v', 'p'}

URL_RE = re.compile(r'https?://[^\s"\'<>]+', re.I)
SECRET_RE = re.compile(r'(?i)(cookie|token|authorization|bearer|signature|x-bogus|msToken)\s*[:=]')

METADATA_FIELDS = (
    'platform',
    'platform_item_id',
    'aweme_id',
    'canonical_url',
    'title',
    'desc',
    'hashtags',
    'author_nickname',
    'author_unique_id',
    'author_sec_uid',
    'create_time',
    'create_time_iso',
    'duration_seconds',
    'digg_count',
    'comment_count',
    'collect_count',
    'share_count',
    'play_count',
    'music_title',
    'music_author',
    'video_width',
    'video_height',
    'page_title',
)

# 平台路由表：所有判定只基于 urlsplit().hostname 的精确匹配或父域后缀，禁止子串包含匹配。
ROUTE_TABLE = (
    {
        'platform': 'douyin',
        'downloader': 'douyin_browser_mcp',
        'hosts': ('douyin.com', 'iesdouyin.com', 'amemv.com'),
        'short_hosts': ('v.douyin.com',),
    },
    {
        'platform': 'bilibili',
        'downloader': 'yt_dlp',
        'hosts': ('bilibili.com', 'b23.tv', 'bili2233.cn'),
        'short_hosts': ('b23.tv', 'bili2233.cn'),
    },
    {
        'platform': 'youtube',
        'downloader': 'yt_dlp',
        'hosts': ('youtube.com', 'youtu.be', 'youtube-nocookie.com'),
        'short_hosts': ('youtu.be',),
    },
)

BLOCKED_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0', '::1'}


class UnsupportedRouteError(ValueError):
    """Raised when a URL uses an invalid scheme, missing host, or unsupported domain."""


# 输入任意字符串；剥离 URL 查询参数与常见凭据字段，避免签名链接或密钥落盘。
def safe_text(value):
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or SECRET_RE.search(value):
        return None

    def _clean(match):
        try:
            p = urlsplit(match.group())
            return urlunsplit((p.scheme, p.netloc, p.path, '', ''))
        except ValueError:
            return '[URL]'

    cleaned = URL_RE.sub(_clean, value)
    cleaned = re.sub(r'(?i)\b(cookie|token|authorization|bearer|signature|x-bogus|msToken)\s*[:=]\s*[^\s,;]+', r'\1=[redacted]', cleaned)
    return cleaned[:10000] if cleaned else None


# 输入待持久化供断点续跑的 URL；只保留白名单视频标识参数 (v/p)，彻底剥离一切凭据或追踪查询参数。
def sanitize_replay_url(url):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        p = urlsplit(url.strip())
    except ValueError:
        return None
    safe_qs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k.lower() in SAFE_REPLAY_QUERY_KEYS]
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(safe_qs), ''))


# 输入待 export 的文本字段；拒绝携带任意外链，防止直链或签名路径泄露。
def safe_public_text(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or SECRET_RE.search(value) or URL_RE.search(value):
        return None
    return value[:10000]


def _host_matches(host, domain):
    return host == domain or host.endswith('.' + domain)


# 输入链接字符串；严格基于 urlsplit().hostname 返回路由信息，未知或非法链接直接抛出 UnsupportedRouteError。
def classify_url_route(url):
    if not isinstance(url, str) or not url.strip():
        raise UnsupportedRouteError('[route] Empty URL')
    raw = url.strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        raise UnsupportedRouteError('[route] Malformed URL') from None
    scheme = (parsed.scheme or '').lower()
    if scheme not in ('http', 'https'):
        raise UnsupportedRouteError(f"[route] Unsupported URL scheme '{scheme or 'empty'}': only http/https are allowed")
    host = (parsed.hostname or '').lower().rstrip('.')
    if not host:
        raise UnsupportedRouteError('[route] URL missing valid hostname')
    if host in BLOCKED_HOSTS:
        raise UnsupportedRouteError(f"[route] Localhost/loopback host '{host}' is blocked")
    path_lower = (parsed.path or '').lower()
    for spec in ROUTE_TABLE:
        if any(_host_matches(host, domain) for domain in spec['hosts']):
            is_short = any(_host_matches(host, sh) for sh in spec['short_hosts']) or ('/share/' in path_lower)
            route_kind = f"{spec['platform']}_{'short' if is_short else 'standard'}"
            return {
                'platform': spec['platform'],
                'downloader': spec['downloader'],
                'host': host,
                'route_kind': route_kind,
                'is_short_link': bool(is_short),
            }
    raise UnsupportedRouteError(
        f"[route] Unsupported URL host '{host}': only Douyin (douyin.com/v.douyin.com/iesdouyin.com) "
        "and explicit external platforms (bilibili.com/b23.tv, youtube.com/youtu.be) are supported"
    )


# 输入可选 CLI 覆盖路径；探测 yt-dlp 可用启动命令，未安装或显式禁用时返回 (None, 原因)。
def resolve_ytdlp_runner(ytdlp_bin=None):
    if os.environ.get('DOUYIN_ANALYZER_DISABLE_YTDLP') == '1' and not ytdlp_bin:
        return None, 'disabled by DOUYIN_ANALYZER_DISABLE_YTDLP'
    if ytdlp_bin is not None:
        raw = str(ytdlp_bin).strip()
        if not raw or raw.lower() in ('none', 'disabled', '__missing__'):
            return None, f'yt-dlp unavailable (override={raw or "empty"})'
        p = Path(raw)
        if p.is_file():
            return [str(p.resolve())], None
        found = shutil.which(raw)
        if found:
            return [found], None
        return None, f'yt-dlp executable not found: {raw}'
    if importlib.util.find_spec('yt_dlp') is not None:
        return [sys.executable, '-m', 'yt_dlp'], None
    found = shutil.which('yt-dlp')
    if found:
        return [found], None
    return None, 'yt-dlp is not installed in Python venv or PATH'


def _unix_to_iso(value):
    try:
        return datetime.fromtimestamp(int(value), timezone(timedelta(hours=8))).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _parse_upload_date_to_ts(upload_date):
    if not isinstance(upload_date, str) or not re.fullmatch(r'\d{8}', upload_date.strip()):
        return None
    try:
        dt = datetime.strptime(upload_date.strip(), '%Y%m%d').replace(tzinfo=timezone(timedelta(hours=8)))
        return int(dt.timestamp())
    except ValueError:
        return None


def _safe_int_or_float(val):
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        return None
    return val


def _canonical_external_url(platform, item_id, info):
    if platform == 'bilibili' and item_id:
        if re.fullmatch(r'BV[0-9A-Za-z]+', str(item_id)):
            return f'https://www.bilibili.com/video/{item_id}'
        if str(item_id).isdigit():
            return f'https://www.bilibili.com/video/av{item_id}'
    if platform == 'youtube' and item_id:
        return f'https://www.youtube.com/watch?v={item_id}'
    webpage = info.get('webpage_url')
    if isinstance(webpage, str) and webpage.startswith(('http://', 'https://')):
        try:
            p = urlsplit(webpage)
            return urlunsplit((p.scheme, p.netloc, p.path, '', ''))
        except ValueError:
            return None
    return None


def _locate_downloaded_video(out_dir, expected_id=None):
    media_exts = {'.mp4', '.mkv', '.webm', '.mov', '.flv', '.m4v'}
    candidates = []
    for p in sorted(Path(out_dir).iterdir()):
        if not p.is_file():
            continue
        if p.name.endswith('.info.json') or p.suffix.lower() in ('.json', '.part', '.ytdl', '.temp'):
            continue
        if p.suffix.lower() in media_exts:
            candidates.append(p)
    if expected_id:
        for p in candidates:
            if expected_id in p.stem:
                return p
    return candidates[0] if candidates else None


# 输入平台、yt-dlp 信息字典与输出文件；提取白名单公开字段并追踪字段来源及缺失原因。
def build_external_metadata(platform, info, media_file=None):
    info = info if isinstance(info, dict) else {}
    item_id = safe_public_text(str(info.get('id') or info.get('display_id') or '')) or None
    if not item_id and media_file is not None:
        stem = Path(media_file).stem
        if stem.startswith('ytdlp_') and len(stem) > 6:
            item_id = safe_public_text(stem[6:])

    metadata = {f: None for f in METADATA_FIELDS}
    sources = {f: None for f in METADATA_FIELDS}

    metadata['platform'] = platform
    sources['platform'] = 'route'
    if item_id:
        metadata['platform_item_id'] = item_id
        sources['platform_item_id'] = 'yt_dlp'
    canonical = _canonical_external_url(platform, item_id, info)
    if canonical:
        metadata['canonical_url'] = canonical
        sources['canonical_url'] = 'yt_dlp'

    text_map = (
        ('title', info.get('title') or info.get('fulltitle')),
        ('desc', info.get('description')),
        ('author_nickname', info.get('uploader') or info.get('channel') or info.get('creator')),
        ('author_unique_id', info.get('uploader_id') or info.get('channel_id')),
        ('page_title', info.get('fulltitle') or info.get('title')),
    )
    for field, raw_val in text_map:
        cleaned = safe_public_text(raw_val)
        if cleaned:
            metadata[field] = cleaned
            sources[field] = 'yt_dlp'

    raw_tags = info.get('tags') or info.get('categories')
    if isinstance(raw_tags, list):
        clean_tags = [t for item in raw_tags if (t := safe_public_text(str(item)))]
        if clean_tags:
            metadata['hashtags'] = clean_tags[:50]
            sources['hashtags'] = 'yt_dlp'

    ts = _safe_int_or_float(info.get('timestamp') or info.get('release_timestamp'))
    if ts is not None:
        metadata['create_time'] = int(ts)
        sources['create_time'] = 'yt_dlp'
    else:
        parsed_ts = _parse_upload_date_to_ts(info.get('upload_date'))
        if parsed_ts is not None:
            metadata['create_time'] = parsed_ts
            sources['create_time'] = 'yt_dlp_upload_date'
    if metadata['create_time'] is not None:
        iso = _unix_to_iso(metadata['create_time'])
        if iso:
            metadata['create_time_iso'] = iso
            sources['create_time_iso'] = sources['create_time']

    num_map = (
        ('duration_seconds', info.get('duration')),
        ('digg_count', info.get('like_count')),
        ('comment_count', info.get('comment_count')),
        ('share_count', info.get('repost_count')),
        ('play_count', info.get('view_count')),
        ('video_width', info.get('width')),
        ('video_height', info.get('height')),
    )
    for field, raw_val in num_map:
        num = _safe_int_or_float(raw_val)
        if num is not None:
            metadata[field] = num
            sources[field] = 'yt_dlp'

    missing = {}
    for f in METADATA_FIELDS:
        if metadata[f] is None:
            if f == 'aweme_id':
                missing[f] = f'非抖音平台 ({platform}) 无 aweme_id'
            elif f in ('author_sec_uid', 'collect_count', 'music_title', 'music_author'):
                missing[f] = f'外站公开接口 ({platform}) 未提供该平台专有字段'
            else:
                missing[f] = f'外站来源 ({platform}) 未返回或未公开该字段'
    return metadata, sources, missing


# 将任意平台 fetch 输出归一化为统一媒体结果接口，保留平台、作品 ID、字段来源与缺失原因。
def normalize_media_result(raw_fetch, route, video_path=None):
    raw_fetch = raw_fetch if isinstance(raw_fetch, dict) else {}
    platform = route['platform']
    downloader = route['downloader']
    route_kind = route['route_kind']

    meta = dict(raw_fetch.get('metadata') or {})
    sources = dict(raw_fetch.get('metadata_sources') or {})
    missing = dict(raw_fetch.get('missing') or {})

    item_id = meta.get('platform_item_id') or meta.get('aweme_id')
    if item_id is not None:
        item_id = safe_public_text(str(item_id)) or str(item_id)
    meta['platform'] = platform
    sources.setdefault('platform', 'route')
    if item_id:
        meta['platform_item_id'] = item_id
        sources.setdefault('platform_item_id', sources.get('aweme_id') or ('yt_dlp' if downloader == 'yt_dlp' else 'download'))
    else:
        missing.setdefault('platform_item_id', f'来源 ({platform}) 未提取到作品 ID')

    if platform == 'douyin':
        aweme_id = meta.get('aweme_id')
        if aweme_id:
            sources.setdefault('aweme_id', 'download')
        else:
            missing.setdefault('aweme_id', '公开来源未提供或未能提取 aweme_id')

    resolved_video = str(video_path or raw_fetch.get('video_path') or '')
    task_unique_key = f'{platform}:{item_id}' if item_id else f'{platform}:unresolved'
    return {
        'status': raw_fetch.get('status') or 'completed',
        'platform': platform,
        'platform_item_id': item_id,
        'task_unique_key': task_unique_key,
        'downloader': downloader,
        'route_kind': route_kind,
        'video_path': resolved_video,
        'download': raw_fetch.get('download') or {},
        'metadata': meta,
        'metadata_sources': sources,
        'missing': missing,
        'timings': raw_fetch.get('timings') or {},
    }


# 使用 yt-dlp 执行外站公开视频下载与元数据提取，不读取任何浏览器 cookies，并将故障输出脱敏。
def fetch_external_with_ytdlp(
    url,
    out_dir,
    route,
    ytdlp_cmd,
    ffmpeg_bin,
    command_runner,
    timeout=300,
    socket_timeout=30,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / 'ytdlp_%(id)s.%(ext)s')
    ffmpeg_dir = str(Path(ffmpeg_bin).resolve().parent) if ffmpeg_bin else None
    cmd = [
        *ytdlp_cmd,
        '--no-playlist',
        '--no-cookies',
        '--no-warnings',
        '--write-info-json',
        '--socket-timeout',
        str(int(socket_timeout)),
        '-f',
        'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b',
        '--merge-output-format',
        'mp4',
        '-o',
        template,
    ]
    if ffmpeg_dir:
        cmd += ['--ffmpeg-location', ffmpeg_dir]
    cmd += ['--', url]

    command_runner(cmd, timeout=timeout)

    info_files = sorted(out_dir.glob('*.info.json'))
    info = {}
    if info_files:
        try:
            info = json.loads(info_files[0].read_text(encoding='utf-8'))
        except Exception:
            info = {}

    video_file = _locate_downloaded_video(out_dir, expected_id=str(info.get('id') or '') or None)
    if not video_file or not video_file.is_file():
        raise RuntimeError(f"[fetch] External downloader ({route['platform']}) finished without producing a video file")

    metadata, sources, missing = build_external_metadata(route['platform'], info, media_file=video_file)
    warnings = []
    if not info_files:
        warnings.append(f"外站 ({route['platform']}) 未产出 .info.json，部分标题/作者元数据缺失。")

    raw_fetch = {
        'status': 'completed',
        'video_path': str(video_file),
        'download': {
            'output_path': str(video_file),
            'downloader': 'yt_dlp',
            'platform': route['platform'],
            'warnings': warnings,
        },
        'metadata': metadata,
        'metadata_sources': sources,
        'missing': missing,
        'timings': {},
    }
    return normalize_media_result(raw_fetch, route, video_path=video_file)
