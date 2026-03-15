#!/usr/bin/env python3
"""Upload translated video to Bilibili with dual parts (P1: dubbed, P2: original+subs).
Runs locally on Mac. Reads cookies from Google Drive.

Usage: python3 upload_bilibili.py <folder_path> [--title "标题"] [--tags "tag1,tag2"]
"""
import os, sys, json, re, time, subprocess

# Google Drive local path on macOS — adjust email to match your account
# macOS 中文版: ~/Library/CloudStorage/GoogleDrive-<email>/我的云端硬盘/
# macOS English: ~/Library/CloudStorage/GoogleDrive-<email>/My Drive/
_GDRIVE_EMAIL = os.environ.get('GDRIVE_EMAIL', '')
if _GDRIVE_EMAIL:
    DRIVE_ROOT = os.path.expanduser(
        f"~/Library/CloudStorage/GoogleDrive-{_GDRIVE_EMAIL}/我的云端硬盘/video-translate"
    )
else:
    # Auto-detect Google Drive path
    _cloud = os.path.expanduser("~/Library/CloudStorage")
    _candidates = [d for d in os.listdir(_cloud) if d.startswith("GoogleDrive-")] if os.path.exists(_cloud) else []
    if _candidates:
        # Try Chinese path first, then English
        _base = os.path.join(_cloud, _candidates[0])
        for _subdir in ["我的云端硬盘", "My Drive"]:
            _try = os.path.join(_base, _subdir, "video-translate")
            if os.path.exists(_try):
                DRIVE_ROOT = _try
                break
        else:
            DRIVE_ROOT = os.path.join(_base, "我的云端硬盘", "video-translate")
    else:
        DRIVE_ROOT = os.path.expanduser("~/video-translate")
COOKIES_PATH = os.path.join(DRIVE_ROOT, "bilibili_cookies.json")


def load_cookies():
    """Load cookies from Drive"""
    with open(COOKIES_PATH, 'r') as f:
        creds = json.load(f)
    return {
        "cookie_info": {
            "cookies": [
                {"name": "SESSDATA", "value": creds["SESSDATA"]},
                {"name": "bili_jct", "value": creds["bili_jct"]},
            ]
        }
    }


def load_metadata(folder):
    """Load title/desc/tags/source from available metadata files.

    Priority:
    1. download.info.json (yt-dlp --write-info-json) — most reliable
    2. translation.json — fallback: summarize first ~500 chars
    3. folder name — last resort
    """
    info = {}

    # Try download.info.json first (from yt-dlp --write-info-json)
    info_path = os.path.join(folder, 'download.info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            yt_info = json.load(f)
        info['yt_title'] = yt_info.get('title', '')
        info['uploader'] = yt_info.get('uploader', yt_info.get('channel', ''))
        info['description'] = yt_info.get('description', '')[:500]
        info['categories'] = yt_info.get('categories', [])
        info['yt_tags'] = yt_info.get('tags', [])[:5]

    # Try translation.json for content summary
    translation_path = os.path.join(folder, 'translation.json')
    if os.path.exists(translation_path):
        with open(translation_path, 'r', encoding='utf-8') as f:
            segs = json.load(f)
        # First 500 chars of translation as description
        info['translated_text'] = ''.join(
            seg.get('translation', '') for seg in segs[:20]
        )[:500]

    return info


def extract_cover(folder):
    """Extract a cover frame from video if no cover image exists.
    Returns cover image path or None.
    """
    # Check existing cover images
    for name in ['cover.jpg', 'cover.png', 'video.png', 'video.jpg']:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p

    # Check yt-dlp thumbnail
    for ext in ['.jpg', '.png', '.webp']:
        p = os.path.join(folder, f'download{ext}')
        if os.path.exists(p):
            return p

    # Extract from video at 10% position
    video = os.path.join(folder, 'video_chinese.mp4')
    if not os.path.exists(video):
        video = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video):
        return None

    cover = os.path.join(folder, 'cover.jpg')
    try:
        # Get duration
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', video],
            capture_output=True, text=True
        )
        duration = float(json.loads(r.stdout).get('format', {}).get('duration', 60))
        # Extract frame at 10% of video
        timestamp = min(duration * 0.1, 30)
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(timestamp), '-i', video,
             '-vframes', '1', '-q:v', '2', cover],
            capture_output=True, timeout=30
        )
        if os.path.exists(cover) and os.path.getsize(cover) > 1000:
            return cover
    except Exception:
        pass
    return None


def generate_p2(folder):
    """Generate P2 video (original + Chinese subtitles burned in).
    Uses original timestamps from translation.json.
    Returns P2 video path or None.
    """
    original = os.path.join(folder, 'download.mp4')
    p2_video = os.path.join(folder, 'original_with_subs.mp4')
    translation_path = os.path.join(folder, 'translation.json')

    if os.path.exists(p2_video) and os.path.getsize(p2_video) > 1000:
        print(f'  P2 already exists: {os.path.getsize(p2_video)/1e6:.1f}MB')
        return p2_video

    if not os.path.exists(original) or not os.path.exists(translation_path):
        return None

    with open(translation_path, 'r', encoding='utf-8') as f:
        segs = json.load(f)

    # Generate SRT with original timestamps
    srt_path = os.path.join(folder, 'subtitles_original.srt')

    def fmt_ts(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        h, seconds = divmod(int(seconds), 3600)
        m, s = divmod(seconds, 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(srt_path, 'w', encoding='utf-8') as f:
        idx = 1
        for seg in segs:
            t = seg.get('translation', '').strip()
            if not t or t.startswith('['):
                continue
            # Use original timestamps (not TTS-adjusted)
            s = seg.get('original_start', seg.get('start', 0))
            e = seg.get('original_end', seg.get('end', 0))
            # Split long lines (max 25 chars per line, max 3 lines)
            lines = [t] if len(t) <= 25 else [t[i:i+25] for i in range(0, len(t), 25)][:3]
            f.write(f"{idx}\n{fmt_ts(s)} --> {fmt_ts(e)}\n")
            f.write('\n'.join(lines) + '\n\n')
            idx += 1

    # Burn subtitles into original video
    print('  Burning subtitles to P2...')
    srt_esc = srt_path.replace(':', '\\:').replace("'", "\\'")
    vf = f"subtitles='{srt_esc}':force_style='FontName=Noto Sans CJK SC,FontSize=22'"
    r = subprocess.run(
        ['ffmpeg', '-y', '-i', original, '-vf', vf,
         '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
         p2_video],
        capture_output=True, text=True
    )
    if os.path.exists(p2_video) and os.path.getsize(p2_video) > 1000:
        print(f'  P2 generated: {os.path.getsize(p2_video)/1e6:.1f}MB')
        return p2_video
    else:
        print(f'  P2 generation failed: {r.stderr[:200] if r.stderr else "unknown"}')
        return None


def upload(folder, title=None, desc=None, tags=None):
    """Upload video with P1 (dubbed) + P2 (original+subs) to Bilibili"""
    from biliup.plugins.bili_webup import BiliBili, Data

    # Check if already uploaded
    result_path = os.path.join(folder, 'bilibili.json')
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            r = json.load(f)
        if r.get('code') == 0:
            bvid = r.get('data', {}).get('bvid', '')
            print(f'Already uploaded: https://www.bilibili.com/video/{bvid}')
            return True

    # P1 video
    p1 = os.path.join(folder, 'video_chinese.mp4')
    if not os.path.exists(p1):
        print(f'Error: P1 not found: {p1}')
        return False

    # Load metadata
    meta = load_metadata(folder)

    # Build title
    if not title:
        folder_name = os.path.basename(folder)
        # Remove date prefix (e.g., "2026-03-15 Title Here")
        parts = folder_name.split(' ', 1)
        if len(parts) > 1 and len(parts[0]) == 10 and parts[0][4] == '-':
            raw_title = parts[1]
        else:
            raw_title = folder_name
        title = f'【中配】{raw_title}'
        # Append uploader if space allows
        uploader = meta.get('uploader', '')
        if uploader and len(title) + len(uploader) + 3 <= 80:
            title = f'{title} - {uploader}'
    if len(title) > 80:
        title = title[:77] + '...'

    # Build description
    if not desc:
        # Use translated content summary
        translated = meta.get('translated_text', '')
        if translated:
            # Take first 2-3 sentences
            sentences = re.split(r'[。！？]', translated)
            desc = '。'.join(s for s in sentences[:3] if s.strip())
            if desc and not desc.endswith('。'):
                desc += '。'
        else:
            desc = title.replace('【中配】', '')
    desc += '\n\nP1 中配 | P2 字幕'
    if len(desc) > 2000:
        desc = desc[:2000]

    # Build tags
    if not tags:
        tags = ['翻译', '科技', 'AI', '人工智能']
        # Add yt-dlp tags if available
        yt_tags = meta.get('yt_tags', [])
        for t in yt_tags:
            if len(t) <= 20 and t not in tags:
                tags.append(t)
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(',')]
    tags = [t[:20] for t in tags[:12]]

    # Source (for copyright=2 repost)
    source = meta.get('uploader', 'YouTube')

    # Generate P2 if needed
    p2 = os.path.join(folder, 'original_with_subs.mp4')
    if not os.path.exists(p2):
        p2 = generate_p2(folder)

    # Extract cover
    cover = extract_cover(folder)

    print(f'Title: {title}')
    print(f'Source: {source}')
    print(f'Tags: {", ".join(tags)}')
    print(f'P1: {os.path.basename(p1)} ({os.path.getsize(p1)/1e6:.1f}MB)')
    if p2 and os.path.exists(p2):
        print(f'P2: {os.path.basename(p2)} ({os.path.getsize(p2)/1e6:.1f}MB)')
    else:
        print('P2: not available (uploading P1 only)')
    if cover:
        print(f'Cover: {os.path.basename(cover)}')

    cookies = load_cookies()

    video = Data()
    video.title = title
    video.desc = desc
    video.copyright = 2  # Repost
    video.source = source
    video.tid = 201  # Science & Tech
    video.set_tag(tags)

    for attempt in range(3):
        try:
            with BiliBili(video) as bili:
                bili.login_by_cookies(cookies)

                # Upload cover
                if cover:
                    try:
                        cover_url = bili.cover_up(cover)
                        video.cover = cover_url.replace('http:', '')
                        print(f'  Cover uploaded')
                    except Exception as e:
                        print(f'  Cover upload failed: {e}')

                # Upload P1
                print('Uploading P1 (中配)...')
                part1 = bili.upload_file(p1)
                part1['title'] = '中配'
                video.append(part1)
                print('  P1 uploaded')

                # Upload P2 if available
                if p2 and os.path.exists(p2):
                    print('Uploading P2 (字幕)...')
                    part2 = bili.upload_file(p2)
                    part2['title'] = '字幕'
                    video.append(part2)
                    print('  P2 uploaded')

                # Submit
                ret = bili.submit()
                print(f'Result: {ret}')

                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(ret, f, ensure_ascii=False, indent=2)

                if ret.get('code') == 0:
                    bvid = ret.get('data', {}).get('bvid', '')
                    print(f'Success! https://www.bilibili.com/video/{bvid}')
                    return True
                else:
                    print(f'Submit failed: {ret}')

        except Exception as e:
            print(f'Error (attempt {attempt+1}): {e}')
            time.sleep(10)

    return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Upload to Bilibili')
    parser.add_argument('folder', help='Video folder path')
    parser.add_argument('--title', help='Custom title')
    parser.add_argument('--desc', help='Custom description')
    parser.add_argument('--tags', help='Comma-separated tags')
    args = parser.parse_args()

    # Expand ~ and resolve path
    folder = os.path.expanduser(args.folder)
    if not os.path.isabs(folder):
        folder = os.path.join(DRIVE_ROOT, 'processing', folder)

    upload(folder, title=args.title, desc=args.desc, tags=args.tags)
