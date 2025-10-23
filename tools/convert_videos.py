#!/usr/bin/env python3
"""
Batch-convert videos under a root folder to web-friendly formats using ffmpeg.

Default: convert to H.264 MP4 for wide browser support.
Optional: also produce VP9 WebM.

Requirements:
  - ffmpeg installed and on PATH (https://ffmpeg.org/)

Examples (from repo root):
  python tools/convert_videos.py --root docs/videos
  python tools/convert_videos.py --root docs/videos --make-webm
  python tools/convert_videos.py --root docs/videos --overwrite --preset slow --crf 22
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

VIDEO_EXT_IN = {'.avi', '.mov', '.mkv', '.wmv', '.mpg', '.mpeg', '.m4v', '.flv'}
MP4_EXT = '.mp4'
WEBM_EXT = '.webm'


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
    return proc.wait()


def build_ffmpeg_cmd(src: Path, dst: Path, v_codec: str, a_codec: str, crf: int, preset: str,
                     audio_bitrate: str, extra: Optional[List[str]] = None) -> List[str]:
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(src),
        '-c:v', v_codec,
        '-preset', preset,
        '-crf', str(crf),
    ]
    if v_codec == 'libx264':
        cmd += ['-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    if v_codec == 'libvpx-vp9':
        # Constant quality mode; row-mt speeds up on multi-core
        cmd += ['-b:v', '0', '-row-mt', '1']

    cmd += [
        '-c:a', a_codec,
        '-b:a', audio_bitrate,
        '-ar', '48000',
        str(dst),
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def convert_file(src: Path, make_mp4: bool, make_webm: bool, overwrite: bool, crf: int, preset: str) -> List[Path]:
    out_paths: List[Path] = []
    base = src.with_suffix('')
    if make_mp4:
        dst_mp4 = base.with_suffix(MP4_EXT)
        if overwrite or not dst_mp4.exists():
            print(f"[MP4] {src} -> {dst_mp4}")
            cmd = build_ffmpeg_cmd(
                src, dst_mp4, v_codec='libx264', a_codec='aac', crf=crf, preset=preset, audio_bitrate='160k'
            )
            code = run(cmd)
            if code != 0:
                print(f"ffmpeg failed converting to MP4: {src}", file=sys.stderr)
            else:
                out_paths.append(dst_mp4)
        else:
            print(f"[SKIP] MP4 exists: {dst_mp4}")
            out_paths.append(dst_mp4)

    if make_webm:
        dst_webm = base.with_suffix(WEBM_EXT)
        if overwrite or not dst_webm.exists():
            print(f"[WEBM] {src} -> {dst_webm}")
            cmd = build_ffmpeg_cmd(
                src, dst_webm, v_codec='libvpx-vp9', a_codec='libopus', crf=max(0, min(crf + 8, 45)),
                preset=preset, audio_bitrate='112k'
            )
            code = run(cmd)
            if code != 0:
                print(f"ffmpeg failed converting to WebM: {src}", file=sys.stderr)
            else:
                out_paths.append(dst_webm)
        else:
            print(f"[SKIP] WEBM exists: {dst_webm}")
            out_paths.append(dst_webm)

    return out_paths


def main():
    ap = argparse.ArgumentParser(description='Convert videos to web-playable formats using ffmpeg.')
    ap.add_argument('--root', default='docs/videos', help='Root folder to scan (default: docs/videos)')
    ap.add_argument('--make-mp4', action='store_true', help='Produce H.264 MP4 (yuv420p)')
    ap.add_argument('--make-webm', action='store_true', help='Also produce VP9 WebM')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')
    ap.add_argument('--crf', type=int, default=23, help='Quality factor (lower is better; default: 23 for x264)')
    ap.add_argument('--preset', default='slow', help='Encoder preset (x264/vp9): ultrafast..placebo (default: slow)')
    args = ap.parse_args()

    if not which('ffmpeg'):
        print('Error: ffmpeg not found on PATH. Install from https://ffmpeg.org/', file=sys.stderr)
        return 2

    make_mp4 = args.make_mp4 or (not args.make_webm)
    make_webm = args.make_webm

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2

    converted = 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext in VIDEO_EXT_IN:
                outs = convert_file(p, make_mp4, make_webm, args.overwrite, args.crf, args.preset)
                converted += len([o for o in outs if o.exists()])
            else:
                skipped += 1

    print(f"Done. Converted outputs: {converted}; skipped (non-target files): {skipped}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
