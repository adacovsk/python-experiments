#!/usr/bin/env python3
"""Download YouTube videos/streams as MP3 files with trailing silence removal."""

import re
import subprocess
import sys
from pathlib import Path

import yt_dlp


def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=10,
    )
    return float(r.stdout.strip())


def trim_trailing_silence(path: str, threshold_db: int = -50, min_silence: int = 2) -> None:
    """Trim trailing silence from an MP3 file by detecting silence then re-encoding."""
    duration = get_duration(path)
    seek_to = max(0, duration - 300)

    r = subprocess.run(
        ["ffmpeg", "-ss", str(seek_to), "-i", path,
         "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )

    starts = [float(m) + seek_to for m in re.findall(r"silence_start: ([0-9.]+)", r.stderr)]
    ends = [float(m) + seek_to for m in re.findall(r"silence_end: ([0-9.]+)", r.stderr)]

    if not starts:
        return

    last_start = starts[-1]
    trim_to = None

    if len(starts) > len(ends):
        # Silence extends to EOF
        if duration - last_start > min_silence:
            trim_to = last_start + 0.5
    elif ends and abs(ends[-1] - duration) < 2:
        # Last silence ends at/near EOF
        if duration - last_start > min_silence:
            trim_to = last_start + 0.5

    if trim_to is None:
        return

    removed = duration - trim_to
    print(f"Trimming {removed:.1f}s of trailing silence from {Path(path).name}")

    tmp = path + ".trimmed.mp3"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-t", str(trim_to), "-c:a", "libmp3lame", "-q:a", "0", tmp],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode == 0:
        Path(tmp).replace(path)
    else:
        Path(tmp).unlink(missing_ok=True)


def download_as_mp3(url: str, output_dir: str = ".") -> None:
    downloaded: list[str] = []

    def track_downloads(d: dict) -> None:
        if d["status"] == "finished":
            downloaded.append(d["filename"])

    opts = {
        "ignoreerrors": True,
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "0",
            }
        ],
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "extractor_args": {"youtube": {"js_runtimes": ["deno"]}},
        "remote_components": {"ejs": "github"},
        "progress_hooks": [track_downloads],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # Trim trailing silence from all downloaded files
    for path in downloaded:
        mp3_path = str(Path(path).with_suffix(".mp3"))
        if Path(mp3_path).exists():
            trim_trailing_silence(mp3_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <youtube_url> [output_dir]")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    download_as_mp3(url, output_dir)