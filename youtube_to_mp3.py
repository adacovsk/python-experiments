#!/usr/bin/env python3
"""Download YouTube videos/streams as MP3 files."""

import sys
import yt_dlp


def download_as_mp3(url: str, output_dir: str = ".") -> None:
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
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <youtube_url> [output_dir]")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    download_as_mp3(url, output_dir)
