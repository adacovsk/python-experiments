"""Serve the review gallery live: `pixi run icons-gallery`, then open the printed URL.

Clicking an icon flags it for regeneration (POST /queue); a running build repaints
flagged icons first and clears the flag.

`gallery.json` is computed from the icon folder on every request, so the page shows
icons as a running batch finishes them, including a batch started before this server.
Bound to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import abilities as ability_data
import gallery

HERE = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, out: Path, every: dict, backend: str, **kwargs):
        self.out, self.every, self.backend = out, every, backend
        super().__init__(*args, directory=str(out), **kwargs)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/gallery.html"):
            concepts = json.loads((HERE / "concepts.json").read_text())
            gallery.write(self.out / "gallery.html", self.out / "icons" / "128", self.every, concepts, self.backend)
            self.path = "/gallery.html"
        elif path == "/gallery.json":
            concepts = json.loads((HERE / "concepts.json").read_text())
            body = json.dumps(gallery.entries(self.out / "icons" / "128", self.out, self.every, concepts)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        """Flag or un-flag an icon for regeneration: an empty file per key in queue/."""
        if self.path.split("?", 1)[0] != "/queue":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 2) or b"{}")
        key, queued = str(body.get("key", "")), bool(body.get("queued"))
        # Keys come from the page, so only accept ones that are real abilities: this is
        # what keeps a crafted request from naming a path outside the queue folder.
        if key not in self.every:
            self.send_error(400, "unknown ability key")
            return
        queue = self.out / "queue"
        queue.mkdir(parents=True, exist_ok=True)
        flag = queue / key
        if queued:
            flag.touch()
        else:
            flag.unlink(missing_ok=True)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_: object) -> None:
        pass  # one line per poll every five seconds is noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="main", help="which output set to serve")
    parser.add_argument("--repo", type=Path, default=Path.home() / "code" / "bevy-rpg")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    out = HERE / "output" / args.set
    every = ability_data.load(args.repo)
    # The reroll command the page builds names a backend; Cloudflare is the fast one.
    handler = functools.partial(Handler, out=out, every=every, backend="cloudflare")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Live gallery: http://127.0.0.1:{args.port}/gallery.html  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
