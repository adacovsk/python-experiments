"""Serve every icon set live on one page: `pixi run icons-gallery`, then open the URL.

Abilities (grouped by type), conditions and buff/debuff effects, read fresh from disk
on every poll, so icons appear as running builds finish them. Clicking an icon flags
it for regeneration (POST /queue with its set); a build repaints flagged icons first
and clears the flag. Bound to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import gallery
from build import CONCEPT_FILES, load_records

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# (output set, records, heading or None to group by record type), in page order.
SETS = (
    ("main", "abilities", None),
    ("conditions", "conditions", "Conditions"),
    ("effects", "effects", "Buffs & debuffs"),
)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, records: dict[str, dict], **kwargs):
        self.records = records
        super().__init__(*args, directory=str(OUTPUT), **kwargs)

    def all_entries(self) -> list[dict]:
        found = []
        for section, (set_name, kind, heading) in enumerate(SETS):
            concept_file = HERE / CONCEPT_FILES[kind]
            concepts = json.loads(concept_file.read_text()) if concept_file.exists() else {}
            found += gallery.entries(OUTPUT / set_name / "icons" / "128", OUTPUT, self.records[set_name],
                                     concepts, set_name, section, heading)
        return found

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/gallery.html"):
            gallery.write_page(OUTPUT / "gallery.html", self.all_entries(), "cloudflare")
            self.path = "/gallery.html"
        elif path == "/gallery.json":
            body = json.dumps(self.all_entries()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        """Flag or un-flag an icon for regeneration: an empty file per key in <set>/queue/."""
        if self.path.split("?", 1)[0] != "/queue":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 2) or b"{}")
        set_name, key, queued = str(body.get("set", "")), str(body.get("key", "")), bool(body.get("queued"))
        # Both come from the page, so only accept a known set and one of its real keys:
        # that is what keeps a crafted request from naming a path outside a queue folder.
        if key not in self.records.get(set_name, {}):
            self.send_error(400, "unknown set or key")
            return
        queue = OUTPUT / set_name / "queue"
        queue.mkdir(parents=True, exist_ok=True)
        flag = queue / key
        if queued:
            # Already being regenerated: the claim is the flag, so a second one would
            # only queue a needless second repaint once this one finishes.
            if not any(queue.glob(f"{key}.*")):
                flag.touch()
        else:
            flag.unlink(missing_ok=True)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_: object) -> None:
        pass  # one line per poll every five seconds is noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "code" / "bevy-rpg")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    records = {set_name: load_records(kind, args.repo) for set_name, kind, _ in SETS}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, records=records))
    print(f"Live gallery: http://127.0.0.1:{args.port}/gallery.html  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
