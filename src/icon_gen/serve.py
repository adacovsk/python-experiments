"""Serve every icon set live on one page: `pixi run icons-gallery`, then open the URL.

Abilities (grouped by type), conditions, buff/debuff effects and player / NPC /
monster portraits, read fresh from disk
on every poll, so icons appear as running builds finish them. Clicking an icon flags
it for regeneration (POST /queue with its set); a build repaints flagged icons first
and clears the flag. POST /review marks icons reviewed (the page hides them), and
POST /concept rewrites a concept line, which running builds pick up. Bound to
127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import gallery
import portraits
from build import CONCEPT_FILES, PORTRAITS, default_size, load_records

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# (output set, records, heading or None to group by record type), in page order.
SETS = (
    ("main", "abilities", None),
    ("conditions", "conditions", "Conditions"),
    ("effects", "effects", "Buffs & debuffs"),
    # Portraits group by ancestry (players), role (NPCs) or environment (monsters), the
    # kind their records carry.
    ("players", "players", None),
    ("npcs", "npcs", None),
    ("monsters", "monsters", None),
)
KINDS = {set_name: kind for set_name, kind, _ in SETS}
# Review marks and concept files are read-modify-written; the server is threaded.
LOCK = threading.Lock()


def write_json(path: Path, data: dict) -> None:
    """In the concept files' own layout, replaced atomically: builds read them mid-run."""
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, records: dict[str, dict], repo: Path, **kwargs):
        self.records = records
        self.repo = repo
        super().__init__(*args, directory=str(OUTPUT), **kwargs)

    def all_entries(self) -> list[dict]:
        found = []
        parts = {**portraits.concepts(), "monster": portraits.monster_concepts()}
        for section, (set_name, kind, heading) in enumerate(SETS):
            concept_file = HERE / CONCEPT_FILES[kind]
            # A portrait's concept is its record's assembled subject, not a lookup.
            concepts = {} if kind in PORTRAITS or not concept_file.exists() else json.loads(concept_file.read_text())
            found += gallery.entries(OUTPUT / set_name / "icons" / str(default_size(kind)), OUTPUT, self.records[set_name],
                                     concepts, set_name, section, heading, parts)
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
        route = {"/queue": self.post_queue, "/review": self.post_review, "/concept": self.post_concept}
        handler = route.get(self.path.split("?", 1)[0])
        if handler is None:
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 2) or b"{}")
        set_name = str(body.get("set", ""))
        # Set and keys come from the page, so only accept a known set and its real keys:
        # that is what keeps a crafted request from naming a path outside the output.
        if set_name not in self.records:
            self.send_error(400, "unknown set")
            return
        handler(set_name, body)

    def reply(self, data: dict | None = None) -> None:
        if data is None:
            self.send_response(204)
            self.end_headers()
            return
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def post_queue(self, set_name: str, body: dict) -> None:
        """Flag or un-flag an icon for regeneration: an empty file per key in <set>/queue/."""
        key = str(body.get("key", ""))
        if key not in self.records[set_name]:
            self.send_error(400, "unknown key")
            return
        self.flag(set_name, key, bool(body.get("queued")))
        self.reply()

    def flag(self, set_name: str, key: str, queued: bool = True) -> None:
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

    def icons_dir(self, set_name: str) -> Path:
        return OUTPUT / set_name / "icons" / str(default_size(KINDS[set_name]))

    def post_review(self, set_name: str, body: dict) -> None:
        """Mark icons reviewed, as the mtime of the picture that was looked at."""
        keys = [str(k) for k in body.get("keys", [])]
        if not keys or any(k not in self.records[set_name] for k in keys):
            self.send_error(400, "unknown key")
            return
        icons = self.icons_dir(set_name)
        with LOCK:
            marks = gallery.reviewed_marks(icons)
            for key in keys:
                icon = icons / f"{key}.png"
                if body.get("reviewed") and icon.exists():
                    marks[key] = int(icon.stat().st_mtime)
                else:
                    marks.pop(key, None)
            write_json(gallery.reviewed_path(icons), marks)
        self.reply()

    def post_concept(self, set_name: str, body: dict) -> None:
        """Rewrite one concept line, then flag this record or every one that uses it.

        An icon set's line is its picture's, in that set's concept file; a portrait's
        is one of the parts its subject is assembled from. Empty text removes the
        line, so the record falls back to its own name and description."""
        key, part, part_key = str(body.get("key", "")), body.get("part"), str(body.get("part_key", ""))
        record = self.records[set_name].get(key)
        text = " ".join(str(body.get("text", "")).split())
        regenerate = body.get("regenerate")
        kind = KINDS[set_name]
        if record is None or regenerate not in ("this", "all", "none"):
            self.send_error(400, "unknown key or regenerate mode")
            return
        if kind in PORTRAITS:
            if (part, part_key) not in record.parts:
                self.send_error(400, "not a part of this portrait")
                return
            path = portraits.MONSTER_CONCEPT_FILE if part == "monster" else portraits.CONCEPT_FILE
            users = [k for k, r in self.records[set_name].items() if (part, part_key) in r.parts]
        else:
            if part is not None or part_key != (record.picture or key):
                self.send_error(400, "not this icon's concept")
                return
            path = HERE / CONCEPT_FILES[kind]
            users = [k for k, r in self.records[set_name].items() if (r.picture or k) == part_key]
        with LOCK:
            data = json.loads(path.read_text()) if path.exists() else {}
            lines = data if part in (None, "monster") else data.setdefault(part, {})
            if text:
                lines[part_key] = text
            else:
                lines.pop(part_key, None)
            write_json(path, data)
            if kind in PORTRAITS:
                # Their subjects are assembled at load, so re-assemble every portrait set.
                for name, k in KINDS.items():
                    if k in PORTRAITS:
                        self.records[name].clear()
                        self.records[name].update(load_records(k, self.repo))
        flagged = {"this": [key], "all": users, "none": []}[regenerate]
        for k in flagged:
            self.flag(set_name, k)
        self.reply({"flagged": len(flagged)})

    def log_message(self, *_: object) -> None:
        pass  # one line per poll every five seconds is noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "code" / "bevy-rpg")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    records = {set_name: load_records(kind, args.repo) for set_name, kind, _ in SETS}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, records=records, repo=args.repo))
    print(f"Live gallery: http://127.0.0.1:{args.port}/gallery.html  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
