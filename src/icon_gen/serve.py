"""Serve every icon set live on one page: `pixi run icons-gallery`, then open the URL.

Abilities (grouped by type), conditions, buff/debuff effects and player / NPC /
monster portraits, read fresh from disk
on every poll, so icons appear as running builds finish them. Clicking an icon flags
it for regeneration (POST /queue with its set); a build repaints flagged icons first
and clears the flag. POST /review marks icons reviewed (the page hides them), and
POST /concept rewrites a concept line, which running builds pick up. POST /improve
flags an icon for an edit of its current picture from an instruction, and POST /undo
for putting back the picture from before the last edit, the original, or the picture
a repaint replaced. GET /prompt.json shows the full prompt an icon is painted from, and
POST /style rewrites the style line every prompt in its sets ends with. POST /open
puts a working copy of an icon's picture in <set>/manual/ and opens it in the image
editor; once the copy is saved over, the next poll takes it as the icon's picture.
Bound to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from PIL import Image

import gallery
import portraits
import prompts
from build import CONCEPT_FILES, PORTRAITS, Build, default_size, load_records
from paths import EDITOR, OUTPUT, REPO

HERE = Path(__file__).resolve().parent

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
        self.take_hand_edits()
        found = []
        parts = {**portraits.concepts(), "monster": portraits.monster_concepts()}
        for section, (set_name, kind, heading) in enumerate(SETS):
            found += gallery.entries(OUTPUT / set_name / "icons" / str(default_size(kind)), OUTPUT, self.records[set_name],
                                     self.concepts(kind), set_name, section, heading, parts)
        return found

    def concepts(self, kind: str) -> dict[str, str]:
        """An icon set's concept lines; a portrait's are already folded into its subject."""
        concept_file = HERE / CONCEPT_FILES[kind]
        return {} if kind in PORTRAITS or not concept_file.exists() else json.loads(concept_file.read_text())

    def prompt_info(self, query: dict[str, list[str]]) -> dict | None:
        """The pieces of one icon's prompt as the build assembles it, the style's sets,
        and the prompt its current picture was actually painted from."""
        set_name, key = query.get("set", [""])[0], query.get("key", [""])[0]
        record = self.records.get(set_name, {}).get(key)
        if record is None:
            return None
        kind = KINDS[set_name]
        portrait = kind in PORTRAITS
        style_key = prompts.style_key(kind, portrait)
        manifest_path = OUTPUT / set_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return {"subject": prompts.subject(record, self.concepts(kind)), "colours": prompts.colours(record, portrait),
                "style_key": style_key, "style": prompts.styles()[style_key],
                "style_sets": [name for name, k, _ in SETS if prompts.style_key(k, k in PORTRAITS) == style_key],
                "painted": manifest.get(record.picture or key, {}).get("prompt", "")}

    def send_json(self, data: object) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/prompt.json":
            info = self.prompt_info(parse_qs(query))
            if info is None:
                self.send_error(400, "unknown set or key")
            else:
                self.send_json(info)
            return
        if path in ("/", "/gallery.html"):
            gallery.write_page(OUTPUT / "gallery.html", self.all_entries(), "cloudflare")
            self.path = "/gallery.html"
        elif path == "/gallery.json":
            self.send_json(self.all_entries())
            return
        super().do_GET()

    def do_POST(self) -> None:
        route = {"/queue": self.post_queue, "/review": self.post_review, "/concept": self.post_concept,
                 "/improve": self.post_improve, "/undo": self.post_undo, "/style": self.post_style,
                 "/open": self.post_open}
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

    def ask(self, set_name: str, key: str, payload: dict) -> None:
        """Flag an icon with a request a build reads from the flag (build.Queue).

        Written beside the queue and moved in, so a build never claims a half-written
        flag; and written even while the icon is being handled, unlike a plain flag,
        since the instruction is new work, not a repeat of the running one."""
        queue = OUTPUT / set_name / "queue"
        queue.mkdir(parents=True, exist_ok=True)
        temporary = queue.parent / f".{key}.{os.getpid()}.request"
        temporary.write_text(json.dumps(payload))
        os.replace(temporary, queue / key)

    def post_improve(self, set_name: str, body: dict) -> None:
        """Ask for the current picture edited as an instruction says, not repainted."""
        key, text = str(body.get("key", "")), " ".join(str(body.get("text", "")).split())
        if key not in self.records[set_name] or not text:
            self.send_error(400, "unknown key or empty instruction")
            return
        self.ask(set_name, key, {"improve": text})
        self.reply()

    def post_undo(self, set_name: str, body: dict) -> None:
        """Ask for the picture from before the last edit, with `revert` the original, or
        with `previous` the picture a repaint replaced."""
        key = str(body.get("key", ""))
        if key not in self.records[set_name]:
            self.send_error(400, "unknown key")
            return
        which = "previous" if body.get("previous") else "revert" if body.get("revert") else "undo"
        self.ask(set_name, key, {which: True})
        self.reply()

    def build(self, set_name: str) -> Build:
        """This set as a build sees it, for the work that needs no model."""
        kind = KINDS[set_name]
        return Build(argparse.Namespace(out=OUTPUT, set=set_name, records=kind, repo=self.repo,
                                        icon_size=default_size(kind), backend="cloudflare", model=""))

    def post_open(self, set_name: str, body: dict) -> None:
        """Open an icon's picture in the image editor, as a working copy in <set>/manual/.

        The copy rather than the picture itself: an editor saves in place and not
        atomically, and a build may repaint or read the picture at any moment. A copy
        saved since the picture was painted is unfinished work and is opened as it is."""
        key = str(body.get("key", ""))
        record = self.records[set_name].get(key)
        command = shlex.split(EDITOR)
        if record is None or not command or shutil.which(command[0]) is None:
            self.send_error(400, "unknown key, or ICON_GEN_EDITOR names no installed editor")
            return
        picture = record.picture or key
        source = OUTPUT / set_name / "glyphs" / f"{picture}.png"
        if not source.exists():
            self.send_error(400, "no picture yet")
            return
        copy = OUTPUT / set_name / "manual" / f"{picture}.png"
        copy.parent.mkdir(exist_ok=True)
        with LOCK:
            if not copy.exists() or copy.stat().st_mtime_ns <= source.stat().st_mtime_ns:
                shutil.copy2(source, copy)
        # In a unit of its own where systemd runs the session: a child of this server
        # would be stopped with it, taking unsaved work, whenever the gallery restarts.
        if shutil.which("systemd-run"):
            command = ["systemd-run", "--user", "--collect", "--quiet", "--", *command]
        subprocess.Popen([*command, str(copy)], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.reply({"opened": str(copy)})

    def take_hand_edits(self) -> None:
        """Take every working copy saved since its picture was painted as that picture.

        Checked on each poll, so saving in the editor is all it takes. A copy still
        being written does not load as an image, or is under a second old, and waits
        for the next poll."""
        with LOCK:
            for set_name in self.records:
                for copy in sorted((OUTPUT / set_name / "manual").glob("*.png")):
                    picture = copy.stem
                    source = OUTPUT / set_name / "glyphs" / copy.name
                    key = next((k for k, r in self.records[set_name].items() if (r.picture or k) == picture), None)
                    stamp = copy.stat().st_mtime_ns
                    if (key is None or not source.exists() or stamp <= source.stat().st_mtime_ns
                            or time.time_ns() - stamp < 1_000_000_000):
                        continue
                    try:
                        with Image.open(copy) as image:
                            encoded = io.BytesIO()
                            # Flattened onto the backdrop the composer cuts subjects out of.
                            flat = Image.new("RGB", image.size, (0, 0, 0))
                            flat.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
                            flat.save(encoded, format="PNG")
                    except (OSError, SyntaxError):
                        continue
                    with Image.open(source) as current:
                        unchanged = current.size == flat.size and current.convert("RGB").tobytes() == flat.tobytes()
                    if unchanged:
                        os.utime(source)  # saved without a change: nothing to take, and not asked again
                        continue
                    self.build(set_name).by_hand(key, encoded.getvalue(), self.records[set_name])

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

    def post_style(self, set_name: str, body: dict) -> None:
        """Rewrite the style line of this set's prompts, and flag this icon if asked.

        Not empty: the composer cuts the subject out against a plain background, and a
        prompt with no style asks for none."""
        key, text = str(body.get("key", "")), " ".join(str(body.get("text", "")).split())
        regenerate = body.get("regenerate")
        if key not in self.records[set_name] or not text or regenerate not in ("this", "none"):
            self.send_error(400, "unknown key, empty style or bad regenerate mode")
            return
        kind = KINDS[set_name]
        with LOCK:
            prompts.save_style(prompts.style_key(kind, kind in PORTRAITS), text)
        if regenerate == "this":
            self.flag(set_name, key)
        self.reply({"flagged": int(regenerate == "this")})

    def log_message(self, *_: object) -> None:
        pass  # one line per poll every five seconds is noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    records = {set_name: load_records(kind, args.repo) for set_name, kind, _ in SETS}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, records=records, repo=args.repo))
    print(f"Live gallery: http://127.0.0.1:{args.port}/gallery.html  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
