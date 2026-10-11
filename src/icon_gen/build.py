"""Render and compose an icon for every ability in a bevy-rpg checkout.

    pixi run icons --sample 20                 # a spread across ability types, for review
    pixi run icons --only fireball,heal        # specific ability keys
    pixi run icons --all                       # every ability (resumable; run overnight)
    pixi run icons --report                    # coverage only, renders nothing
    pixi run icons --reroll hermit,frostbite   # throw away these pictures and paint new ones
    pixi run icons --all --model flux --reverse   # local FLUX from the back while Cloudflare takes the front
    pixi run icons --backend cloudflare --watch   # regenerate whatever gets flagged in the live gallery

For each ability:
1. **Picture** — the AI paints one concrete subject in full colour on a plain black
   background, from the ability's line in concepts.json (falling back to the record's
   own name and description). Cached: re-rendered only when its prompt or seed
   changes, so an interrupted `--all` picks up where it stopped.
2. **Compose** — frame, picture, colour coding and badges from the record's data
   (compose.py).
3. **Uniqueness** — any two finished icons too alike to tell apart get the later key's
   picture re-rolled with a new seed, up to `--max-rerolls` times.
4. **Coverage** — every ability key has an icon and a concept, and nothing is left for
   a deleted key.

Outputs land in src/icon_gen/output/<set>/ (default `main`, shared by both backends): `glyphs/` (raw pictures),
`icons/<size>/` (finished icons, named by ability key), `manifest.json`, `sheet.png`
(this run's batch), and `gallery.html` (every icon, labelled, click to build a
`--reroll` command).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

from PIL import Image

import abilities as ability_data
import gallery
from compose import compose, palette
from render import CloudflareRenderer, PromptRefused, QuotaExhausted, Renderer
from unique import collisions, fingerprint

HERE = Path(__file__).resolve().parent

# Describes only what we want. These models run without CFG, so negative prompts are
# inert, and naming an unwanted thing in the positive prompt ("no circle") tends to
# draw it. "Emblem" and "symbol" are avoided too: they produce ornate seals, which blur
# into look-alike medallions at icon size, where a single concrete subject stays legible.
# The look is antique engraving and occult tarot with a cosmic-horror mood, described
# rather than named after any game or artist.
STYLE = ("an eldritch cosmic-horror illustration in the style of an antique engraving and an "
         "occult tarot card, ominous and ancient, unsettling stylised forms, fine etched linework "
         "with subtle painterly texture, muted limited palette with tarnished gold accents, "
         "one subject centered on a plain black background")
# Only read with CFG above 1; at the distilled models' settings it is inert, which is
# why the composer fades picture edges rather than relying on it.
NEGATIVE = "text, letters, numbers, words, border, frame, photo, multiple objects, clutter, watermark"

# Art direction, one per ability key: a single concrete image drawn from the record's
# name, description and traits. Kept here rather than in the game's records because the
# engine never reads it.
CONCEPTS: dict[str, str] = json.loads((HERE / "concepts.json").read_text())


def seed_for(key: str, attempt: int) -> int:
    """Stable per key, so a re-run reproduces the same icon; a re-roll bumps it."""
    return zlib.crc32(key.encode()) % 1_000_000 + attempt * 1_000_003


class Queue:
    """Icons flagged for regeneration in the live gallery: one empty file per key in
    `<set>/queue/`. A build claims a flag by renaming it to `<key>.<pid>` — atomic, so
    when two builds fill the set at once exactly one repaints it — and deletes it once
    the new picture is in, which is what clears the flag on the page."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def flagged(self) -> list[str]:
        return sorted(p.name for p in self.directory.iterdir() if "." not in p.name)

    def claim(self, key: str) -> Path | None:
        claimed = self.directory / f"{key}.{os.getpid()}"
        try:
            os.rename(self.directory / key, claimed)
        except FileNotFoundError:
            return None  # un-flagged, or the other build got there first
        return claimed

    def release(self, claimed: Path, key: str) -> None:
        """Put a claim back unfinished (allowance ran out), so the flag stays up."""
        os.rename(claimed, self.directory / key)


class Build:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # One set per --set name, shared by every backend: the prompts are identical, so a
        # picture from either is interchangeable, and both can fill the same set at once.
        self.out = args.out / args.set
        self.glyphs = self.out / "glyphs"
        self.icons = self.out / "icons" / str(args.icon_size)
        self.glyphs.mkdir(parents=True, exist_ok=True)
        self.icons.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out / "manifest.json"
        self.manifest: dict[str, dict] = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        self._renderer: Renderer | CloudflareRenderer | None = None
        self.refused: list[str] = []
        self.bumped: dict[str, int] = {}  # re-rolled keys -> the attempt to render next
        self.queue = Queue(self.out / "queue")

    def renderer(self) -> Renderer | CloudflareRenderer:
        # Started on first use: a run with every picture cached never loads the model.
        if self._renderer is None:
            if self.args.backend == "cloudflare":
                self._renderer = CloudflareRenderer().__enter__()
            else:
                print(f"Loading {self.args.model} ...", flush=True)
                self._renderer = Renderer(self.args.model).__enter__()
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.__exit__()

    def glyph(self, ability: ability_data.Ability) -> bytes:
        # Read fresh: the other build filling this set may have painted it since we
        # started. A re-rolled key's bumped attempt is applied on top.
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        entry = self.manifest.get(ability.key, {})
        attempt = self.bumped.get(ability.key, entry.get("attempt", 0))
        # A written concept is one concrete image ("a hawk on a gloved fist"); the record's
        # own text is the fallback, and often has nothing drawable in it.
        subject = CONCEPTS.get(ability.key) or ability.prompt_subject()
        # Asking for the palette's colours up front means the gradient map refines the
        # picture rather than fighting it.
        prompt = f"{subject}, rendered in {palette(ability)['words']}, {STYLE}"
        seed = seed_for(ability.key, attempt)
        path = self.glyphs / f"{ability.key}.png"
        if path.exists() and entry.get("prompt") == prompt and entry.get("seed") == seed:
            return path.read_bytes()
        png = self.renderer().render(prompt, NEGATIVE, seed)
        path.write_bytes(png)
        self.record(ability.key, {"prompt": prompt, "seed": seed, "attempt": attempt})
        return png

    def record(self, key: str, entry: dict) -> None:
        """Save one key's manifest entry without losing anyone else's.

        Two builds can fill a set at once (Cloudflare from the front, local from the
        back), so the file is re-read and merged rather than overwritten from this
        process's copy, and replaced atomically so a reader never sees half of it.
        """
        on_disk = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        on_disk[key] = entry
        self.manifest = {**on_disk, **{key: entry}}
        temporary = self.manifest_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(on_disk, indent=1, sort_keys=True))
        os.replace(temporary, self.manifest_path)

    def drain(self, every: dict[str, ability_data.Ability]) -> int:
        """Repaint every icon flagged in the gallery, ahead of the run's own list."""
        done = 0
        for key in self.queue.flagged():
            if key not in every or (claimed := self.queue.claim(key)) is None:
                continue
            self.reroll(key)
            try:
                made = self.make(every[key])
            except QuotaExhausted:
                self.queue.release(claimed, key)
                raise
            claimed.unlink(missing_ok=True)
            done += 1
            print(f"  regenerated {key} (flagged in the gallery){'' if made else ', refused by the content filter'}", flush=True)
        return done

    def make(self, ability: ability_data.Ability) -> Path | None:
        try:
            picture = self.glyph(ability)
        except PromptRefused:
            self.refused.append(ability.key)
            return None
        icon = compose(ability, picture, self.args.icon_size)
        path = self.icons / f"{ability.key}.png"
        icon.save(path)
        return path

    def reroll(self, key: str) -> None:
        current = self.bumped.get(key, self.manifest.get(key, {}).get("attempt", 0))
        self.bumped[key] = current + 1


def sheet(paths: list[Path], dest: Path, cell: int) -> None:
    cols = math.ceil(math.sqrt(len(paths)))
    rows = math.ceil(len(paths) / cols)
    gap = max(4, cell // 16)
    image = Image.new("RGBA", (cols * (cell + gap) + gap, rows * (cell + gap) + gap), (40, 40, 46, 255))
    for i, path in enumerate(paths):
        with Image.open(path) as icon:
            image.alpha_composite(icon.convert("RGBA").resize((cell, cell), Image.Resampling.LANCZOS),
                                  (gap + (i % cols) * (cell + gap), gap + (i // cols) * (cell + gap)))
    image.save(dest)


def sample(all_abilities: dict[str, ability_data.Ability], count: int) -> list[str]:
    """A deterministic spread: round-robin over ability types."""
    by_kind: dict[str, list[str]] = {}
    for key in sorted(all_abilities):
        by_kind.setdefault(all_abilities[key].kind, []).append(key)
    picked: list[str] = []
    while len(picked) < count and any(by_kind.values()):
        for kind in sorted(by_kind):
            if by_kind[kind] and len(picked) < count:
                # Spread within a kind too, instead of its alphabetically first keys.
                keys = by_kind[kind]
                picked.append(keys.pop(zlib.crc32(kind.encode() + bytes([len(picked)])) % len(keys)))
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.home() / "code" / "bevy-rpg")
    parser.add_argument("--model", choices=("turbo", "flux"), default="turbo", help="local backend's model")
    parser.add_argument("--set", default="main", help="output set; every backend can fill the same one")
    parser.add_argument("--reverse", action="store_true",
                        help="work from the end of the list, so a second build can fill the set from the other side")
    parser.add_argument("--backend", choices=("local", "cloudflare"), default="local",
                        help="cloudflare: FLUX.1 schnell on Workers AI (seconds per image, daily free allowance)")
    parser.add_argument("--icon-size", type=int, default=128, help="export size; 128 stays sharp to 250%% UI scale")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--only", help="comma-separated ability keys")
    which.add_argument("--sample", type=int, help="this many abilities spread across types")
    which.add_argument("--all", action="store_true")
    which.add_argument("--report", action="store_true", help="coverage and collisions only")
    which.add_argument("--reroll", help="comma-separated ability keys whose pictures to replace (bad hands, wrong subject)")
    parser.add_argument("--watch", action="store_true",
                        help="after the list (or with no list), keep regenerating icons flagged in the live gallery until stopped")
    # Report-only by default: on dark engraved icons the shape hash matches pictures a
    # person tells apart at a glance, and every automatic re-roll spends allowance.
    parser.add_argument("--max-rerolls", type=int, default=0,
                        help="automatic re-rolls of look-alike icons (default 0: just list them)")
    parser.add_argument("--max-bits", type=int, default=40, help="shape distance (of 256) at or below which icons collide")
    parser.add_argument("--max-colour", type=float, default=40.0, help="colour distance at or below which icons collide")
    parser.add_argument("--export", type=Path, help="copy finished icons here, e.g. a bevy-rpg assets dir")
    parser.add_argument("--out", type=Path, default=HERE / "output")
    args = parser.parse_args()
    if not (args.only or args.sample or args.all or args.report or args.reroll or args.watch):
        parser.error("choose what to render: --only, --sample, --all, --reroll, --report or --watch")

    every = ability_data.load(args.repo)
    if not every:
        sys.exit(f"no ability records under {args.repo}/assets/data/en")
    if args.only or args.reroll:
        targets = (args.only or args.reroll).split(",")
        unknown = [key for key in targets if key not in every]
        if unknown:
            sys.exit(f"not ability keys in {args.repo}: {', '.join(unknown)}")
    elif args.sample:
        targets = sample(every, args.sample)
    elif args.all:
        targets = sorted(every)
    else:
        targets = []
    if args.reverse:
        targets = targets[::-1]

    build = Build(args)
    if args.reroll:
        # A new attempt number changes the cache key, so the next render replaces it;
        # the old picture is overwritten, not kept.
        for key in targets:
            build.reroll(key)
    clashes: list[tuple[str, str, int, float]] = []
    try:
        try:
            done = 0
            for done, key in enumerate(targets, start=1):
                # Flags set in the gallery jump the queue: someone is waiting on those.
                build.drain(every)
                made = build.make(every[key])
                print(f"[{done}/{len(targets)}] {key}{'' if made else '  (refused by the content filter)'}", flush=True)
                # A long batch runs for hours; keep the review page current as it goes.
                if done % 25 == 0:
                    gallery.write(build.out / "gallery.html", build.icons, every, CONCEPTS, args.backend)
            if not args.report:
                build.drain(every)
            while args.watch:
                if not build.drain(every):
                    time.sleep(10)
        except QuotaExhausted as spent:
            # Everything rendered so far is cached; re-running the same command resumes.
            print(f"Daily allowance used up after {max(done - 1, 0)} new picture(s): {spent}", flush=True)
            print("Re-run the same command after 00:00 UTC to continue.", flush=True)
            targets = targets[:max(done - 1, 0)]
            args.max_rerolls = 0  # re-rolls would need the allowance that just ran out

        # Uniqueness over every finished icon on disk, not only this run's, so a new
        # icon cannot quietly duplicate one rendered in an earlier batch.
        for round_ in range(args.max_rerolls + 1):
            prints = {p.stem: fingerprint(p) for p in build.icons.glob("*.png") if p.stem in every}
            clashes = collisions(prints, args.max_bits, args.max_colour)
            if not clashes or args.report or round_ == args.max_rerolls:
                break
            rerolled = sorted({later for _, later, _, _ in clashes})
            print(f"Uniqueness pass {round_ + 1}: re-rolling {len(rerolled)} picture(s)", flush=True)
            for key in rerolled:
                build.reroll(key)
                build.make(every[key])
        for a, b, bits, colour in clashes:
            print(f"  still alike: {a} / {b} (shape {bits}/256, colour {colour:.0f})")
    finally:
        build.close()

    present = {p.stem for p in build.icons.glob("*.png")}
    missing = sorted(set(every) - present)
    orphans = sorted(present - set(every))
    unconceived = sorted(set(every) - CONCEPTS.keys())
    print(f"Concepts: {len(every) - len(unconceived)}/{len(every)} abilities have one "
          f"(the rest fall back to their own name and description)")
    for key in sorted(CONCEPTS.keys() - set(every)):
        print(f"  concept for a key no longer in the data: {key}")
    print(f"Coverage: {len(present & set(every))}/{len(every)} abilities have an icon; "
          f"{len(missing)} missing, {len(orphans)} orphaned (key no longer in the data)")
    for key in orphans:
        print(f"  orphan: {key}")

    if build.refused:
        print(f"Refused by the content filter, reword their concepts: {', '.join(build.refused)}")
    shown = [build.icons / f"{key}.png" for key in targets if (build.icons / f"{key}.png").exists()]
    if shown:
        sheet(shown, build.out / "sheet.png", cell=args.icon_size)
        print(f"Contact sheet: {build.out / 'sheet.png'}")
    page = gallery.write(build.out / "gallery.html", build.icons, every, CONCEPTS, args.backend)
    print(f"Review gallery: {page}")
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        for key in sorted(present & set(every)):
            shutil.copy2(build.icons / f"{key}.png", args.export / f"{key}.png")
        print(f"Exported {len(present & set(every))} icons to {args.export}")


if __name__ == "__main__":
    main()
