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
import io
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
import portraits
from compose import PALETTE, compose, compose_portrait, palette
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
# Portraits are lit like chiaroscuro but painted as illustrations: naming an old master
# pulled the renders photoreal. A face read out of the dark by one hard light, framed as
# a bust, has to stand on its own at gallery size. "Head and
# upper body" rather than "head and shoulders", which a worm or an ooze does not have.
# A bare "muted limited palette" greyed every face to the same pallor, so a goblin's
# green or a nephilim's gold, the quickest tell of an ancestry, never came through: the
# palette is scoped to clothing and metal, and skin is named as keeping its colour.
# Otherwise the vocabulary is the icons' STYLE, so a portrait reads as the same hand:
# etched engraving linework, tarnished gold. "Weathered" and "gaunt" keep the model off
# the pretty anime and grinning cartoon faces a plain "portrait" drifts into.
# No fixed pose either: the subject carries its own angle and light (portraits.Variety).
PORTRAIT_STYLE = ("an eldritch dark-fantasy portrait in the style of an antique engraving and an occult "
                  "tarot card, fine etched ink linework with cross-hatched shading and subtle painterly "
                  "texture, a weathered stylised face with gaunt, grim features, ominous and ancient, "
                  "muted limited palette with tarnished gold accents on clothing and metal, skin, eyes "
                  "and hair keeping their own natural colours, one hard light from the side against deep "
                  "shadow, clearly an illustration rather than a photograph, the head and upper body "
                  "filling the frame, the face clearly lit, on a plain black background")
NEGATIVE = "text, letters, numbers, words, border, frame, photo, multiple objects, clutter, watermark"

# Art direction, one per ability key: a single concrete image drawn from the record's
# name, description and traits. Kept here rather than in the game's records because the
# engine never reads it.
# Conditions state their side in palette.toml; they have no mechanics to read it from.
PALETTE_SIDES: dict[str, list[str]] = PALETTE.get("condition_side", {})
CONCEPT_FILES = {"abilities": "concepts.json", "conditions": "concepts_conditions.json", "effects": "concepts_effects.json",
                 "players": portraits.CONCEPT_FILE.name, "npcs": portraits.CONCEPT_FILE.name,
                 "monsters": portraits.MONSTER_CONCEPT_FILE.name}
# Record sets that are portraits rather than icons: assembled subjects, their own style,
# composed without palette or badges, at a larger size.
PORTRAITS = ("players", "npcs", "monsters")


def default_size(records: str) -> int:
    """128 keeps an icon sharp to 250% UI scale; a portrait is shown larger."""
    return 256 if records in PORTRAITS else 128
CONCEPTS: dict[str, str] = json.loads((HERE / CONCEPT_FILES["abilities"]).read_text())


def load_records(kind: str, repo: Path) -> dict[str, ability_data.Ability]:
    """The records one set renders: abilities, conditions, buff/debuff effects, or portraits."""
    if kind == "abilities":
        return ability_data.load(repo)
    if kind == "players":
        return portraits.load_players(repo)
    if kind == "npcs":
        return portraits.load_npcs(repo)
    if kind == "monsters":
        return portraits.load_monsters(repo)
    conditions = ability_data.load_conditions(repo, PALETTE_SIDES)
    if kind == "conditions":
        return conditions
    return ability_data.load_effects(repo, set(conditions))


def seed_for(key: str, attempt: int) -> int:
    """Stable per key, so a re-run reproduces the same icon; a re-roll bumps it."""
    return zlib.crc32(key.encode()) % 1_000_000 + attempt * 1_000_003


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def write_atomically(path: Path, data: bytes) -> None:
    """Readers (the gallery, the other build) see the old file or the new one, never half."""
    temporary = path.with_suffix(f".{os.getpid()}.part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class Queue:
    """Icons flagged for regeneration in the live gallery: one empty file per key in
    `<set>/queue/`. A build claims a flag by renaming it to `<key>.<pid>` — atomic, so
    when two builds fill the set at once exactly one repaints it — and deletes it once
    the new picture is in, which is what clears the flag on the page."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def flagged(self) -> list[str]:
        self.recover()
        return sorted(p.name for p in self.directory.iterdir() if "." not in p.name)

    def recover(self) -> None:
        """Release claims whose process is gone (killed, crashed, rebooted), so their
        flags go back in the queue instead of reading "regenerating" forever."""
        for claimed in self.directory.glob("*.*"):
            key, _, pid = claimed.name.rpartition(".")
            if not pid.isdigit() or alive(int(pid)):
                continue
            try:
                os.rename(claimed, self.directory / key)
            except FileNotFoundError:
                pass  # another build recovered it first

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
        self.pictures: dict[str, str] = {}  # record key -> shared picture key, where one is shared
        self.portrait = args.records in PORTRAITS
        self.concept_file = HERE / CONCEPT_FILES[args.records]
        self.concepts_seen = self.concept_stamp()

    def concept_stamp(self) -> int:
        return self.concept_file.stat().st_mtime_ns if self.concept_file.exists() else 0

    def refresh(self, every: dict[str, ability_data.Ability]) -> None:
        """Pick up concept lines edited in the gallery since this build started: a
        `--watch` build runs for days, and would otherwise repaint an edited icon
        from the line it was trying to replace."""
        global CONCEPTS
        stamp = self.concept_stamp()
        if stamp == self.concepts_seen:
            return
        self.concepts_seen = stamp
        if self.portrait:
            # A portrait's concepts are folded into its record's subject at load.
            every.clear()
            every.update(load_records(self.args.records, self.args.repo))
        else:
            CONCEPTS = json.loads(self.concept_file.read_text())

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
        picture = ability.picture or ability.key
        entry = self.manifest.get(picture, {})
        attempt = self.bumped.get(picture, entry.get("attempt", 0))
        # A written concept is one concrete image ("a hawk on a gloved fist"); the record's
        # own text is the fallback, and often has nothing drawable in it.
        subject = CONCEPTS.get(picture) or ability.prompt_subject()
        # Asking for the palette's colours up front means the gradient map refines the
        # picture rather than fighting it — except for a shared picture, whose records
        # take different palettes, so it is painted neutral and only recoloured.
        # A portrait is painted in its own colours and never recoloured.
        colours = "" if ability.picture or self.portrait else f", rendered in {palette(ability)['words']}"
        prompt = f"{subject}{colours}, {PORTRAIT_STYLE if self.portrait else STYLE}"
        seed = seed_for(picture, attempt)
        path = self.glyphs / f"{picture}.png"
        if path.exists() and entry.get("prompt") == prompt and entry.get("seed") == seed:
            return path.read_bytes()
        png = self.renderer().render(prompt, NEGATIVE, seed)
        write_atomically(path, png)
        self.record(picture, {"prompt": prompt, "seed": seed, "attempt": attempt})
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
        self.refresh(every)
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

    def approved(self, key: str) -> bool:
        """Approved in the gallery and not replaced since: the mark is the icon's mtime."""
        path = self.icons / f"{key}.png"
        mark = gallery.reviewed_marks(self.icons).get(key)
        return mark is not None and path.exists() and int(path.stat().st_mtime) == mark

    def make(self, ability: ability_data.Ability) -> Path | None:
        # An approved icon is final: a later style or prompt change must not repaint it and
        # send it back to review. Only an explicit re-roll or gallery flag replaces it.
        if ability.key not in self.bumped and self.approved(ability.key):
            return self.icons / f"{ability.key}.png"
        try:
            picture = self.glyph(ability)
        except PromptRefused:
            self.refused.append(ability.key)
            return None
        icon = (compose_portrait(picture, self.args.icon_size) if self.portrait
                else compose(ability, picture, self.args.icon_size))
        path = self.icons / f"{ability.key}.png"
        encoded = io.BytesIO()
        icon.save(encoded, format="PNG")
        # Leave an unchanged icon untouched: an approval is the icon's mtime, and an `--all`
        # pass re-composes every cached picture, so rewriting identical bytes un-approved them.
        if not path.exists() or path.read_bytes() != encoded.getvalue():
            write_atomically(path, encoded.getvalue())
        return path

    def reroll(self, key: str) -> None:
        # Re-rolling one direction of an effect repaints the picture both share.
        key = self.pictures.get(key, key)
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
    parser.add_argument("--records", choices=tuple(CONCEPT_FILES), default="abilities",
                        help="what to make icons for: abilities, conditions, buff/debuff effects, "
                             "or player / NPC / monster portraits")
    parser.add_argument("--set", help="output set (default: main for abilities, else the records name)")
    parser.add_argument("--reverse", action="store_true",
                        help="work from the end of the list, so a second build can fill the set from the other side")
    parser.add_argument("--backend", choices=("local", "cloudflare"), default="local",
                        help="cloudflare: FLUX.1 schnell on Workers AI (seconds per image, daily free allowance)")
    parser.add_argument("--icon-size", type=int, help="export size (default 128 for icons, 256 for portraits)")
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

    global CONCEPTS
    args.icon_size = args.icon_size or default_size(args.records)
    concept_file = HERE / CONCEPT_FILES[args.records]
    # A portrait's concepts are per part and already folded into its record's subject.
    CONCEPTS = {} if args.records in PORTRAITS else (json.loads(concept_file.read_text()) if concept_file.exists() else {})
    args.set = args.set or ("main" if args.records == "abilities" else args.records)
    every = load_records(args.records, args.repo)
    if not every:
        sys.exit(f"no {args.records} records under {args.repo}/assets/data/en")
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
    build.pictures = {key: record.picture for key, record in every.items() if record.picture}
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
        # icon cannot quietly duplicate one rendered in an earlier batch. Not for
        # portraits: every bust on black is "alike" to a shape hash, and the pairs would
        # number in the thousands; they are told apart by eye in the gallery.
        for round_ in range(0 if build.portrait else args.max_rerolls + 1):
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
    if build.portrait:
        gaps = portraits.missing_parts(args.repo, args.records)
        print(f"Concepts: {'every' if not gaps else 'not every'} {' and '.join(portraits.PARTS[args.records])} has one"
              f"{' (the rest fall back to their name)' if gaps else ''}")
        for gap in gaps:
            print(f"  concept missing or stale: {gap}")
    else:
        needed = {record.picture or key for key, record in every.items()}
        unconceived = sorted(needed - CONCEPTS.keys())
        print(f"Concepts: {len(needed) - len(unconceived)}/{len(needed)} pictures have one "
              f"(the rest fall back to their own name and description)")
        for key in sorted(CONCEPTS.keys() - needed):
            print(f"  concept for a key no longer in the data: {key}")
    print(f"Coverage: {len(present & set(every))}/{len(every)} {args.records} have {'a portrait' if build.portrait else 'an icon'}; "
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
