"""The prompt a picture is painted from: `subject, rendered in <colours>, <style>`.

One assembly for the build that sends it and the gallery that shows it, so what the
review page displays is exactly what the next repaint sends. The style lines live in
styles.json, editable from the gallery; a build reads them fresh for every picture,
so a running `--watch` build picks up an edit without a restart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from abilities import Ability
from compose import palette

HERE = Path(__file__).resolve().parent
STYLE_FILE = HERE / "styles.json"

# What the style lines have learned, for whoever rewords them:
# - Describe only what is wanted. These models run without CFG, so negative prompts are
#   inert, and naming an unwanted thing in the positive prompt ("no circle") tends to
#   draw it. "Emblem" and "symbol" produce ornate seals that blur into look-alike
#   medallions at icon size; a single concrete subject stays legible.
# - Keep "on a plain black background": the composer cuts the subject out by its
#   difference from the background, so a busy backdrop comes through as the picture.
# - Portraits: naming an old master pulled renders photoreal. "Head and upper body",
#   not "head and shoulders", which a worm or an ooze does not have. A bare "muted
#   limited palette" greyed every face to one pallor, so it is scoped to clothing and
#   metal and skin keeps its colour; that is the quickest tell of an ancestry. "Gaunt"
#   and "grim" keep faces off pretty anime and grinning cartoons; "weathered" aged
#   everyone. "A stylised face" outvoted any ancestry without a human head. No fixed
#   pose: the subject carries its own angle and light (portraits.Variety).
# - Monsters: "portrait", "upper body" and gold "on clothing" describe a dressed bust,
#   so every boar and shark came out robed. The monster line names only a body in its
#   natural hide and never clothing, which at these settings would draw it.
STYLE_KEYS = ("icons", "portraits", "monsters")
# Only read with CFG above 1; at the distilled models' settings it is inert, which is
# why the composer fades picture edges rather than relying on it.
NEGATIVE = "text, letters, numbers, words, border, frame, photo, multiple objects, clutter, watermark"


def style_key(records: str, portrait: bool) -> str:
    return "monsters" if records == "monsters" else "portraits" if portrait else "icons"


def styles() -> dict[str, str]:
    return json.loads(STYLE_FILE.read_text())


def save_style(key: str, text: str) -> None:
    """Replaced atomically: a build reads the file for every picture it paints."""
    data = styles()
    data[key] = text
    temporary = STYLE_FILE.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(temporary, STYLE_FILE)


def colours(ability: Ability, portrait: bool) -> str:
    """Asking for the palette's colours up front means the gradient map refines the
    picture rather than fighting it — except for a shared picture, whose records take
    different palettes, so it is painted neutral and only recoloured. A portrait is
    painted in its own colours and never recoloured."""
    return "" if ability.picture or portrait else f", rendered in {palette(ability)['words']}"


def subject(ability: Ability, concepts: dict[str, str]) -> str:
    """A written concept is one concrete image ("a hawk on a gloved fist"); the record's
    own text is the fallback, and often has nothing drawable in it."""
    return concepts.get(ability.picture or ability.key) or ability.prompt_subject()


def prompt(ability: Ability, concepts: dict[str, str], records: str, portrait: bool,
           style: dict[str, str] | None = None) -> str:
    style = style if style is not None else styles()
    return f"{subject(ability, concepts)}{colours(ability, portrait)}, {style[style_key(records, portrait)]}"
