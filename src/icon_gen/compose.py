"""Compose a finished icon from a greyscale glyph render and the ability's own data
(and a portrait from its render alone: `compose_portrait`).

Layers, back to front:
- **frame** — a shape per ability type (`[frame]` in palette.toml), dark fill, metal rim;
- **picture** — the AI render in its own colours, cut from its background and scaled
  to fill the frame;
- **colour coding** — the palette that applies (damage type, else harm/heal/help,
  else tradition, else ability type): the picture is gradient-mapped onto its ramp,
  and the frame's glow and inner rim take its middle colour;
- **badges** — action cost pips top-left, area shape mark bottom-right, drawn as
  vectors so they are crisp at any export size.

Composed at 4x the export size and downscaled once, for clean antialiasing.
"""
from __future__ import annotations

import io
import math
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from abilities import Ability

HERE = Path(__file__).resolve().parent
PALETTE = tomllib.loads((HERE / "palette.toml").read_text())

BACKGROUND = (16, 16, 22, 255)
RIM = (176, 168, 152, 255)
BADGE = (240, 232, 214, 255)
BADGE_SHADOW = (0, 0, 0, 200)


def palette(ability: Ability) -> dict:
    """The first palette that applies, in palette.toml's order: damage type, then
    harm/heal/help, then tradition, then ability type. What it does outranks where it
    comes from: a divine heal takes the healing palette, not the divine one."""
    for table, value in (
        ("damage_type", ability.damage_type),
        # "neutral" names no side palette, so it falls through to the ability type's.
        ("side", ability.side),
        ("tradition", ability.tradition),
        ("ability_type", ability.kind),
    ):
        entry = PALETTE[table].get(value or "")
        if entry:
            return entry
    return PALETTE["ability_type"]["General"]


def gradient_map(rgb: np.ndarray, ramp: list[list[int]], blend: float) -> np.ndarray:
    """Recolour by brightness: each pixel's luminance picks its colour off the ramp, so
    the picture keeps its shading while every icon in a palette shares one scheme."""
    luminance = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    stops = np.linspace(0.0, 1.0, len(ramp))
    colours = np.asarray(ramp, dtype=np.float32)
    mapped = np.stack([np.interp(luminance, stops, colours[:, c]) for c in range(3)], axis=-1)
    return mapped * blend + rgb * (1.0 - blend)


def _polygon(sides: int, size: int, inset: float, rotation: float = -math.pi / 2) -> list[tuple[float, float]]:
    c, r = size / 2, size / 2 - inset
    return [(c + r * math.cos(rotation + 2 * math.pi * i / sides), c + r * math.sin(rotation + 2 * math.pi * i / sides))
            for i in range(sides)]


def frame_mask(shape: str, size: int, inset: float) -> Image.Image:
    """A filled mask of the frame shape; every shape name is one palette.toml can use."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    box = (inset, inset, size - inset, size - inset)
    if shape == "circle":
        draw.ellipse(box, fill=255)
    elif shape == "rounded":
        draw.rounded_rectangle(box, radius=size * 0.16, fill=255)
    elif shape == "square":
        draw.rectangle(box, fill=255)
    elif shape == "diamond":
        draw.polygon(_polygon(4, size, inset), fill=255)
    elif shape == "hexagon":
        draw.polygon(_polygon(6, size, inset, 0.0), fill=255)
    elif shape == "octagon":
        draw.polygon(_polygon(8, size, inset, math.pi / 8), fill=255)
    elif shape == "shield":
        w, h = size - 2 * inset, size - 2 * inset
        draw.polygon([(inset, inset), (inset + w, inset), (inset + w, inset + h * 0.55),
                      (size / 2, inset + h), (inset, inset + h * 0.55)], fill=255)
    elif shape == "pentagon":
        draw.polygon(_polygon(5, size, inset), fill=255)
    elif shape == "star":
        c = size / 2
        points = []
        for i in range(16):
            r = (size / 2 - inset) * (1.0 if i % 2 == 0 else 0.82)
            a = -math.pi / 2 + math.pi * i / 8
            points.append((c + r * math.cos(a), c + r * math.sin(a)))
        draw.polygon(points, fill=255)
    else:
        raise ValueError(f"unknown frame shape {shape!r}")
    return mask


def glyph_layer(render_png: bytes, box: int, ramp: list[list[int]], blend: float) -> Image.Image:
    """The render's subject, cut from its background, recoloured onto its palette's
    ramp, cropped to its own extent, scaled to fill `box`, with a dark outline.

    The prompt asks for a plain black background, so the subject is whatever differs
    from it; the background colour is read off the render's border rather than assumed.
    Cropping to the subject's bounding box is what makes it large: the model rarely
    fills its canvas, and placing the whole render left the subject a fraction of the
    icon. Dark strokes inside the subject cut out too, which is invisible: the frame's
    fill behind them is just as dark.
    """
    with Image.open(io.BytesIO(render_png)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    n = rgb.shape[0]
    edge = max(4, n // 25)
    border = np.concatenate([rgb[:edge].reshape(-1, 3), rgb[-edge:].reshape(-1, 3),
                             rgb[:, :edge].reshape(-1, 3), rgb[:, -edge:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    difference = np.abs(rgb - background).max(axis=2) / 255.0
    alpha = np.clip((difference - 0.08) / 0.2, 0.0, 1.0)
    # Fade the edges out. The distilled models run without CFG, so a negative prompt
    # ("border, frame, text") is ignored and they often draw their own frame or app-icon
    # tile around the subject; this keeps only the centre, where the subject is.
    yy, xx = np.mgrid[0:n, 0:n]
    radius = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
    alpha *= np.clip((0.92 - radius) / 0.22, 0.0, 1.0)

    ys, xs = np.nonzero(alpha > 0.3)
    if len(xs):
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * 1.06
        left, top = int(max(0, cx - half)), int(max(0, cy - half))
        right, bottom = int(min(n, cx + half)), int(min(n, cy + half))
        rgb, alpha = rgb[top:bottom, left:right], alpha[top:bottom, left:right]

    # Cut on the original colours, recolour after: the background is found by how far
    # a pixel is from the render's own backdrop, which the ramp would blur.
    rgb = gradient_map(rgb, ramp, blend)
    rgba = np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8)
    glyph = Image.fromarray(rgba, "RGBA").resize((box, box), Image.Resampling.LANCZOS)

    spread = max(3, box // 36) | 1  # MaxFilter wants an odd size
    halo = glyph.getchannel("A").filter(ImageFilter.MaxFilter(spread)).filter(ImageFilter.GaussianBlur(spread / 3))
    shadow = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    shadow.paste(Image.new("RGBA", (box, box), (0, 0, 0, 235)), mask=halo)
    return Image.alpha_composite(shadow, glyph)


def glow(size: int, mask: Image.Image, colour: tuple[int, int, int]) -> Image.Image:
    """The frame's fill: dark, lit from the centre in the ability's colour, so the type
    coding reads behind a full-colour picture without repainting the picture itself."""
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    strength = np.clip(1.0 - radius, 0.0, 1.0) ** 1.6 * 0.55
    fill = np.zeros((size, size, 4), dtype=np.float32)
    for channel in range(3):
        fill[..., channel] = BACKGROUND[channel] * (1 - strength) + colour[channel] * strength
    fill[..., 3] = np.asarray(mask, dtype=np.float32)
    return Image.fromarray(fill.astype(np.uint8), "RGBA")


def _diamond(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, filled: bool, width: int) -> None:
    points = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    draw.polygon([(x + width / 2, y + width / 2) for x, y in points], fill=BADGE_SHADOW)
    draw.polygon(points, fill=BADGE if filled else None, outline=BADGE, width=width)


def draw_badges(canvas: Image.Image, ability: Ability) -> None:
    size = canvas.width
    draw = ImageDraw.Draw(canvas)
    r, w = size * 0.055, max(2, size // 64)
    x0, y0 = size * 0.2, size * 0.2
    ax, ay = size * 0.8, size * 0.8
    # A dark plate behind each badge group, so it reads over the rim and the glyph.
    plate = (8, 8, 12, 215)
    pips = int(ability.actions) if ability.actions in ("1", "2", "3") else (1 if ability.actions != "passive" else 0)
    if pips:
        draw.rounded_rectangle((x0 - r * 1.6, y0 - r * 1.6, x0 + r * (2.1 * (pips - 1) + 1.6), y0 + r * 1.6),
                               radius=r, fill=plate)
    if ability.area:
        draw.ellipse((ax - r * 1.9, ay - r * 1.9, ax + r * 1.9, ay + r * 1.9), fill=plate)
    if ability.actions in ("1", "2", "3"):
        for i in range(int(ability.actions)):
            _diamond(draw, x0 + i * r * 2.1, y0, r, True, w)
    elif ability.actions == "free":
        _diamond(draw, x0, y0, r, False, w)
    elif ability.actions == "reaction":
        box = (x0 - r, y0 - r, x0 + r, y0 + r)
        draw.arc(box, 200, 520, fill=BADGE, width=w * 2)
        draw.polygon([(x0 + r * 0.2, y0 - r * 1.35), (x0 + r * 0.95, y0 - r * 0.95), (x0 + r * 0.25, y0 - r * 0.45)], fill=BADGE)
    elif ability.actions == "long":
        draw.polygon([(x0 - r, y0 - r), (x0 + r, y0 - r), (x0, y0)], fill=BADGE)
        draw.polygon([(x0 - r, y0 + r), (x0 + r, y0 + r), (x0, y0)], fill=BADGE)

    if ability.area == "Burst" or ability.area == "Ring":
        draw.ellipse((ax - r, ay - r, ax + r, ay + r), outline=BADGE, width=w * 2)
        if ability.area == "Burst":
            draw.ellipse((ax - r * 0.35, ay - r * 0.35, ax + r * 0.35, ay + r * 0.35), fill=BADGE)
    elif ability.area == "Cone":
        draw.pieslice((ax - r * 1.6, ay - r * 1.6, ax + r * 1.6, ay + r * 1.6), 200, 250, fill=BADGE)
    elif ability.area == "Line":
        draw.line((ax - r * 1.2, ay + r * 1.2, ax + r * 1.2, ay - r * 1.2), fill=BADGE, width=w * 3)


def compose(ability: Ability, render_png: bytes, export_size: int) -> Image.Image:
    size = export_size * 4
    inset = size * 0.02
    shape = PALETTE["frame"].get(ability.kind, "rounded")
    mask = frame_mask(shape, size, inset)
    scheme = palette(ability)
    colour = tuple(scheme["ramp"][len(scheme["ramp"]) // 2])

    canvas = glow(size, mask, colour)

    glyph_size = int(size * 0.84)
    glyph = glyph_layer(render_png, glyph_size, scheme["ramp"], PALETTE["blend"])
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = (size - glyph_size) // 2
    layer.paste(glyph, (offset, offset))
    clipped = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    clipped.paste(layer, mask=Image.fromarray(np.minimum(np.asarray(layer)[..., 3], np.asarray(mask)), "L"))
    canvas = Image.alpha_composite(canvas, clipped)

    # With the glow behind the picture, the rim's inner band is where the type coding lives.
    canvas = Image.alpha_composite(canvas, rim(shape, size, inset, colour))

    draw_badges(canvas, ability)
    return canvas.resize((export_size, export_size), Image.Resampling.LANCZOS)


def rim(shape: str, size: int, inset: float, colour: tuple[int, int, int]) -> Image.Image:
    """The frame outline in metal, then an inner band in `colour`."""
    rim_width = max(4, size // 40)
    outline = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    edge = np.asarray(frame_mask(shape, size, inset), dtype=np.int16)
    inner = np.asarray(frame_mask(shape, size, inset + rim_width), dtype=np.int16)
    ring = np.clip(edge - inner, 0, 255).astype(np.uint8)
    outline.paste(Image.new("RGBA", (size, size), RIM), mask=Image.fromarray(ring, "L"))
    inner_ring = np.clip(inner - np.asarray(frame_mask(shape, size, inset + rim_width * 2), dtype=np.int16), 0, 255)
    outline.paste(Image.new("RGBA", (size, size), (*colour, 230)), mask=Image.fromarray(inner_ring.astype(np.uint8), "L"))
    return outline


# Tarnished gold, the accent the style prompt asks for.
PORTRAIT_RIM = (160, 137, 79)
# One warm sepia-to-bone grade over every portrait, lightly: it puts each render in the
# icons' tonal key whatever palette the model chose. Past about 0.3 it washes out the
# skin colours that tell ancestries apart.
PORTRAIT_GRADE = [[14, 11, 10], [66, 52, 40], [146, 120, 84], [206, 186, 140], [238, 228, 205]]
PORTRAIT_GRADE_BLEND = 0.25


def compose_portrait(render_png: bytes, export_size: int) -> Image.Image:
    """A portrait: the render in its own colours under a light shared grade, centre-cropped
    square, its edges darkened into the frame, inside a rounded frame with the icons' rim.

    No full gradient map, unlike an icon: a face mapped onto a five-stop ramp goes flat and
    waxy, and a portrait carries no palette coding to show. No crop to the subject
    either: a bust already fills the render, and cropping to its extent would cut the
    shoulders unevenly. The vignette does what the icons' edge fade does, hiding any
    border the model drew, since the negative prompt is inert at these settings.
    """
    size = export_size * 4
    inset = size * 0.02
    mask = frame_mask("rounded", size, inset)
    with Image.open(io.BytesIO(render_png)) as image:
        picture = ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    rgb = gradient_map(np.asarray(picture, dtype=np.float32), PORTRAIT_GRADE, PORTRAIT_GRADE_BLEND)
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    keep = np.clip((1.2 - radius) / 0.45, 0.0, 1.0)[..., None]
    rgb = rgb * keep + np.asarray(BACKGROUND[:3], dtype=np.float32) * (1.0 - keep)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(Image.fromarray(rgb.astype(np.uint8), "RGB"), mask=mask)
    canvas = Image.alpha_composite(canvas, rim("rounded", size, inset, PORTRAIT_RIM))
    return canvas.resize((export_size, export_size), Image.Resampling.LANCZOS)
