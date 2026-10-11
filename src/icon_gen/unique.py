"""Find finished icons that look too alike to tell apart.

Two signals, both on the composed icon as a player would see it:
- **shape** — a 256-bit difference hash of the luminance (16x16 gradient signs), so a
  near-identical glyph in the same frame scores a small Hamming distance;
- **colour** — the mean RGB of the opaque pixels, so two similar shapes in clearly
  different tints are not a collision.

A pair collides only when both are close: same-looking shape *and* same-looking colour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Fingerprint:
    bits: np.ndarray  # 256 booleans
    colour: np.ndarray  # mean RGB, float

    def distance(self, other: "Fingerprint") -> tuple[int, float]:
        return int(np.count_nonzero(self.bits != other.bits)), float(np.linalg.norm(self.colour - other.colour))


def fingerprint(path: Path) -> Fingerprint:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        grey = np.asarray(rgba.convert("L").resize((17, 16), Image.Resampling.LANCZOS), dtype=np.int16)
        pixels = np.asarray(rgba, dtype=np.float32)
    bits = (grey[:, 1:] > grey[:, :-1]).flatten()
    opaque = pixels[..., 3] > 128
    colour = pixels[opaque][:, :3].mean(axis=0) if opaque.any() else np.zeros(3)
    return Fingerprint(bits, colour)


def collisions(
    prints: dict[str, Fingerprint], max_bits: int, max_colour: float
) -> list[tuple[str, str, int, float]]:
    """Every pair whose shape distance is at most `max_bits` and colour distance at most
    `max_colour`, as (earlier key, later key, bits, colour) in key order."""
    keys = sorted(prints)
    found = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            bits, colour = prints[a].distance(prints[b])
            if bits <= max_bits and colour <= max_colour:
                found.append((a, b, bits, colour))
    return found
