"""Where the game checkout and the working set of renders live.

Both are set in pixi.toml's `[activation.env]`, so no machine's layout is written
into the code. The working set (raw pictures, queue, review marks) is kept out of the
game checkout: only approved icons reach the game, through `build.py --export`.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ["BEVY_RPG"])
OUTPUT = Path(os.environ["ICON_GEN_OUT"])
