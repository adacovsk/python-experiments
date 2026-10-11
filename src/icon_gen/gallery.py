"""A local review page for every finished icon: labelled, grouped by ability type,
click-to-select, with the `--reroll` command for the selection ready to copy.

Written next to the icons as `gallery.html` after every build, and opened straight
from disk (file://), so nothing is uploaded anywhere.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from abilities import Ability

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Icon review</title>
<style>
:root {{ --bg:#16161b; --card:#202028; --text:#e8e4dc; --muted:#9a968e; --pick:#e0a040; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }}
header {{ position:sticky; top:0; z-index:2; background:#101014ee; padding:12px 16px; border-bottom:1px solid #333; }}
header h1 {{ font-size:16px; margin:0 0 8px; }}
.bar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
input[type=search] {{ flex:1; min-width:180px; padding:6px 8px; background:#222; color:var(--text); border:1px solid #444; border-radius:6px; }}
button {{ padding:6px 10px; background:#333; color:var(--text); border:1px solid #555; border-radius:6px; cursor:pointer; }}
#cmd {{ width:100%; box-sizing:border-box; margin-top:8px; padding:6px 8px; background:#0c0c10; color:var(--pick);
        border:1px solid #444; border-radius:6px; font:12px ui-monospace, monospace; }}
section {{ padding:8px 16px; }}
section h2 {{ font-size:14px; color:var(--muted); margin:16px 0 8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(128px, 1fr)); gap:10px; }}
figure {{ margin:0; background:var(--card); border:2px solid transparent; border-radius:8px; padding:6px; cursor:pointer; }}
figure.picked {{ border-color:var(--pick); }}
figure img {{ width:100%; aspect-ratio:1; display:block; }}
figcaption {{ font-size:11px; margin-top:4px; overflow-wrap:anywhere; }}
figcaption b {{ display:block; font-size:12px; }}
figcaption span {{ color:var(--muted); }}
</style></head><body>
<header>
  <h1>Icon review — {count} icons · click the bad ones</h1>
  <div class="bar">
    <input type="search" id="q" placeholder="Filter by key, name or concept">
    <button id="copy">Copy reroll command</button>
    <button id="clear">Clear selection</button>
    <span id="n">0 selected</span>
  </div>
  <input id="cmd" readonly value="">
</header>
{sections}
<script>
const BACKEND = {backend};
const picked = new Set();
const cmd = document.getElementById('cmd'), n = document.getElementById('n');
function refresh() {{
  const keys = [...picked].sort();
  cmd.value = keys.length ? `pixi run icons ${{BACKEND}}--reroll ${{keys.join(',')}}` : '';
  n.textContent = `${{keys.length}} selected`;
}}
document.querySelectorAll('figure').forEach(f => f.addEventListener('click', () => {{
  const k = f.dataset.key;
  picked.has(k) ? picked.delete(k) : picked.add(k);
  f.classList.toggle('picked');
  refresh();
}}));
document.getElementById('copy').onclick = () => {{ cmd.select(); navigator.clipboard?.writeText(cmd.value); }};
document.getElementById('clear').onclick = () => {{
  picked.clear(); document.querySelectorAll('figure.picked').forEach(f => f.classList.remove('picked')); refresh();
}};
document.getElementById('q').oninput = e => {{
  const q = e.target.value.toLowerCase();
  document.querySelectorAll('figure').forEach(f => f.style.display = f.dataset.text.includes(q) ? '' : 'none');
}};
</script></body></html>
"""


def write(dest: Path, icons_dir: Path, abilities: dict[str, Ability], concepts: dict[str, str], backend: str) -> Path:
    """Every icon on disk whose key is still an ability, grouped by ability type."""
    by_kind: dict[str, list[str]] = {}
    for path in sorted(icons_dir.glob("*.png")):
        if path.stem in abilities:
            by_kind.setdefault(abilities[path.stem].kind, []).append(path.stem)
    sections = []
    for kind in sorted(by_kind):
        cards = []
        for key in by_kind[kind]:
            ability, concept = abilities[key], concepts.get(key, "")
            text = html.escape(f"{key} {ability.name} {concept}".lower(), quote=True)
            src = (icons_dir / f"{key}.png").relative_to(dest.parent).as_posix()
            cards.append(
                f'<figure data-key="{html.escape(key, quote=True)}" data-text="{text}" title="{html.escape(concept, quote=True)}">'
                f'<img loading="lazy" src="{html.escape(src, quote=True)}" alt="">'
                f'<figcaption><b>{html.escape(ability.name)}</b><span>{html.escape(key)}</span></figcaption></figure>'
            )
        sections.append(f'<section><h2>{html.escape(kind)} · {len(cards)}</h2><div class="grid">{"".join(cards)}</div></section>')
    flag = "--backend cloudflare " if backend == "cloudflare" else ""
    dest.write_text(PAGE.format(
        count=sum(len(keys) for keys in by_kind.values()),
        sections="\n".join(sections),
        backend=json.dumps(flag),
    ))
    return dest
