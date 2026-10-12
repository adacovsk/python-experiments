"""A review page for every finished icon: labelled, grouped by ability type,
click-to-select, with the `--reroll` command for the selection ready to copy.

Written next to the icons as `gallery.html` after builds. Opened from disk it is a
snapshot of the icons at the time it was written. Served by `serve.py`
(`pixi run icons-gallery`) it is live: it polls `gallery.json`, which the server
reads fresh from disk, so icons appear as a running batch finishes them and a
re-rolled icon refreshes in place. Nothing is uploaded anywhere either way.
"""
from __future__ import annotations

import json
from pathlib import Path

from abilities import Ability

REPO = Path(__file__).resolve().parents[2]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Icon review</title>
<style>
:root { --bg:#16161b; --card:#202028; --text:#e8e4dc; --muted:#9a968e; --pick:#e0a040; --new:#5fbf7f; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:2; background:#101014ee; padding:12px 16px; border-bottom:1px solid #333; }
header h1 { font-size:16px; margin:0 0 8px; }
.bar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
input[type=search] { flex:1; min-width:180px; padding:6px 8px; background:#222; color:var(--text); border:1px solid #444; border-radius:6px; }
button { padding:6px 10px; background:#333; color:var(--text); border:1px solid #555; border-radius:6px; cursor:pointer; }
#cmd { width:100%; box-sizing:border-box; margin-top:8px; padding:6px 8px; background:#0c0c10; color:var(--pick);
       border:1px solid #444; border-radius:6px; font:12px ui-monospace, monospace; }
#live { font-size:12px; color:var(--muted); }
details { padding:4px 16px; }
summary { font-size:14px; color:var(--muted); margin:12px 0 8px; cursor:pointer; user-select:none; }
summary:hover { color:var(--text); }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(128px, 1fr)); gap:10px; }
figure { margin:0; background:var(--card); border:2px solid transparent; border-radius:8px; padding:6px; cursor:pointer; }
figure.picked { border-color:var(--pick); }
figure.queued, figure.rendering { border-color:var(--pick); position:relative; }
figure.queued::after, figure.rendering::after { position:absolute; top:10px; right:10px; padding:2px 6px; border-radius:4px;
  background:var(--pick); color:#111; font-size:11px; font-weight:600; }
figure.queued::after { content:"queued"; }
figure.rendering::after { content:"regenerating…"; }
figure.fresh { box-shadow:0 0 0 2px var(--new); }
figure img { width:100%; aspect-ratio:1; display:block; }
figcaption { font-size:11px; margin-top:4px; overflow-wrap:anywhere; }
figcaption b { display:block; font-size:12px; }
figcaption span { color:var(--muted); }
</style></head><body>
<header>
  <h1 id="title">Icon review</h1>
  <p id="how" style="margin:0 0 8px;color:var(--muted);font-size:12px"></p>
  <div class="bar">
    <input type="search" id="q" placeholder="Filter by key, name or concept">
    <button id="copy">Copy reroll command</button>
    <button id="clear">Clear selection</button>
    <button id="fold">Collapse all</button>
    <button id="unfold">Expand all</button>
    <span id="n">0 selected</span>
    <span id="live"></span>
  </div>
  <input id="cmd" readonly value="">
</header>
<main id="main"></main>
<script>
const SNAPSHOT = __DATA__;
const BACKEND = __BACKEND__, REPO = __REPO__;
const picked = new Set(), seen = new Map();
// Collapsed sections survive live re-renders and reloads; storage can be unavailable
// (private windows), in which case they simply start expanded.
let collapsed = new Set();
try { collapsed = new Set(JSON.parse(localStorage.getItem('icon-gallery-collapsed') || '[]')); } catch {}
const remember = () => { try { localStorage.setItem('icon-gallery-collapsed', JSON.stringify([...collapsed])); } catch {} };
const LIVE = location.protocol.startsWith('http');
let first = true;
const $ = id => document.getElementById(id);

function card(e) {
  const f = document.createElement('figure');
  f.dataset.key = e.key;
  f.dataset.text = `${e.key} ${e.name} ${e.concept}`.toLowerCase();
  f.title = e.concept;
  f.innerHTML = `<img loading="lazy" alt=""><figcaption><b></b><span></span></figcaption>`;
  f.querySelector('img').src = `${e.src}?v=${e.mtime}`;
  f.querySelector('b').textContent = e.name;
  f.querySelector('span').textContent = e.key;
  if (LIVE) {
    if (e.queue) f.classList.add(e.queue);
    // Live: a click flags it for regeneration on disk; a running build repaints it and
    // the flag clears itself. Clicking a queued icon un-flags it.
    f.addEventListener('click', async () => {
      if (f.classList.contains('rendering')) return;
      const queued = !f.classList.contains('queued');
      f.classList.toggle('queued', queued);
      await fetch('queue', {method: 'POST', headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({set: e.set, key: e.key, queued})});
    });
  } else {
    if (picked.has(e.key)) f.classList.add('picked');
    f.addEventListener('click', () => {
      picked.has(e.key) ? picked.delete(e.key) : picked.add(e.key);
      f.classList.toggle('picked'); refresh();
    });
  }
  return f;
}

function render(entries) {
  const groups = new Map(), order = new Map();
  for (const e of entries) {
    if (!groups.has(e.kind)) { groups.set(e.kind, []); order.set(e.kind, e.section); }
    groups.get(e.kind).push(e);
  }
  const main = $('main');
  main.replaceChildren();
  // Abilities by type first, then the other sets, each as its own section.
  const kinds = [...groups.keys()].sort((a, b) => (order.get(a) - order.get(b)) || a.localeCompare(b));
  for (const kind of kinds) {
    const s = document.createElement('details');
    s.open = !collapsed.has(kind);
    s.innerHTML = `<summary></summary><div class="grid"></div>`;
    const queued = groups.get(kind).filter(e => e.queue).length;
    s.querySelector('summary').textContent =
      `${kind} · ${groups.get(kind).length}${queued ? ` · ${queued} flagged` : ''}`;
    s.addEventListener('toggle', () => { s.open ? collapsed.delete(kind) : collapsed.add(kind); remember(); });
    const grid = s.querySelector('.grid');
    for (const e of groups.get(kind)) {
      const f = card(e);
      // Highlight what arrived or changed since the page opened.
      if (!first && seen.get(e.key) !== e.mtime) f.classList.add('fresh');
      grid.appendChild(f);
    }
    main.appendChild(s);
  }
  for (const e of entries) seen.set(e.key, e.mtime);
  first = false;
  $('title').textContent = `Icon review — ${entries.length} icons · click the bad ones`;
  filter();
}

function refresh() {
  const keys = [...picked].sort();
  $('cmd').value = keys.length ? `cd ${REPO} && pixi run icons ${BACKEND}--reroll ${keys.join(',')}` : '';
  $('n').textContent = `${keys.length} selected`;
}
function filter() {
  const q = $('q').value.toLowerCase();
  document.querySelectorAll('figure').forEach(f => f.style.display = f.dataset.text.includes(q) ? '' : 'none');
}
$('q').oninput = filter;
$('fold').onclick = () => document.querySelectorAll('details').forEach(d => d.open = false);
$('unfold').onclick = () => document.querySelectorAll('details').forEach(d => d.open = true);
$('copy').onclick = () => { $('cmd').select(); navigator.clipboard?.writeText($('cmd').value); };
$('clear').onclick = () => { picked.clear(); document.querySelectorAll('figure.picked').forEach(f => f.classList.remove('picked')); refresh(); };

render(SNAPSHOT);
if (LIVE) {
  for (const id of ['copy', 'clear', 'n', 'cmd']) $(id).style.display = 'none';
  $('how').textContent = 'Click an icon to flag it for regeneration (click again to un-flag). ' +
    'A running build repaints flagged icons first; the new one swaps in with a green outline.';
  let last = JSON.stringify(SNAPSHOT.map(e => [e.key, e.mtime, e.queue]));
  const poll = async () => {
    try {
      const entries = await (await fetch('gallery.json', {cache: 'no-store'})).json();
      const sig = JSON.stringify(entries.map(e => [e.key, e.mtime, e.queue]));
      if (sig !== last) { last = sig; render(entries); }
      $('live').textContent = `live · ${new Date().toLocaleTimeString()}`;
    } catch { $('live').textContent = 'live · server not responding'; }
  };
  poll(); setInterval(poll, 5000);
} else {
  $('live').textContent = 'snapshot — run `pixi run icons-gallery` for live updates';
}
</script></body></html>
"""


def queue_state(icons_dir: Path, key: str) -> str:
    """"" if not flagged, "queued" if waiting, "rendering" once a build has claimed it."""
    queue = icons_dir.parent.parent / "queue"
    if (queue / key).exists():
        return "queued"
    return "rendering" if any(queue.glob(f"{key}.*")) else ""


def entries(icons_dir: Path, base: Path, abilities: dict[str, Ability], concepts: dict[str, str],
            set_name: str = "", section: int = 0, heading: str | None = None) -> list[dict]:
    """Every icon on disk whose key is still a record, as the page renders them.

    `heading` groups the whole set under one title (Conditions); without it each
    record's own type is its group (Spell, Class...). `section` orders the sets."""
    found = []
    # "*.png" alone would also match a half-written "<key>.<pid>.part"-style temp file
    # only if it ended in .png; the stem check keeps any dotted name out regardless.
    for path in sorted(icons_dir.glob("*.png")):
        if "." in path.stem:
            continue
        ability = abilities.get(path.stem)
        if ability is None:
            continue
        found.append({
            "key": path.stem,
            "name": ability.name,
            "kind": heading or ability.kind,
            "set": set_name,
            "section": section,
            "concept": concepts.get(ability.picture or path.stem, ""),
            "src": path.relative_to(base).as_posix(),
            # Changes when an icon is re-rolled, so the page reloads that one image.
            "mtime": int(path.stat().st_mtime),
            "queue": queue_state(icons_dir, path.stem),
        })
    return found


def write(dest: Path, icons_dir: Path, abilities: dict[str, Ability], concepts: dict[str, str], backend: str) -> Path:
    """A static snapshot of one set (written by builds next to its icons)."""
    return write_page(dest, entries(icons_dir, dest.parent, abilities, concepts, dest.parent.name), backend)


def write_page(dest: Path, data: list[dict], backend: str) -> Path:
    flag = "--backend cloudflare " if backend == "cloudflare" else ""
    page = (PAGE
            # "</" cannot appear inside the inline script, or a concept could end it early.
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
            .replace("__BACKEND__", json.dumps(flag))
            # The command is pasted into whatever terminal is open, so it carries its own cd.
            .replace("__REPO__", json.dumps(str(REPO))))
    dest.write_text(page)
    return dest
