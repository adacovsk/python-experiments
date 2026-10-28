"""A review page for every finished icon: labelled, grouped by ability type,
click-to-select, with the `--reroll` command for the selection ready to copy.

Written next to the icons as `gallery.html` after builds. Opened from disk it is a
snapshot of the icons at the time it was written. Served by `serve.py`
(`pixi run icons-gallery`) it is live: it polls `gallery.json`, which the server
reads fresh from disk, so icons appear as a running batch finishes them and a
re-rolled icon refreshes in place. Nothing is uploaded anywhere either way.

Live, each icon can also be approved (moved from the To review tab to Approved) and
have its prompt edited. A review mark is the icon's mtime, kept in
`<set>/reviewed.json`, so a regenerated icon comes back up for review on its own.
"""
from __future__ import annotations

import json
from collections import Counter
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
figure.queued, figure.rendering, figure.edit-queued, figure.editing { border-color:var(--pick); position:relative; }
figure.queued::after, figure.rendering::after, figure.edit-queued::after, figure.editing::after { position:absolute;
  top:10px; right:10px; padding:2px 6px; border-radius:4px; background:var(--pick); color:#111; font-size:11px; font-weight:600; }
figure.queued::after { content:"queued"; }
figure.rendering::after { content:"regenerating…"; }
figure.edit-queued::after { content:"edit queued"; }
figure.editing::after { content:"editing…"; }
figcaption i { color:var(--pick); font-style:normal; }
dialog ol { margin:4px 0; padding-left:20px; color:var(--muted); font-size:12px; }
figure.fresh { box-shadow:0 0 0 2px var(--new); }
figure img { width:100%; aspect-ratio:1; display:block; }
figcaption { font-size:11px; margin-top:4px; overflow-wrap:anywhere; }
figcaption b { display:block; font-size:12px; }
figcaption span { color:var(--muted); }
figure { position:relative; }
.tools { position:absolute; bottom:44px; right:10px; display:none; gap:4px; }
figure:hover .tools { display:flex; }
.tools button { padding:2px 7px; font-size:13px; background:#111d; }
summary button { margin-left:12px; font-size:12px; padding:2px 8px; }
.tabs { display:flex; gap:4px; margin-bottom:8px; }
.tabs button { border-radius:6px 6px 0 0; border-bottom:2px solid transparent; background:transparent; }
.tabs button.on { border-bottom-color:var(--pick); color:var(--pick); background:#222; }
dialog { background:var(--card); color:var(--text); border:1px solid #555; border-radius:10px; width:min(720px, 92vw); }
dialog::backdrop { background:#000a; }
dialog h2 { font-size:15px; margin:0 0 4px; }
dialog img { width:160px; float:right; margin:0 0 8px 12px; border-radius:8px; }
dialog .part { clear:both; margin-top:12px; }
dialog .part b { font-size:12px; color:var(--muted); font-weight:400; }
dialog textarea { width:100%; box-sizing:border-box; min-height:84px; margin:4px 0; padding:6px; background:#111;
  color:var(--text); border:1px solid #444; border-radius:6px; font:13px/1.4 system-ui, sans-serif; }
dialog .row { display:flex; gap:6px; flex-wrap:wrap; }
dialog .note { color:var(--muted); font-size:12px; }
</style></head><body>
<header>
  <h1 id="title">Icon review</h1>
  <div class="tabs"><button id="tab-pending">To review (<span id="npend">0</span>)</button><button id="tab-approved">Approved (<span id="nrev">0</span>)</button></div>
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
<dialog id="edit"><form method="dialog">
  <img id="eimg" alt="">
  <h2 id="ename"></h2>
  <div class="part">
    <b>Improve this picture</b>
    <p class="note">Say what to change ("make the flame blue", "remove the second hand"); the rest of the
      picture is kept. Done by the Cloudflare build, so it waits for the daily allowance if that is spent.
      Every earlier version is kept, so you can step back one edit or revert to the original.
      "Previous picture" brings back the one a repaint replaced (edits and all), and asked again swaps back.</p>
    <ol id="ehistory"></ol>
    <textarea id="eimprove" placeholder="What to change"></textarea>
    <div class="row"><button type="button" id="eimprove-go">Improve</button><button type="button" id="eundo">Undo last edit</button><button type="button" id="erevert">Revert to original</button><button type="button" id="eprevious">Previous picture</button></div>
  </div>
  <div class="part"><b>Prompt — repaints from scratch</b></div>
  <p class="note">Edits the concept line in the concept files. Leave a line empty to fall back to the
    record's own name and description. A concept shared by several records changes all of them;
    the ones not regenerated now pick it up the next time a build reaches them, except pictures that
    were improved or brought back, which keep theirs until regenerated. Regenerating sets the old
    picture aside under glyphs/history/replaced/; "Previous picture" brings it back.</p>
  <div id="eparts"></div>
  <div class="row" style="margin-top:12px"><button value="close">Close</button><span id="estatus" class="note"></span></div>
</form></dialog>
<script>
const SNAPSHOT = __DATA__;
const BACKEND = __BACKEND__, REPO = __REPO__;
const picked = new Set(), seen = new Map();
// Collapsed sections survive live re-renders and reloads; storage can be unavailable
// (private windows), in which case they simply start expanded.
let collapsed = new Set();
try { collapsed = new Set(JSON.parse(localStorage.getItem('icon-gallery-collapsed') || '[]')); } catch {}
const remember = () => { try { localStorage.setItem('icon-gallery-collapsed', JSON.stringify([...collapsed])); } catch {} };
// Two tabs over one list: icons still to look at, and the ones approved.
let tab = 'pending';
try { tab = localStorage.getItem('icon-gallery-tab') === 'approved' ? 'approved' : 'pending'; } catch {}
let current = SNAPSHOT;
const post = (path, body) => fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
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
  if (e.edits.length) {
    const mark = document.createElement('i');
    mark.textContent = ` · ${e.edits.length} edit${e.edits.length > 1 ? 's' : ''}`;
    mark.title = e.edits.join('\\n');
    f.querySelector('span').after(mark);
  }
  if (LIVE) {
    if (e.queue) f.classList.add(e.queue);
    const tools = document.createElement('div');
    tools.className = 'tools';
    tools.innerHTML = `<button></button><button title="Improve this picture, or edit its prompt">✎</button>`;
    const [ok, edit] = tools.querySelectorAll('button');
    ok.textContent = e.reviewed ? '↶' : '✓';
    ok.title = e.reviewed ? 'Send back to review' : 'Looks good: approve';
    ok.addEventListener('click', ev => { ev.stopPropagation(); review([e], !e.reviewed); });
    edit.addEventListener('click', ev => { ev.stopPropagation(); openEditor(e); });
    f.appendChild(tools);
    // Live: a click flags it for regeneration on disk; a running build repaints it and
    // the flag clears itself. Clicking a queued icon un-flags it.
    // A waiting edit counts as flagged, so clicking it cancels the edit.
    f.addEventListener('click', async () => {
      if (f.classList.contains('rendering') || f.classList.contains('editing')) return;
      const queued = !(f.classList.contains('queued') || f.classList.contains('edit-queued'));
      f.classList.remove('edit-queued');
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
  current = entries;
  const reviewedCount = entries.filter(e => e.reviewed).length;
  $('nrev').textContent = reviewedCount;
  $('npend').textContent = entries.length - reviewedCount;
  $('tab-pending').classList.toggle('on', tab === 'pending');
  $('tab-approved').classList.toggle('on', tab === 'approved');
  const groups = new Map(), order = new Map();
  for (const e of entries) {
    if (e.reviewed !== (tab === 'approved')) continue;
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
    const summary = s.querySelector('summary');
    summary.textContent = `${kind} · ${groups.get(kind).length}${queued ? ` · ${queued} flagged` : ''}`;
    if (LIVE && tab === 'pending') {
      // Review a section by flagging the bad ones, then clearing the rest in one go.
      const rest = document.createElement('button');
      rest.textContent = 'Approve the rest';
      rest.title = 'Every icon in this section that is not flagged and matches the filter';
      rest.addEventListener('click', ev => {
        ev.preventDefault(); ev.stopPropagation();
        const q = $('q').value.toLowerCase();
        const keep = groups.get(kind).filter(e => !e.queue && !e.reviewed && `${e.key} ${e.name} ${e.concept}`.toLowerCase().includes(q));
        if (keep.length && confirm(`Approve ${keep.length} icon(s) in "${kind}"?`)) review(keep, true);
      });
      summary.appendChild(rest);
    }
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

// Moves the icons between tabs at once and saves in the background; the next poll
// confirms. Waiting on the save and a full refetch made every click lag.
function review(list, reviewed) {
  const bySet = new Map();
  for (const e of list) {
    e.reviewed = reviewed;
    if (!bySet.has(e.set)) bySet.set(e.set, []);
    bySet.get(e.set).push(e.key);
  }
  const y = scrollY;
  render(current);
  scrollTo(0, y);
  for (const [set, keys] of bySet) post('review', {set, keys, reviewed});
}

function openEditor(e) {
  $('ename').textContent = `${e.name} — ${e.key}`;
  $('eimg').src = `${e.src}?v=${e.mtime}`;
  $('estatus').textContent = '';
  const history = $('ehistory');
  history.replaceChildren(...e.edits.map(text => Object.assign(document.createElement('li'), {textContent: text})));
  $('eimprove').value = '';
  $('eundo').disabled = $('erevert').disabled = !e.edits.length;
  $('eprevious').disabled = !e.previous;
  const ask = async (path, body, label) => {
    $('estatus').textContent = 'saving…';
    const r = await post(path, {set: e.set, key: e.key, ...body});
    $('estatus').textContent = r.ok ? label : `failed: ${await r.text()}`;
    poll();
  };
  $('eimprove-go').onclick = () => {
    if ($('eimprove').value.trim()) ask('improve', {text: $('eimprove').value}, 'edit queued');
  };
  $('eundo').onclick = () => ask('undo', {}, 'undo queued');
  $('erevert').onclick = () => {
    if (confirm(`Revert to the original picture, before all ${e.edits.length} edit(s)?`)) ask('undo', {revert: true}, 'revert queued');
  };
  $('eprevious').onclick = () => ask('undo', {previous: true}, 'previous picture queued');
  const box = $('eparts');
  box.replaceChildren();
  for (const p of e.edit) {
    const d = document.createElement('div');
    d.className = 'part';
    d.innerHTML = `<b></b><textarea></textarea><div class="row"></div>`;
    d.querySelector('b').textContent = p.part
      ? `${p.part} · ${p.key}${p.uses > 1 ? ` — shared by ${p.uses} in this set` : ''}`
      : `concept · ${p.key}${p.uses > 1 ? ` — shared by ${p.uses}` : ''}`;
    const text = d.querySelector('textarea');
    text.value = p.text;
    text.placeholder = p.fallback || '(empty: falls back to the name)';
    const row = d.querySelector('.row');
    const save = async regenerate => {
      $('estatus').textContent = 'saving…';
      const r = await post('concept', {set: e.set, key: e.key, part: p.part, part_key: p.key, text: text.value, regenerate});
      $('estatus').textContent = r.ok ? `saved${(await r.json()).flagged ? ', flagged for regeneration' : ''}` : `failed: ${await r.text()}`;
      poll();
    };
    const buttons = [['Save & regenerate this', 'this']];
    if (p.uses > 1) buttons.push([`Save & regenerate all ${p.uses}`, 'all']);
    buttons.push(['Save only', 'none']);
    for (const [label, mode] of buttons) {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = label;
      b.addEventListener('click', () => save(mode));
      row.appendChild(b);
    }
    box.appendChild(d);
  }
  $('edit').showModal();
}

for (const name of ['pending', 'approved']) {
  $(`tab-${name}`).onclick = () => {
    tab = name;
    try { localStorage.setItem('icon-gallery-tab', tab); } catch {}
    render(current);
  };
}

let last = '';
async function poll() {
  if (!LIVE) return;
  try {
    const entries = await (await fetch('gallery.json', {cache: 'no-store'})).json();
    const sig = JSON.stringify(entries.map(e => [e.key, e.mtime, e.queue, e.reviewed, e.concept, e.edits.length, e.previous]));
    if (sig !== last) { last = sig; render(entries); }
    $('live').textContent = `live · ${new Date().toLocaleTimeString()}`;
  } catch { $('live').textContent = 'live · server not responding'; }
}

render(SNAPSHOT);
if (LIVE) {
  for (const id of ['copy', 'clear', 'n', 'cmd']) $(id).style.display = 'none';
  $('how').textContent = 'Click an icon to flag it for regeneration (click again to un-flag). ' +
    'Hover for ✓ (approve) and ✎ (improve this picture or edit its prompt). A running build repaints flagged icons first; ' +
    'the new one swaps in with a green outline, and a regenerated approved icon moves back to To review.';
  last = JSON.stringify(SNAPSHOT.map(e => [e.key, e.mtime, e.queue, e.reviewed, e.concept, e.edits.length, e.previous]));
  poll(); setInterval(poll, 5000);
} else {
  $('live').textContent = 'snapshot — run `pixi run icons-gallery` for live updates';
}
</script></body></html>
"""


def queue_states(icons_dir: Path) -> dict[str, str]:
    """key -> "queued" while waiting, "rendering" once a build has claimed it, and
    "edit-queued" / "editing" for an improve or undo request (a flag with content);
    unflagged keys are absent. One listing of the folder, not a glob per icon: compiling
    a pattern for each of hundreds of keys was most of a gallery poll's time."""
    queue = icons_dir.parent.parent / "queue"
    states: dict[str, str] = {}
    for flag in queue.iterdir() if queue.is_dir() else ():
        key, claimed, _ = flag.name.partition(".")
        # A plain flag outranks a claim: re-flagged while rendering means "do it again".
        if not claimed or key not in states:
            try:
                edit = flag.stat().st_size > 0
            except FileNotFoundError:
                continue  # claimed or finished between the listing and this look
            states[key] = ("editing" if edit else "rendering") if claimed else ("edit-queued" if edit else "queued")
    return states


def replaced_versions(history: Path) -> dict[str, list[int]]:
    """picture -> the timestamps of its set-aside versions in `history/replaced/`
    (`<picture>@<ns>/`), for "Previous picture"."""
    versions: dict[str, list[int]] = {}
    replaced = history / "replaced"
    for directory in replaced.iterdir() if replaced.is_dir() else ():
        picture, _, stamp = directory.name.rpartition("@")
        if picture and stamp.isdigit():
            versions.setdefault(picture, []).append(int(stamp))
    return versions


def reviewed_path(icons_dir: Path) -> Path:
    return icons_dir.parent.parent / "reviewed.json"


def reviewed_marks(icons_dir: Path) -> dict[str, int]:
    """key -> the mtime of the icon that was reviewed."""
    path = reviewed_path(icons_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def entries(icons_dir: Path, base: Path, abilities: dict[str, Ability], concepts: dict[str, str],
            set_name: str = "", section: int = 0, heading: str | None = None,
            parts: dict[str, dict[str, str]] | None = None) -> list[dict]:
    """Every icon on disk whose key is still a record, as the page renders them.

    `heading` groups the whole set under one title (Conditions); without it each
    record's own type is its group (Spell, Class...). `section` orders the sets.
    `parts` holds a portrait set's per-part concept lines (`{"ancestry": {...}}`);
    an icon set's lines are `concepts`, one per picture."""
    marks = reviewed_marks(icons_dir)
    queued = queue_states(icons_dir)
    manifest_path = icons_dir.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous = replaced_versions(icons_dir.parent.parent / "glyphs" / "history")
    uses = Counter(p for ability in abilities.values() for p in ability.parts)
    shared = Counter(ability.picture for ability in abilities.values() if ability.picture)
    found = []
    # "*.png" alone would also match a half-written "<key>.<pid>.part"-style temp file
    # only if it ended in .png; the stem check keeps any dotted name out regardless.
    for path in sorted(icons_dir.glob("*.png")):
        if "." in path.stem:
            continue
        ability = abilities.get(path.stem)
        if ability is None:
            continue
        mtime = int(path.stat().st_mtime)
        if ability.parts:
            edit = [{"part": part, "key": key, "text": (parts or {}).get(part, {}).get(key, ""),
                     "uses": uses[(part, key)], "fallback": ""} for part, key in ability.parts]
        else:
            picture = ability.picture or path.stem
            edit = [{"part": None, "key": picture, "text": concepts.get(picture, ""),
                     "uses": shared[picture] or 1,
                     "fallback": ability.prompt_subject()}]
        found.append({
            "key": path.stem,
            "name": ability.name,
            "kind": heading or ability.kind,
            "set": set_name,
            "section": section,
            "concept": concepts.get(ability.picture or path.stem) or ability.subject or "",
            "src": path.relative_to(base).as_posix(),
            # Changes when an icon is re-rolled, so the page reloads that one image.
            "mtime": mtime,
            "queue": queued.get(path.stem, ""),
            # A mark on an older picture does not count: the new one has not been seen.
            "reviewed": marks.get(path.stem) == mtime,
            "edit": edit,
            # The improvements applied to the current picture, oldest first.
            "edits": manifest.get(ability.picture or path.stem, {}).get("edits", []),
            # How many pictures a repaint replaced and "Previous picture" can bring back.
            "previous": len(previous.get(ability.picture or path.stem, [])),
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
