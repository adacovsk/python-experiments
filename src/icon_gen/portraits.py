"""Player, NPC and monster portrait records from a bevy-rpg checkout, as records the
icon pipeline renders.

- **players** — every ancestry × class × sex, keyed `<ancestry>_<class>_<sex>`
  (`dwarf_cleric_female`): what character creation offers.
- **npcs** — every NPC role × ancestry, keyed `<role>_<ancestry>` (`guard_captain_elf`).
  The game composes an NPC from a role in `npcs.json` and the ancestry it spawns with,
  so this is every NPC it can produce.
- **monsters** — every monster record, keyed by its own key, the file its `portrait`
  field names.

A portrait's subject is assembled from per-part concepts in concepts_portraits.json
(`ancestry`, `class`, `role`) rather than written per record: 952 records come from 66
lines, and a new ancestry needs one line, not 60. A part with no concept falls back to
its record's own name. `ancestry_sex` (`dwarf_female`) adds a clause for one sex where
the ancestry line alone pulls the model wrong.

`ancestry_form` holds alternative bodies for an ancestry with more than one
(`leshy_fungus`, `leshy_gourd`): each portrait takes one, picked per key. One line
reading "leaves, flowers or fungus" let the model take the cheapest option every time
— a human with a wreath — so each form is its own concrete line instead.

`ancestry_dress` is how that people makes and wears things — materials, motifs, fit —
and it restyles the class or role's kit. Without it the class line is the only thing
describing clothes and gear, so every ancestry's alchemist wore the identical apron and
goggles and only the face changed.

Players and NPCs also take one age, head angle and light from the `variety` lists,
picked per key: with one fixed framing every bust came out the same frontal,
evenly-lit face, and the ancestry was the only thing left to tell them apart. Monsters share no parts, so each has its own line in
concepts_monsters.json; their records carry no description, so one without a concept
falls back to its name, size and traits.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

from abilities import Ability

HERE = Path(__file__).resolve().parent
CONCEPT_FILE = HERE / "concepts_portraits.json"
MONSTER_CONCEPT_FILE = HERE / "concepts_monsters.json"
SEXES = ("male", "female")


# Parts concepts_portraits.json holds; the gallery may edit any of them.
PORTRAIT_PARTS = ("ancestry", "ancestry_form", "ancestry_sex", "ancestry_dress", "class", "role")


def _dress(parts: dict[str, dict[str, str]], a_key: str) -> str:
    """The clause that restyles a class or role's kit in this ancestry's own make."""
    line = parts["ancestry_dress"].get(a_key)
    return f"their clothing and gear made in their own people's style, {line}" if line else ""


def _form(parts: dict[str, dict[str, str]], key: str, a_key: str) -> str | None:
    """The `ancestry_form` key this portrait takes, if its ancestry has any: stable per key."""
    forms = sorted(k for k in parts["ancestry_form"] if k.startswith(f"{a_key}_"))
    return forms[zlib.crc32(f"{key}:form".encode()) % len(forms)] if forms else None


def _file() -> dict:
    return json.loads(CONCEPT_FILE.read_text()) if CONCEPT_FILE.exists() else {}


def concepts() -> dict[str, dict[str, str]]:
    parts = _file()
    return {part: parts.get(part, {}) for part in PORTRAIT_PARTS}


class Variety:
    """One age, head angle and light per portrait, from the `variety` lists: stable per
    key, and picked independently per list so they do not move in lockstep."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = _file().get("variety", {})

    def pick(self, key: str, name: str) -> str:
        options = self.lists.get(name) or [""]
        return options[zlib.crc32(f"{key}:{name}".encode()) % len(options)]

    def who(self, key: str, *words: str) -> str:
        """"an old male dwarf" / "a female elf": the article agrees with the first word."""
        phrase = " ".join(w for w in (self.pick(key, "age"), *words) if w)
        return f"{'an' if phrase[0] in 'aeiou' else 'a'} {phrase}"

    def framing(self, key: str) -> str:
        return ", ".join(x for x in (self.pick(key, "angle"), self.pick(key, "light")) if x)


def _merged(repo: Path, folder: str) -> dict[str, dict]:
    """Every record under `assets/data/en/<folder>/*/*.json`, whose top level is `{folder: {key: record}}`."""
    merged: dict[str, dict] = {}
    for path in sorted((repo / "assets" / "data" / "en" / folder).glob("*/*.json")):
        merged.update(json.loads(path.read_text())[folder])
    return merged


def _roles(repo: Path) -> dict[str, dict]:
    return json.loads((repo / "assets" / "data" / "en" / "npcs.json").read_text())


def _look(part: dict[str, str], key: str, fallback: str) -> str:
    return part.get(key) or fallback


def load_players(repo: Path) -> dict[str, Ability]:
    parts, vary = concepts(), Variety()
    ancestries, classes = _merged(repo, "ancestries"), _merged(repo, "classes")
    records = {}
    for a_key, ancestry in ancestries.items():
        for c_key, cls in classes.items():
            for sex in SEXES:
                key = f"{a_key}_{c_key}_{sex}"
                extra = parts["ancestry_sex"].get(f"{a_key}_{sex}")
                form = _form(parts, key, a_key)
                records[key] = Ability(
                    key=key, name=f"{ancestry['name']} {cls['name']} ({sex})", description="",
                    kind=f"Players · {ancestry['name']}", traits=(), mechanics=(), damage_type=None,
                    actions="passive", area=None,
                    subject="; ".join(x for x in (
                        # "a female leshy", not "a leshy woman": a noun outweighs the
                        # ancestry, so "woman" painted every leshy as a human with a
                        # wreath. Where the adjective loses to the ancestry's own pull
                        # (beards on dwarf women), `ancestry_sex` says it outright.
                        f"{vary.who(key, sex, ancestry['name'].lower(), cls['name'].lower())}: "
                        f"{_look(parts['ancestry'], a_key, ancestry['name'].lower())}",
                        parts["ancestry_form"].get(form, ""),
                        extra,
                        # Dress before the kit: the earlier clause wins, and the kit's
                        # "long robes" overrode a leshy's "barely any clothing".
                        _dress(parts, a_key),
                        _look(parts["class"], c_key, cls["name"].lower()),
                        vary.framing(key)) if x),
                    parts=(("ancestry", a_key), *((("ancestry_form", form),) if form else ()),
                           ("ancestry_sex", f"{a_key}_{sex}"), ("class", c_key), ("ancestry_dress", a_key)),
                )
    return records


def load_npcs(repo: Path) -> dict[str, Ability]:
    parts, vary = concepts(), Variety()
    ancestries, roles = _merged(repo, "ancestries"), _roles(repo)
    records = {}
    for r_key, role in roles.items():
        for a_key, ancestry in ancestries.items():
            key = f"{r_key}_{a_key}"
            form = _form(parts, key, a_key)
            records[key] = Ability(
                key=key, name=f"{ancestry['name']} {role['name']}", description="",
                kind=f"NPCs · {role['name']}", traits=(), mechanics=(), damage_type=None,
                actions="passive", area=None,
                subject="; ".join(x for x in (
                    f"{vary.who(key, ancestry['name'].lower())}: "
                    f"{_look(parts['ancestry'], a_key, ancestry['name'].lower())}",
                    parts["ancestry_form"].get(form, ""),
                    _dress(parts, a_key),
                    _look(parts["role"], r_key, role["name"].lower()),
                    vary.framing(key)) if x),
                parts=(("ancestry", a_key), *((("ancestry_form", form),) if form else ()),
                       ("role", r_key), ("ancestry_dress", a_key)),
            )
    return records


def _monsters(repo: Path) -> dict[str, tuple[str, dict]]:
    """Every monster record, with the environment its file is named for (`plains`)."""
    found = {}
    for path in sorted((repo / "assets" / "data" / "en" / "monsters").rglob("*.monsters.json")):
        environment = path.name.split(".", 1)[0]
        for key, record in json.loads(path.read_text()).items():
            found[key] = (environment, record)
    return found


def monster_concepts() -> dict[str, str]:
    return json.loads(MONSTER_CONCEPT_FILE.read_text()) if MONSTER_CONCEPT_FILE.exists() else {}


def load_monsters(repo: Path) -> dict[str, Ability]:
    written = monster_concepts()
    records = {}
    for key, (environment, record) in _monsters(repo).items():
        traits = ", ".join(t for t in record.get("traits", []) if t != record.get("size"))
        fallback = f"a {record.get('size', '')} {record['name'].lower()}, {traits}".replace("  ", " ")
        records[key] = Ability(
            key=key, name=record["name"], description="", kind=f"Monsters · {environment.capitalize()}",
            traits=(), mechanics=(), damage_type=None, actions="passive", area=None,
            subject=written.get(key) or fallback,
            parts=(("monster", key),),
        )
    return records


# The concept parts each portrait set is assembled from.
PARTS = {"players": ("ancestry", "class"), "npcs": ("ancestry", "role"), "monsters": ("monster",)}


def missing_parts(repo: Path, records: str) -> list[str]:
    """`part/key` for every part of this set with no concept yet, and stale ones."""
    parts = {**concepts(), "monster": monster_concepts()}
    sources = {"ancestry": lambda: _merged(repo, "ancestries"), "class": lambda: _merged(repo, "classes"),
               "role": lambda: _roles(repo), "monster": lambda: _monsters(repo)}
    wanted = {part: set(sources[part]()) for part in PARTS[records]}
    missing = [f"{part}/{key}" for part, keys in wanted.items() for key in sorted(keys - parts[part].keys())]
    stale = [f"{part}/{key} (no longer in the data)" for part, keys in wanted.items()
             for key in sorted(parts[part].keys() - keys)]
    return missing + stale
