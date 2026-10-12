"""Read every ability record from a bevy-rpg checkout, reduced to what an icon needs.

The game's ability data lives in `assets/data/en/**/*.json` files whose top level is
`{"abilities": {key: record}}`. Each record's mechanics are `{"Kind": {fields}}`
objects (or a bare string for a field-less kind). Nothing here knows any specific
ability: every choice is read off the record's own fields.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

TRADITIONS = ("arcane", "divine", "occult", "primal")

# Mechanic kinds whose area lands on the caster's side, by the same reading the game
# makes (`AbilityMechanic::area_side`): the kind itself states harm or help.
HARMFUL = {"Damage", "Debuff", "DamageOverTime", "TerrainHazard", "MultipleAttack", "AuraCondition"}
HEALING = {"Healing", "TemporaryHitPoints"}
BENEFICIAL = {"Buff", "StatBonus", "SkillBonus", "SavingThrowBonus", "ConditionalBonus", "DamageResistance", "AuraEffect"}


@dataclass(frozen=True)
class Ability:
    key: str
    name: str
    description: str
    kind: str  # the record's `type`: "Spell", "Class", "Skill", ...
    traits: tuple[str, ...]
    mechanics: tuple[str, ...]  # mechanic kinds, in authored order
    damage_type: str | None  # the first damage type any mechanic deals
    actions: str  # "1", "2", "3", "reaction", "free", "passive", "long" (an activity or a cast time)
    area: str | None  # "Burst", "Cone", "Line", "Ring"
    # A side stated outright, for records with no mechanics to read it from (conditions).
    stated_side: str | None = None
    # Records that share one picture (an effect's buff and debuff) name it here; the
    # palette, applied when composing, is what tells them apart.
    picture: str | None = None

    @property
    def tradition(self) -> str | None:
        return next((t for t in self.traits if t in TRADITIONS), None)

    @property
    def side(self) -> str | None:
        """Whether the first mechanic with a side harms, heals or helps."""
        if self.stated_side:
            return self.stated_side
        for kind in self.mechanics:
            if kind in HARMFUL:
                return "harmful"
            if kind in HEALING:
                return "healing"
            if kind in BENEFICIAL:
                return "beneficial"
        return None

    def prompt_subject(self) -> str:
        """The ability's own name and the first clause of its description.

        Numbers are dropped: "20-foot radius" and "+1 circumstance bonus" are rules,
        not imagery, and an image model renders digits as stray glyphs.
        """
        first = re.split(r"(?<=[.;:])\s", self.description.strip(), maxsplit=1)[0]
        # The clause before the first comma is the image; what follows is usually the
        # rules ("dealing 6d6 fire damage with a basic Reflex save").
        first = first.split(",", 1)[0]
        first = re.sub(r"\b\d+d\d+\b", "", first)  # dice
        first = re.sub(r"\b(?:within|up to|of)?\s*\d+[- ](?:foot|feet|ft)\b", "", first)  # distances
        first = re.sub(r"[+\-]?\d+", "", first)
        first = re.sub(r"\s{2,}", " ", first).strip(" .,;:")
        return f"{self.name}: {first[:140]}" if first else self.name


def _activation(value: object) -> str:
    if isinstance(value, dict):
        kind, amount = next(iter(value.items()))
        return str(amount) if kind == "Action" else "long"
    return {"Passive": "passive", "Reaction": "reaction", "FreeAction": "free"}.get(str(value), "long")


def _mechanic(entry: object) -> tuple[str, dict]:
    if isinstance(entry, dict):
        kind, body = next(iter(entry.items()))
        return kind, body if isinstance(body, dict) else {}
    return str(entry), {}


def load(repo: Path) -> dict[str, Ability]:
    """Every ability record under `repo/assets/data/en`, keyed by its data key."""
    abilities: dict[str, Ability] = {}
    for path in sorted((repo / "assets" / "data" / "en").rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        records = data.get("abilities") if isinstance(data, dict) else None
        if not isinstance(records, dict):
            continue
        for key, record in records.items():
            mechanics = [_mechanic(entry) for entry in record.get("mechanics", [])]
            damage = next((body["damage_type"] for _, body in mechanics if isinstance(body.get("damage_type"), str)), None)
            area = next((next(iter(body["area"])) for _, body in mechanics if isinstance(body.get("area"), dict)), None)
            abilities[key] = Ability(
                key=key,
                name=record.get("name", key),
                description=record.get("description", ""),
                kind=record.get("type", "General"),
                traits=tuple(record.get("traits", [])),
                mechanics=tuple(kind for kind, _ in mechanics),
                damage_type=damage,
                actions=_activation(record.get("activation", "Passive")),
                area=area,
            )
    return abilities


def load_conditions(repo: Path, sides: dict[str, list[str]]) -> dict[str, Ability]:
    """Every condition in `repo/assets/data/en/conditions.json`, as records the same
    pipeline renders. A condition has no mechanics to read harm or help from, so its
    side comes from `sides` (palette.toml's `[condition_side]`); anything it does not
    list is harmful, which is most of them."""
    data = json.loads((repo / "assets" / "data" / "en" / "conditions.json").read_text())
    side_of = {key: side for side, keys in sides.items() for key in keys}
    return {
        key: Ability(
            key=key, name=entry.get("name", key), description="", kind="Condition",
            traits=(), mechanics=(), damage_type=None, actions="passive", area=None,
            stated_side=side_of.get(key, "harmful"),
        )
        for key, entry in data["indicators"].items()
    }


def load_effects(repo: Path, conditions: set[str]) -> dict[str, Ability]:
    """Every stat or state a `Buff` or `Debuff` mechanic applies that is not itself a
    condition (those have their own set): `armor_class`, `speed`, `fast_healing`...

    One record per direction the data actually uses, `<key>_buff` and `<key>_debuff`,
    each sided by its mechanic kind, so +AC and -AC share a picture but not a palette.
    The description lists the abilities that apply it, as context for its concept.
    """
    used: dict[tuple[str, str], list[str]] = {}
    for path in sorted((repo / "assets" / "data" / "en").rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        records = data.get("abilities") if isinstance(data, dict) else None
        if not isinstance(records, dict):
            continue
        for record in records.values():
            for kind, body in (_mechanic(entry) for entry in record.get("mechanics", [])):
                if kind not in ("Buff", "Debuff") or not isinstance(body.get("effect_type"), str):
                    continue
                base, _, suffix = body["effect_type"].rpartition("_")
                key = base if suffix.isdigit() and base else body["effect_type"]
                if key in conditions or key.replace("_", "-") in conditions:
                    continue
                used.setdefault((key, kind), []).append(record.get("name", ""))
    return {
        f"{key}_{kind.lower()}": Ability(
            key=f"{key}_{kind.lower()}",
            name=f"{key.replace('_', ' ').capitalize()} {'up' if kind == 'Buff' else 'down'}",
            description="Applied by: " + ", ".join(sorted(set(names))[:6]),
            kind="Effect", traits=(), mechanics=(kind,), damage_type=None,
            actions="passive", area=None, picture=key,
        )
        for (key, kind), names in sorted(used.items())
    }
