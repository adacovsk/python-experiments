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

    @property
    def tradition(self) -> str | None:
        return next((t for t in self.traits if t in TRADITIONS), None)

    @property
    def side(self) -> str | None:
        """Whether the first mechanic with a side harms, heals or helps."""
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
