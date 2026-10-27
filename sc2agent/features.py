"""Turns the game state into a fixed-size observation vector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from sc2agent.actions import army_units

if TYPE_CHECKING:
    from sc2agent.bot import RLBot

# Each feature is (name, normaliser). Values are divided by the normaliser and
# clipped to [0, OBS_CLIP] so the policy sees roughly unit-scale inputs.
FEATURES: tuple[tuple[str, float], ...] = (
    ("minerals", 1000),
    ("vespene", 500),
    ("supply_used", 200),
    ("supply_cap", 200),
    ("supply_left", 20),
    ("workers", 70),
    ("workers_pending", 5),
    ("marines", 100),
    ("marines_pending", 10),
    ("townhalls", 4),
    ("townhalls_pending", 1),
    ("supply_depots", 20),
    ("supply_depots_pending", 3),
    ("barracks", 10),
    ("barracks_pending", 3),
    ("barracks_idle", 10),
    ("refineries", 8),
    ("engineering_bay", 1),
    ("infantry_weapons_1", 1),
    ("attacking", 1),
    ("enemy_units_visible", 50),
    ("enemy_structures_known", 30),
    ("enemy_near_base", 20),
    ("game_minutes", 30),
    ("enemy_terran", 1),
    ("enemy_zerg", 1),
    ("enemy_protoss", 1),
)

FEATURE_INDEX = {name: i for i, (name, _) in enumerate(FEATURES)}
OBS_SIZE = len(FEATURES)
OBS_CLIP = 5.0


def raw_features(bot: RLBot) -> dict[str, float]:
    depots = bot.structures({UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED})
    rax = bot.structures(UnitTypeId.BARRACKS)
    near_base = 0
    if bot.townhalls:
        near_base = sum(1 for u in bot.enemy_units if bot.townhalls.closest_distance_to(u) < 20)
    return {
        "minerals": bot.minerals,
        "vespene": bot.vespene,
        "supply_used": bot.supply_used,
        "supply_cap": bot.supply_cap,
        "supply_left": bot.supply_left,
        "workers": bot.workers.amount,
        "workers_pending": bot.already_pending(UnitTypeId.SCV),
        "marines": army_units(bot).amount,
        "marines_pending": bot.already_pending(UnitTypeId.MARINE),
        "townhalls": bot.townhalls.ready.amount,
        "townhalls_pending": bot.already_pending(UnitTypeId.COMMANDCENTER),
        "supply_depots": depots.ready.amount,
        "supply_depots_pending": bot.already_pending(UnitTypeId.SUPPLYDEPOT),
        "barracks": rax.ready.amount,
        "barracks_pending": bot.already_pending(UnitTypeId.BARRACKS),
        "barracks_idle": rax.ready.idle.amount,
        "refineries": bot.gas_buildings.amount,
        "engineering_bay": bot.structures(UnitTypeId.ENGINEERINGBAY).ready.amount,
        "infantry_weapons_1": float(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1 in bot.state.upgrades),
        "attacking": float(bot.attacking),
        "enemy_units_visible": bot.enemy_units.amount,
        "enemy_structures_known": bot.enemy_structures.amount,
        "enemy_near_base": near_base,
        "game_minutes": bot.time / 60,
        "enemy_terran": float(bot.enemy_race == Race.Terran),
        "enemy_zerg": float(bot.enemy_race == Race.Zerg),
        "enemy_protoss": float(bot.enemy_race == Race.Protoss),
    }


def observe(bot: RLBot) -> np.ndarray:
    raw = raw_features(bot)
    obs = np.array([raw[name] / norm for name, norm in FEATURES], dtype=np.float32)
    return np.clip(obs, 0.0, OBS_CLIP)


def describe(obs: np.ndarray) -> dict[str, float]:
    """Inverse of `observe` (up to clipping), handy for scripted policies and debugging."""
    return {name: float(obs[i]) * norm for i, (name, norm) in enumerate(FEATURES)}
