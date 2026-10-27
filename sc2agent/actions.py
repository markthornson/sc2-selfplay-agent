"""Macro action space for the Terran agent.

The policy picks one high-level action per decision step. Low-level details
(worker selection, building placement, target selection) are scripted, which
keeps the action space small enough to learn from scratch on one GPU.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

if TYPE_CHECKING:
    from sc2agent.bot import RLBot


class Action(IntEnum):
    NOOP = 0
    TRAIN_SCV = 1
    BUILD_SUPPLY_DEPOT = 2
    BUILD_BARRACKS = 3
    TRAIN_MARINE = 4
    BUILD_REFINERY = 5
    EXPAND = 6
    BUILD_ENGINEERING_BAY = 7
    RESEARCH_INFANTRY_WEAPONS = 8
    ATTACK = 9
    DEFEND = 10


NUM_ACTIONS = len(Action)

MAX_WORKERS = 66
MAX_TOWNHALLS = 4
WEAPON_UPGRADES = (
    UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
    UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
    UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
)


def army_units(bot: RLBot):
    return bot.units(UnitTypeId.MARINE)


def next_weapon_upgrade(bot: RLBot) -> UpgradeId | None:
    for upgrade in WEAPON_UPGRADES:
        if upgrade not in bot.state.upgrades and not bot.already_pending_upgrade(upgrade):
            return upgrade
    return None


def action_mask(bot: RLBot) -> np.ndarray:
    """Boolean mask of the actions that can currently do something."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[Action.NOOP] = True
    if not bot.townhalls:
        return mask

    idle_ths = bot.townhalls.ready.idle
    idle_rax = bot.structures(UnitTypeId.BARRACKS).ready.idle
    has_workers = bool(bot.workers)
    n_workers = bot.workers.amount + bot.already_pending(UnitTypeId.SCV)

    mask[Action.TRAIN_SCV] = bool(idle_ths) and n_workers < MAX_WORKERS and bot.can_afford(UnitTypeId.SCV)
    mask[Action.BUILD_SUPPLY_DEPOT] = has_workers and bot.supply_cap < 200 and bot.can_afford(UnitTypeId.SUPPLYDEPOT)
    mask[Action.BUILD_BARRACKS] = (
        has_workers and bot.tech_requirement_progress(UnitTypeId.BARRACKS) == 1 and bot.can_afford(UnitTypeId.BARRACKS)
    )
    mask[Action.TRAIN_MARINE] = bool(idle_rax) and bot.can_afford(UnitTypeId.MARINE)
    mask[Action.BUILD_REFINERY] = (
        has_workers
        and bot.can_afford(UnitTypeId.REFINERY)
        and any(_free_geysers(bot, th) for th in bot.townhalls.ready)
    )
    mask[Action.EXPAND] = (
        has_workers
        and bot.townhalls.amount + bot.already_pending(UnitTypeId.COMMANDCENTER) < MAX_TOWNHALLS
        and bot.can_afford(UnitTypeId.COMMANDCENTER)
    )
    mask[Action.BUILD_ENGINEERING_BAY] = (
        has_workers
        and bot.structures(UnitTypeId.ENGINEERINGBAY).amount + bot.already_pending(UnitTypeId.ENGINEERINGBAY) == 0
        and bot.tech_requirement_progress(UnitTypeId.ENGINEERINGBAY) == 1
        and bot.can_afford(UnitTypeId.ENGINEERINGBAY)
    )
    upgrade = next_weapon_upgrade(bot)
    mask[Action.RESEARCH_INFANTRY_WEAPONS] = (
        upgrade is not None
        and bool(bot.structures(UnitTypeId.ENGINEERINGBAY).ready.idle)
        and bot.can_afford(upgrade)
        # Level 2 and 3 need an armory; that is out of scope for this action set.
        and upgrade == UpgradeId.TERRANINFANTRYWEAPONSLEVEL1
    )
    has_army = bool(army_units(bot))
    mask[Action.ATTACK] = has_army
    mask[Action.DEFEND] = has_army
    return mask


def _free_geysers(bot: RLBot, townhall):
    taken = {r.position for r in bot.gas_buildings}
    return [g for g in bot.vespene_geyser.closer_than(10, townhall) if g.position not in taken]


async def execute(bot: RLBot, action: int) -> None:
    """Carry out one macro action. Invalid actions are silently ignored."""
    action = Action(action)
    if action == Action.NOOP or not bot.townhalls:
        return
    main = bot.townhalls.closest_to(bot.start_location)

    if action == Action.TRAIN_SCV:
        th = bot.townhalls.ready.idle
        if th and bot.can_afford(UnitTypeId.SCV):
            th.first.train(UnitTypeId.SCV)

    elif action == Action.BUILD_SUPPLY_DEPOT:
        await bot.build(UnitTypeId.SUPPLYDEPOT, near=main.position.towards(bot.game_info.map_center, 6))

    elif action == Action.BUILD_BARRACKS:
        await bot.build(UnitTypeId.BARRACKS, near=main.position.towards(bot.game_info.map_center, 9), placement_step=3)

    elif action == Action.TRAIN_MARINE:
        bot.train(UnitTypeId.MARINE, amount=bot.structures(UnitTypeId.BARRACKS).ready.idle.amount)

    elif action == Action.BUILD_REFINERY:
        for th in bot.townhalls.ready:
            geysers = _free_geysers(bot, th)
            if geysers:
                worker = bot.select_build_worker(geysers[0].position)
                if worker:
                    worker.build_gas(geysers[0])
                break

    elif action == Action.EXPAND:
        await bot.expand_now()

    elif action == Action.BUILD_ENGINEERING_BAY:
        await bot.build(UnitTypeId.ENGINEERINGBAY, near=main.position.towards(bot.game_info.map_center, -6))

    elif action == Action.RESEARCH_INFANTRY_WEAPONS:
        if bot.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1):
            bot.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)

    elif action == Action.ATTACK:
        bot.attacking = True

    elif action == Action.DEFEND:
        bot.attacking = False
