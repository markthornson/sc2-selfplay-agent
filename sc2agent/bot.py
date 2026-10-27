"""The in-game bot. It hands observations to the learner and waits for actions.

Each game runs in its own worker process (python-sc2 installs signal handlers
and a process-wide kill switch, so a game must own the main thread of its
process). The bot talks to the environment over a multiprocessing pipe:

    bot -> env: ("obs", obs, mask, reward, info)     every decision step
    env -> bot: action (int)
    bot -> env: ("done", obs, mask, reward, info)    once, when the game ends

The env may answer -1 instead of an action to abandon the game early.
"""

from __future__ import annotations

from sc2.bot_ai import BotAI
from sc2.data import Result
from sc2.position import Point2

from sc2agent import actions, features
from sc2agent.config import GameConfig

LEAVE_GAME = -1  # sent by the env instead of an action to abandon a running game
RESULT_REWARD = {Result.Victory: 1.0, Result.Defeat: -1.0, Result.Tie: 0.0}


class RLBot(BotAI):
    def __init__(self, conn, config: GameConfig):
        super().__init__()
        self.conn = conn
        self.config = config
        self.attacking = False
        self.cleared_locations: set[Point2] = set()
        self._attack_pos: Point2 | None = None
        self.sent_done = False
        self._leaving = False
        self._last_killed_value = 0.0

    async def on_start(self) -> None:
        self.client.game_step = self.config.step_loops

    async def on_step(self, iteration: int) -> None:
        self.conn.send(("obs", features.observe(self), actions.action_mask(self), self._shaping_reward(), {}))
        action = self.conn.recv()
        if action == LEAVE_GAME:
            if not self._leaving:
                self._leaving = True
                await self.client.leave()
            return
        await actions.execute(self, action)
        await self._autopilot(iteration)

    async def on_end(self, game_result: Result) -> None:
        reward = RESULT_REWARD.get(game_result, 0.0) + self._shaping_reward()
        info = {"result": game_result.name, "game_time": self.time, "truncated": game_result == Result.Tie}
        self.conn.send(("done", features.observe(self), actions.action_mask(self), reward, info))
        self.sent_done = True

    def _shaping_reward(self) -> float:
        """Small dense reward for destroying enemy value; the win/loss signal dominates."""
        if self.config.shaping_scale == 0:
            return 0.0
        score = self.state.score
        killed = score.killed_value_units + score.killed_value_structures
        delta, self._last_killed_value = killed - self._last_killed_value, killed
        return delta * self.config.shaping_scale

    async def _autopilot(self, iteration: int) -> None:
        """Scripted chores the policy does not decide: mining and army movement."""
        if iteration % 4 == 0:
            await self.distribute_workers()
        army = actions.army_units(self)
        if not army:
            return
        if self.attacking:
            target = self._attack_target(army.center)
            retarget, self._attack_pos = target != self._attack_pos, target
            for unit in army:
                if retarget or unit.is_idle:
                    unit.attack(target)
        elif self.townhalls:
            rally = self.townhalls.closest_to(self.game_info.map_center).position.towards(self.game_info.map_center, 8)
            threats = self.enemy_units.filter(lambda u: self.townhalls.closest_distance_to(u) < 20)
            for unit in army:
                if threats:
                    unit.attack(threats.closest_to(unit).position)
                elif unit.distance_to(rally) > 6:
                    unit.move(rally)

    def _attack_target(self, from_pos: Point2) -> Point2:
        if self.enemy_structures:
            return self.enemy_structures.closest_to(from_pos).position
        if self.enemy_units:
            return self.enemy_units.closest_to(from_pos).position
        # Nothing known: sweep possible enemy bases, nearest to the enemy start first.
        candidates = list(self.enemy_start_locations) + sorted(
            self.expansion_locations_list, key=lambda p: p.distance_to(self.enemy_start_locations[0])
        )
        for loc in candidates:
            if loc in self.cleared_locations:
                continue
            if actions.army_units(self).closer_than(7, loc):
                self.cleared_locations.add(loc)
                continue
            return loc
        self.cleared_locations.clear()
        return self.enemy_start_locations[0]
