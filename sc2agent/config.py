from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Settings for one game against the built-in AI."""

    map_name: str = "AcropolisLE"
    enemy_race: str = "Random"  # Terran, Zerg, Protoss or Random
    # VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane
    difficulty: str = "VeryEasy"
    # Game loops between decisions. 22.4 loops is one second of game time ("faster" speed).
    step_loops: int = 16
    # Games longer than this many game-time seconds end as a tie (truncated episode).
    game_time_limit: int = 30 * 60
    # Dense reward per unit of enemy resource value destroyed. 0 disables shaping.
    shaping_scale: float = 1e-4
    realtime: bool = False
    save_replay_dir: str | None = None
    # Log level for python-sc2 inside the game processes (DEBUG, INFO, WARNING, ...).
    sc2_log_level: str = "WARNING"
