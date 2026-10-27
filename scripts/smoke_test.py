"""Plays one game against the built-in AI to check that the installation works.

python scripts/smoke_test.py                  # scripted policy, should beat VeryEasy
python scripts/smoke_test.py --policy random  # random valid actions
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sc2agent.actions import Action
from sc2agent.config import GameConfig
from sc2agent.env import SC2Env
from sc2agent.features import describe


def scripted_policy(obs: np.ndarray, mask: np.ndarray) -> int:
    """A fixed marine build. Shows that the action space is able to win."""
    f = describe(obs)
    if mask[Action.BUILD_SUPPLY_DEPOT] and f["supply_left"] < 5 and f["supply_depots_pending"] < 1:
        return Action.BUILD_SUPPLY_DEPOT
    if mask[Action.TRAIN_SCV] and f["workers"] + f["workers_pending"] < 22:
        return Action.TRAIN_SCV
    if mask[Action.BUILD_BARRACKS] and f["barracks"] + f["barracks_pending"] < 4:
        return Action.BUILD_BARRACKS
    if mask[Action.TRAIN_MARINE]:
        return Action.TRAIN_MARINE
    if mask[Action.ATTACK] and f["marines"] >= 20 and f["attacking"] < 0.5:
        return Action.ATTACK
    return Action.NOOP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--map", default=GameConfig.map_name)
    parser.add_argument("--difficulty", default="VeryEasy")
    parser.add_argument("--enemy-race", default="Random")
    parser.add_argument("--time-limit", type=int, default=20 * 60, help="game-time seconds before a tie")
    parser.add_argument("--realtime", action="store_true", help="play at normal speed so you can watch")
    parser.add_argument("--replay-dir", default="replays")
    args = parser.parse_args()

    config = GameConfig(
        map_name=args.map,
        difficulty=args.difficulty,
        enemy_race=args.enemy_race,
        game_time_limit=args.time_limit,
        realtime=args.realtime,
        save_replay_dir=args.replay_dir,
    )
    rng = np.random.default_rng()
    env = SC2Env(config)
    try:
        start = time.time()
        obs, _ = env.reset()
        print(f"Game started after {time.time() - start:.0f}s on {args.map} vs {args.enemy_race} {args.difficulty}")
        steps, total_reward, done = 0, 0.0, False
        while not done:
            mask = env.action_masks()
            if args.policy == "random":
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                action = scripted_policy(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        print(
            f"Result: {info['result']} after {info['game_time'] / 60:.1f} game minutes, "
            f"{steps} decisions, total reward {total_reward:.2f}, {time.time() - start:.0f}s wall time"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
