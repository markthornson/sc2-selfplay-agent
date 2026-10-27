"""Trains the agent with MaskablePPO against the built-in AI.

    python scripts/train.py                               # new run: 6 parallel games vs VeryEasy
    python scripts/train.py --resume checkpoints/<run>/rl_model_100000_steps.zip

Follow the progress with TensorBoard: tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from sc2agent.config import GameConfig
from sc2agent.env import SC2VecEnv


class GameStatsCallback(BaseCallback):
    """Logs win rate and game length over the last `window` finished games."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.results: deque[str] = deque(maxlen=window)
        self.game_minutes: deque[float] = deque(maxlen=window)
        self.games = 0
        self.errors = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            result = info.get("result", "Unknown")
            if result == "Error":
                self.errors += 1
                continue
            self.games += 1
            self.results.append(result)
            self.game_minutes.append(info.get("game_time", 0.0) / 60)
        if self.results:
            n = len(self.results)
            self.logger.record("games/win_rate", self.results.count("Victory") / n)
            self.logger.record("games/loss_rate", self.results.count("Defeat") / n)
            self.logger.record("games/tie_rate", self.results.count("Tie") / n)
            self.logger.record("games/game_minutes", sum(self.game_minutes) / n)
        self.logger.record("games/played", self.games)
        self.logger.record("games/errors", self.errors)
        return True

    def _on_rollout_end(self) -> None:
        if self.results:
            n = len(self.results)
            print(
                f"[{time.strftime('%H:%M:%S')}] {self.num_timesteps} steps, {self.games} games, "
                f"win rate over last {n}: {self.results.count('Victory') / n:.0%}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-envs", type=int, default=6, help="SC2 games running in parallel")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="decisions to train for")
    parser.add_argument("--map", default=GameConfig.map_name)
    parser.add_argument("--difficulty", default="VeryEasy")
    parser.add_argument("--enemy-race", default="Random")
    parser.add_argument("--time-limit", type=int, default=20 * 60, help="game-time seconds before a tie")
    parser.add_argument("--run-name", default=None, help="defaults to the date and time")
    parser.add_argument("--resume", default=None, help="checkpoint .zip to continue from")
    parser.add_argument("--save-every", type=int, default=50_000, help="decisions between checkpoints")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = Path("checkpoints") / run_name
    config = GameConfig(
        map_name=args.map,
        difficulty=args.difficulty,
        enemy_race=args.enemy_race,
        game_time_limit=args.time_limit,
    )
    venv = SC2VecEnv([config] * args.n_envs)

    if args.resume:
        model = MaskablePPO.load(args.resume, env=venv, device=args.device)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log="runs",
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    callbacks = [
        GameStatsCallback(),
        CheckpointCallback(save_freq=max(args.save_every // args.n_envs, 1), save_path=str(checkpoint_dir)),
    ]
    print(f"Run {run_name}: {args.n_envs} games vs {args.enemy_race} {args.difficulty} on {args.map}")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=not args.resume,
        )
        model.save(checkpoint_dir / "final")
        print(f"Done. Model saved to {checkpoint_dir / 'final.zip'}")
    except KeyboardInterrupt:
        model.save(checkpoint_dir / "interrupted")
        print(f"Stopped. Model saved to {checkpoint_dir / 'interrupted.zip'}")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
