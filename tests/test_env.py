"""Tests for the env <-> game-worker protocol, using a fake game instead of SC2."""

import os
from pathlib import Path

import numpy as np
import pytest
from sb3_contrib.common.maskable.utils import get_action_masks

from sc2agent.actions import NUM_ACTIONS
from sc2agent.bot import LEAVE_GAME
from sc2agent.config import GameConfig
from sc2agent.env import SC2Env, SC2VecEnv
from sc2agent.features import OBS_SIZE

GAME_LENGTH = 3


def fake_worker(conn, config):
    """Plays fake games of GAME_LENGTH decisions that always end in victory."""
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    while True:
        if conn.recv() == "close":
            return
        for step in range(GAME_LENGTH):
            obs = np.full(OBS_SIZE, step, dtype=np.float32)
            conn.send(("obs", obs, mask, 0.0, {}))
            if conn.recv() == LEAVE_GAME:
                conn.send(("done", obs, mask, -1.0, {"result": "Defeat", "truncated": False}))
                break
        else:
            conn.send(("done", obs, mask, 1.0, {"result": "Victory", "truncated": False}))


def crash_once_worker(conn, config):
    """Dies mid-game the first time it runs (marker file in save_replay_dir), then behaves."""
    marker = Path(config.save_replay_dir) / "crashed"
    if not marker.exists():
        conn.recv()
        conn.send(("obs", np.zeros(OBS_SIZE, dtype=np.float32), np.ones(NUM_ACTIONS, dtype=bool), 0.0, {}))
        conn.recv()
        marker.touch()
        os._exit(1)
    fake_worker(conn, config)


def broken_worker(conn, config):
    conn.recv()
    conn.send(("error", "Traceback: map not found"))
    conn.recv()


@pytest.fixture
def env():
    e = SC2Env(GameConfig(), worker_fn=fake_worker)
    yield e
    e.close()


def test_episode_runs_to_victory(env):
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,)
    assert env.action_masks().shape == (NUM_ACTIONS,)
    for _ in range(GAME_LENGTH - 1):
        _, reward, terminated, truncated, _ = env.step(0)
        assert (reward, terminated, truncated) == (0.0, False, False)
    _, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["result"] == "Victory"


def test_reset_mid_game_abandons_it(env):
    env.reset()
    env.step(0)
    obs, _ = env.reset()
    assert obs[0] == 0


def test_step_without_reset_fails(env):
    with pytest.raises(RuntimeError):
        env.step(0)


def test_worker_error_is_raised():
    env = SC2Env(GameConfig(), worker_fn=broken_worker)
    try:
        with pytest.raises(RuntimeError, match="map not found"):
            env.reset()
    finally:
        env.close()


def test_vec_env_auto_resets():
    venv = SC2VecEnv([GameConfig()] * 2, worker_fn=fake_worker)
    try:
        obs = venv.reset()
        assert obs.shape == (2, OBS_SIZE)
        assert get_action_masks(venv).shape == (2, NUM_ACTIONS)
        for _ in range(GAME_LENGTH - 1):
            obs, rewards, dones, _ = venv.step(np.zeros(2, dtype=int))
            assert not dones.any()
        obs, rewards, dones, infos = venv.step(np.zeros(2, dtype=int))
        assert dones.all() and (rewards == 1.0).all()
        assert "terminal_observation" in infos[0]
        assert (obs[:, 0] == 0).all()  # first observation of the next game
    finally:
        venv.close()


def test_vec_env_restarts_crashed_game(tmp_path):
    venv = SC2VecEnv([GameConfig(save_replay_dir=str(tmp_path))], worker_fn=crash_once_worker)
    try:
        venv.reset()
        obs, rewards, dones, infos = venv.step(np.zeros(1, dtype=int))
        assert dones[0] and infos[0]["result"] == "Error"
        assert infos[0]["TimeLimit.truncated"]
        for _ in range(GAME_LENGTH - 1):
            obs, rewards, dones, infos = venv.step(np.zeros(1, dtype=int))
            assert not dones[0]
        obs, rewards, dones, infos = venv.step(np.zeros(1, dtype=int))
        assert dones[0] and infos[0]["result"] == "Victory"
    finally:
        venv.close()
