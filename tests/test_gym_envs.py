"""Tests the API compatibility of all PyFlyt Gymnasium Envs."""

import itertools
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env, data_equivalence

import PyFlyt.gym_envs  # noqa
from PyFlyt.gym_envs import FlattenWaypointEnv

# waypoint envs
_WAYPOINT_ENV_CONFIGS = []
for env_name, angle_representation, sparse_reward in itertools.product(
    [
        "PyFlyt/QuadX-Waypoints-v4",
        "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/Fixedwing-Waypoints-v4",
    ],
    ["euler", "quaternion"],
    [True, False],
):
    _WAYPOINT_ENV_CONFIGS.append(
        (
            env_name,
            dict(
                angle_representation=angle_representation,
                sparse_reward=sparse_reward,
            ),
        )
    )

# non waypoint environments
_NORMAL_ENV_CONFIGS = []
for env_name, angle_representation, sparse_reward in itertools.product(
    [
        "PyFlyt/QuadX-Hover-v4",
        "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4",
        "PyFlyt/Rocket-Landing-v4",
    ],
    ["euler", "quaternion"],
    [True, False],
):
    _NORMAL_ENV_CONFIGS.append(
        (
            env_name,
            dict(
                angle_representation=angle_representation,
                sparse_reward=sparse_reward,
            ),
        )
    )

# all env configs
_ALL_ENV_CONFIGS = _NORMAL_ENV_CONFIGS + _WAYPOINT_ENV_CONFIGS

# can be edited depending on gymnasium version
CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "A Box observation space minimum value is infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "Human rendering should return `None`, got <class 'numpy.ndarray'>",
        "RGB-array rendering should return a numpy array in which the last axis has three dimensions, got 4",
    ]
]


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_check_env(env_config):
    """Check that environment pass the gymnasium check_env."""
    env = gym.make(env_config[0], **env_config[1])

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_seeding(env_config):
    """Test that pyflyt seeding works."""
    env_1 = gym.make(env_config[0], **env_config[1])
    env_2 = gym.make(env_config[0], **env_config[1])

    obs_1, info_1 = env_1.reset(seed=42)
    obs_2, info_2 = env_2.reset(seed=42)
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(100):
        actions = env_1.action_space.sample()
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()


@pytest.mark.parametrize("env_config", _WAYPOINT_ENV_CONFIGS)
@pytest.mark.parametrize("context_length", [2, 8])
def test_flatten_env(env_config, context_length):
    """Test that waypoint environments flatten properly."""
    env = gym.make(env_config[0], **env_config[1])
    env = FlattenWaypointEnv(env=env, context_length=context_length)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
@pytest.mark.parametrize("render_mode", ["human", "rgb_array"])
def test_render(env_config, render_mode):
    """Test that pyflyt rendering works."""
    env = gym.make(
        env_config[0],
        render_mode=render_mode,
        **env_config[1],
    )
    env.reset()
    frames = []
    for _ in range(10):
        frames.append(env.render())
        env.step(env.action_space.sample())

    for frame in frames:
        assert isinstance(
            frame, np.ndarray
        ), f"Expected render frames to be of type `np.ndarray`, got {type(frame)}."
        assert (
            frame.shape[-1] == 4
        ), f"Expected 4 channels in the rendered image, got {frame.shape[-1]}."

    env.close()
