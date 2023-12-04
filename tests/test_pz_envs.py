"""Tests the API compatibility of all PyFlyt Pettingzoo Environments."""
import warnings
from typing import Any

import pytest
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.test.seed_test import check_environment_deterministic_parallel

from PyFlyt.pz_envs import MAQuadXHoverEnv

# waypoint envs
_ALL_ENV_CONFIGS = []
_ALL_ENV_CONFIGS.append(
    (
        MAQuadXHoverEnv,
        dict(
            sparse_reward=False,
            flight_dome_size=50.0,
            agent_hz=30,
            angle_representation="quaternion",
        ),
    ),
)
_ALL_ENV_CONFIGS.append(
    (
        MAQuadXHoverEnv,
        dict(
            sparse_reward=True,
            flight_dome_size=50.0,
            agent_hz=30,
            angle_representation="euler",
        ),
    ),
)

# can be edited depending on gymnasium version
CHECK_ENV_IGNORE_WARNINGS = [
    "Agent's minimum observation space value is -infinity. This is probably too low.",
    "Agent's maximum observation space value is infinity. This is probably too high",
]


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_check_env(env_config: tuple[ParallelEnv, dict[str, Any]]):
    """Check that environment pass the pettingzoo api_test."""
    env = env_config[0](**env_config[1])  # pyright: ignore

    with warnings.catch_warnings(record=True) as caught_warnings:
        print(caught_warnings)
        parallel_api_test(env, num_cycles=1000)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise AssertionError(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_seeding(env_config: tuple[ParallelEnv, dict[str, Any]]):
    """Check that two AEC environments execute the same way."""
    env1 = env_config[0](**env_config[1])  # pyright: ignore
    env2 = env_config[0](**env_config[1])  # pyright: ignore

    check_environment_deterministic_parallel(env1, env2, num_cycles=1000)
