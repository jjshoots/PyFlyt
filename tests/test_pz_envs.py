"""Tests the API compatibility of all PyFlyt Pettingzoo Environments."""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import pytest
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.test.seed_test import check_environment_deterministic_parallel

from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2, MAQuadXHoverEnvV2

_QUADX_HOVER_ENVS = []
for env_class, angle_representation, sparse_reward in itertools.product(
    [
        MAQuadXHoverEnvV2,
    ],
    ["euler", "quaternion"],
    [True, False],
):
    _QUADX_HOVER_ENVS.append(
        (
            env_class,
            dict(
                angle_representation=angle_representation,
                sparse_reward=sparse_reward,
            ),
        )
    )

_FIXEDWING_DOGFIGHT_ENVS = []
for (
    env_class,
    team_size,
    assisted_flight,
    sparse_reward,
    flatten_observation,
) in itertools.product(
    [MAFixedwingDogfightEnvV2],
    [1, 2, 3],
    [True, False],
    [True, False],
    [True, False],
):
    _FIXEDWING_DOGFIGHT_ENVS.append(
        (
            env_class,
            dict(
                team_size=team_size,
                assisted_flight=assisted_flight,
                sparse_reward=sparse_reward,
                flatten_observation=flatten_observation,
            ),
        )
    )

# all env configs
_ALL_ENV_CONFIGS = _QUADX_HOVER_ENVS + _FIXEDWING_DOGFIGHT_ENVS

# can be edited depending on gymnasium version
CHECK_ENV_IGNORE_WARNINGS = [
    "Agent's minimum observation space value is -infinity. This is probably too low.",
    "Agent's maximum observation space value is infinity. This is probably too high",
]


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_parallel_api(env_config: tuple[type[ParallelEnv], dict[str, Any]]):
    """Check that environment pass the pettingzoo api_test."""
    env = env_config[0](**env_config[1])

    with warnings.catch_warnings(record=True) as caught_warnings:
        parallel_api_test(env, num_cycles=1000)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise AssertionError(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_seeding(env_config: tuple[type[ParallelEnv], dict[str, Any]]):
    """Check that two AEC environments execute the same way."""
    env1 = env_config[0](**env_config[1])
    env2 = env_config[0](**env_config[1])

    check_environment_deterministic_parallel(env1, env2, num_cycles=10000)


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_observation_action_space(env_config: tuple[type[ParallelEnv], dict[str, Any]]):
    """Tests that the observation returned by the environment throughout the lifecycle is in the observation space."""
    env = env_config[0](**env_config[1])

    observations, _ = env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, *_ = env.step(actions)

        # check actions and observations
        for k, v in actions.items():
            assert env.action_space(k).contains(v)
        for k, v in observations.items():
            assert env.observation_space(k).contains(v)

    env.close()
