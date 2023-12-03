"""Tests the API compatibility of all PyFlyt Pettingzoo Environments."""
import warnings

import pytest
from gymnasium.utils.env_checker import data_equivalence
from pettingzoo.test import api_test
from pettingzoo.utils import wrappers

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
def test_check_env(env_config):
    """Check that environment pass the pettingzoo api_test."""
    env = env_config[0](**env_config[1])
    env = wrappers.OrderEnforcingWrapper(env)

    with warnings.catch_warnings(record=True) as caught_warnings:
        print(caught_warnings)
        api_test(env)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise AssertionError(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_config", _ALL_ENV_CONFIGS)
def test_seeding(env_config):
    """Check that two AEC environments execute the same way."""
    env1 = env_config[0](**env_config[1])
    env2 = env_config[0](**env_config[1])
    env1 = wrappers.OrderEnforcingWrapper(env1)
    env2 = wrappers.OrderEnforcingWrapper(env2)
    env1.reset(seed=42)
    env2.reset(seed=42)

    for i in env1.agents:
        env1.action_space(i).seed(seed=42)
        env2.action_space(i).seed(seed=42)
        env1.observation_space(i).seed(seed=42)
        env2.observation_space(i).seed(seed=42)

    iterations = 0
    for agent1, agent2 in zip(env1.agent_iter(), env2.agent_iter()):
        assert data_equivalence(agent1, agent2), f"Incorrect agent: {agent1} {agent2}"

        obs1, reward1, termination1, truncation1, info1 = env1.last()
        obs2, reward2, termination2, truncation2, info2 = env2.last()

        assert data_equivalence(obs1, obs2), "Incorrect observation"
        assert data_equivalence(reward1, reward2), "Incorrect reward."
        assert data_equivalence(termination1, termination2), "Incorrect termination."
        assert data_equivalence(truncation1, truncation2), "Incorrect truncation."
        assert data_equivalence(info1, info2), "Incorrect info."

        if termination1 or truncation1:
            break

        action1 = env1.action_space(agent1).sample()
        action2 = env2.action_space(agent2).sample()

        assert data_equivalence(
            action1, action2
        ), f"Incorrect actions: {action1} {action2}"

        env1.step(action1)
        env2.step(action2)

        iterations += 1

        if iterations >= 100:
            break

    env1.close()
    env2.close()
