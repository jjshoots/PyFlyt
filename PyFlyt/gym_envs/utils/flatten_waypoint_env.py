"""Wrapper class for flattening the waypoint envs to use homogeneous observation spaces."""
from __future__ import annotations

import numpy as np
from gymnasium.core import Env, ObservationWrapper
from gymnasium.spaces import Box

from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class FlattenWaypointEnv(ObservationWrapper):
    """FlattenWaypontEnv."""

    def __init__(self, env: Env, context_length=2):
        """__init__.

        Args:
            env (Env): a PyFlyt Waypoints environment.
            context_length: how many waypoints should be included in the flattened observation space.
        """
        super().__init__(env=env)
        if not hasattr(env, "waypoints") and not isinstance(
            env.unwrapped.waypoints,  # type: ignore[reportAttributeAccess]
            WaypointHandler,
        ):
            raise AttributeError(
                "Only a waypoints environment can be used with the `FlattenWaypointEnv` wrapper."
            )
        self.context_length = context_length
        self.attitude_shape = env.observation_space["attitude"].shape[0]  # type: ignore [reportGeneralTypeIssues]
        self.target_shape = env.observation_space["target_deltas"].feature_space.shape[  # type: ignore [reportGeneralTypeIssues]
            0
        ]  # type: ignore [reportGeneralTypeIssues]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.attitude_shape + self.target_shape,)
        )

    def observation(self, observation) -> np.ndarray:
        """Flattens an observation from the super env.

        Args:
            observation: a dictionary observation with an "attitude" and "target_deltas" keys.
        """
        num_targets = min(self.context_length, observation["target_deltas"].shape[0])  # pyright: ignore[reportGeneralTypeIssues]

        targets = np.zeros((self.context_length, self.target_shape))
        targets[:num_targets] = observation["target_deltas"][:num_targets]  # pyright: ignore[reportGeneralTypeIssues]

        new_obs = np.concatenate([observation["attitude"], *targets])  # pyright: ignore[reportGeneralTypeIssues]

        return new_obs
