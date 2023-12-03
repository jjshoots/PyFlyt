"""Multiagent QuadX Hover Environment."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv



class MAQuadXHoverEnv(MAQuadXBaseEnv):
    """Simple Multiagent Hover Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to not crash for the longest time possible.

    Args:
        sparse_reward (bool): sparse_reward
        flight_dome_size (float): flight_dome_size
        max_duration_seconds (float): max_duration_seconds
        angle_representation (str): angle_representation
        agent_hz (int): agent_hz
        render_mode (None | str): render_mode
    """

    metadata = {"render_modes": ["human"], "name": "ma_quadx_hover"}

    def __init__(
        self,
        sparse_reward: bool = False,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            sparse_reward (bool): sparse_reward
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
        """
        super().__init__(
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
        )
        self.sparse_reward = sparse_reward

        # observation space
        low = np.concatenate(self.combined_space.low, np.array([-np.inf, -np.inf, -np.inf]))
        high = np.concatenate(self.combined_space.high, np.array([np.inf, np.inf, np.inf]))
        self._observation_space = spaces.Box(low=low, high=high)

    def observation_space(self, _):
        """observation_space.

        Args:
            _:
        """
        return self._observation_space

    def reset(self, seed=None, options=dict()):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        super().begin_reset(seed, options)
        super().end_reset(seed, options)

    def observe_by_id(self, agent_id: int) -> np.ndarray:
        """observe.

        Args:
            agent:
        """
        # get all the relevant things
        raw_state = self.compute_attitude_by_id(agent_id)
        aux_state = self.compute_auxiliary_by_id(agent_id)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = raw_state

        # depending on angle representation, return the relevant thing
        if self.angle_representation == 0:
            return np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *aux_state, *self.past_actions[agent_id], *self.start_pos[agent_id]]
            )
        elif self.angle_representation == 1:
            return np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *aux_state, *self.past_actions[agent_id], *self.start_pos[agent_id]]
            )
        else:
            raise AssertionError("Not supposed to end up here!")

    def compute_term_trunc_reward_info_by_id(self, agent_id: int) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        term, trunc, reward, info = super().compute_base_term_trunc_reward_info_by_id(agent_id)

        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            linear_distance = np.linalg.norm(self.aviary.state(agent_id)[-1] - self.start_pos[agent_id])

            # how far are we from 0 roll pitch
            angular_distance = np.linalg.norm(self.aviary.state(agent_id)[1][:2])

            reward -= float(linear_distance + angular_distance)

        return term, trunc, reward, info
