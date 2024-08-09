"""QuadX Pole Balance Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.pole_handler import PoleHandler


class QuadXPoleBalanceEnv(QuadXBaseEnv):
    """Simple Hover Environment with the additional goal of keeping a pole upright.

    Actions are direct motor PWM commands because any underlying controller introduces too much control latency.
    The target is to not crash and not let the pole hit the ground for the longest time possible.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_mode (int): the flight mode of the UAV
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.

    """

    def __init__(
        self,
        sparse_reward: bool = False,
        flight_mode: int = -1,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 20.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode.
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )
        # init the pole
        self.pole = PoleHandler()

        """GYMNASIUM STUFF"""
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.combined_space.shape[0] + self.pole.observation_space.shape[0],
            ),
            dtype=np.float64,
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(
            seed,
            options,
            drone_options={
                "drone_model": "primitive_drone",
                "camera_position_offset": np.array([-3.0, 0.0, 1.0]),
            },
        )
        self.pole.reset(p=self.env, start_location=np.array([0.0, 0.0, 1.55]))
        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        - 12 values for the pole's positions relative to self:
        ------ top position XYZ
        ------ bottom position XYZ
        ------ top velocity XYZ
        ------ bottom velocity XYZ
        - auxiliary information (vector of 4 values)
        """
        # compute attitude of self
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()
        rotation = (
            np.array(self.env.getMatrixFromQuaternion(quaternion)).reshape(3, 3).T
        )

        # compute the pole's states
        (
            pole_top_pos,
            pole_top_vel,
            pole_bot_pos,
            pole_bot_vel,
        ) = self.pole.compute_state(
            rotation=rotation,
            uav_lin_pos=lin_pos,
            uav_lin_vel=lin_vel,
        )

        # combine everything
        if self.angle_representation == 0:
            self.state = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                    pole_top_pos,
                    pole_bot_pos,
                    pole_top_vel,
                    pole_bot_vel,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            self.state = np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                    pole_top_pos,
                    pole_bot_pos,
                    pole_top_vel,
                    pole_bot_vel,
                ],
                axis=-1,
            )

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            linear_distance = np.linalg.norm(
                self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0])
            )

            # how far are we from 0 roll pitch
            angular_distance = np.linalg.norm(self.env.state(0)[1][:2])

            self.reward -= linear_distance + angular_distance
            self.reward -= self.pole.leaningness
            self.reward += 1.0
