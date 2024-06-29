"""QuadX Hover Environment."""
from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv


class QuadXPoleBalanceEnv(QuadXBaseEnv):
    """Simple Hover Environment with the additional goal of keeping a pole upright.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to not crash and not let the pole hit the ground for the longest time possible.

    Args:
    ----
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
        flight_mode: int = 0,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 20.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
        ----
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_mode (int): the flight mode of the UAV
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
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

        """GYMNASIUM STUFF"""
        self.observation_space = self.combined_space

        # the pole urdf
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.pole_obj_dir = os.path.join(file_dir, "../../models/pole.urdf")

        # modify the state to take into account the pole's state
        pole_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float64,
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + pole_space.shape[0],),
            dtype=np.float64,
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
        ----
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(
            seed,
            options,
            drone_options={"drone_model": "primitive_drone"},
        )

        # spawn in a pole and make it have enough friction
        self.poleId = self.env.loadURDF(
            self.pole_obj_dir,
            basePosition=np.array([0.0, 0.0, 1.55]),
            useFixedBase=False,
        )
        self.env.changeDynamics(
            self.poleId,
            linkIndex=1,
            lateralFriction=1.0e5,
            restitution=0.01,
        )

        self.pole_leaningness = 0.0
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
        """
        # compute attitude of self
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        rotation = (
            np.array(self.env.getMatrixFromQuaternion(quaternion)).reshape(3, 3).T
        )
        aux_state = super().compute_auxiliary()

        # we measure the top and bottom linear position and velocity of the pole
        # compute the attitude of the pole in global coords
        pole_lin_pos, pole_quaternion = self.env.getBasePositionAndOrientation(
            self.poleId
        )
        pole_lin_vel, pole_ang_vel = self.env.getBaseVelocity(self.poleId)

        pole_top_pos, *_, pole_top_vel, _ = self.env.getLinkState(
            self.poleId, linkIndex=0, computeLinkVelocity=True
        )
        pole_bot_pos, *_, pole_bot_vel, _ = self.env.getLinkState(
            self.poleId, linkIndex=1, computeLinkVelocity=True
        )
        pole_top_pos = np.array(pole_top_pos)
        pole_top_vel = np.array(pole_top_vel)
        pole_bot_pos = np.array(pole_bot_pos)
        pole_bot_vel = np.array(pole_bot_vel)

        # compute the uprightness of the pole
        if pole_top_pos[-1] > pole_bot_pos[-1]:
            self.pole_leaningness = np.linalg.norm(pole_top_pos[:2] - pole_bot_pos[:2])
        else:
            self.pole_leaningness = 1.0

        # get everything relative to self
        pole_top_pos = np.matmul(rotation, (pole_top_pos - lin_pos))
        pole_bot_pos = np.matmul(rotation, (pole_bot_pos - lin_pos))
        pole_top_vel = np.matmul(rotation, pole_top_vel) - lin_vel
        pole_bot_vel = np.matmul(rotation, pole_bot_vel) - lin_vel

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
            self.reward -= self.pole_leaningness
            self.reward += 1.0
