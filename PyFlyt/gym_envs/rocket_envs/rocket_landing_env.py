"""Rocket Landing Environment."""
from __future__ import annotations

import os

import numpy as np
import pybullet as p
from gymnasium.spaces import Box

from .rocket_base_env import RocketBaseEnv


class RocketLandingEnv(RocketBaseEnv):
    """Rocket Landing Environment.

    Actions are finlet_x, finlet_y, finlet_roll, booster ignition, throttle, booster gimbal x, booster gimbal y
    The goal is to land the rocket on the landing pad.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        ceiling (float): the absolute ceiling of the flying area.
        max_displacement (float): the maximum horizontal distance the rocket can go.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (str): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction..
        render_mode (None | str): can be "human" or None.
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        sparse_reward: bool = False,
        ceiling: float = 500.0,
        max_displacement: float = 200.0,
        max_duration_seconds: float = 30.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            ceiling (float): the absolute ceiling of the flying area.
            max_displacement (float): the maximum horizontal distance the rocket can go.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (str): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction..
            render_mode (None | str): can be "human" or None.
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, ceiling * 0.9]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            ceiling=ceiling,
            max_displacement=max_displacement,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        """GYMNASIUM STUFF"""
        # the space is the standard space + pad touch indicator + relative pad location
        max_offset = max(ceiling, max_displacement)
        low = self.combined_space.low
        low = np.concatenate(
            (low, np.array([0.0, -max_offset, -max_offset, -max_offset]))
        )
        high = self.combined_space.high
        high = np.concatenate(
            (high, np.array([1.0, max_offset, max_offset, max_offset]))
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float64)

        # the landing pad
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, "../../models/landing_pad.urdf")

        """CONSTANTS"""
        self.sparse_reward = sparse_reward

    def reset(self, seed=None, options=dict()):
        """Resets the environment.

        Args:
            seed: int
            options: None
        """
        options = dict(randomize_drop=True, accelerate_drop=True)
        drone_options = dict(starting_fuel_ratio=0.01)

        super().begin_reset(seed, options, drone_options)

        # reset the tracked parameters
        self.landing_pad_contact = 0.0
        self.ang_vel = np.zeros((3,))
        self.lin_vel = np.zeros((3,))
        self.distance = np.zeros((3,))
        self.previous_ang_vel = np.zeros((3,))
        self.previous_lin_vel = np.zeros((3,))
        self.previous_distance = np.zeros((3,))

        # randomly generate the target landing location
        theta = self.np_random.uniform(0.0, 2.0 * np.pi)
        distance = self.np_random.uniform(0.0, 0.05 * self.ceiling)
        self.landing_pad_position = (
            np.array([np.cos(theta), np.sin(theta), 0.1]) * distance
        )
        self.landing_pad_id = self.env.loadURDF(
            self.targ_obj_dir,
            basePosition=self.landing_pad_position,
            useFixedBase=True,
        )

        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self):
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        """
        # update the previous values to current values
        self.previous_ang_vel = self.ang_vel.copy()
        self.previous_lin_vel = self.lin_vel.copy()
        self.previous_distance = self.distance.copy()

        # update current values
        (
            self.ang_vel,
            self.ang_pos,
            self.lin_vel,
            lin_pos,
            quarternion,
        ) = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # drone to landing pad
        rotation = np.array(p.getMatrixFromQuaternion(quarternion)).reshape(3, 3)
        self.distance = lin_pos - self.landing_pad_position
        rotated_distance = np.matmul(self.distance, rotation)

        # combine everything
        if self.angle_representation == 0:
            self.state = np.array(
                [
                    *self.ang_vel,
                    *self.ang_pos,
                    *self.lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    self.landing_pad_contact,
                    *rotated_distance,
                ]
            )
        elif self.angle_representation == 1:
            self.state = np.array(
                [
                    *self.ang_vel,
                    *quarternion,
                    *self.lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    self.landing_pad_contact,
                    *rotated_distance,
                ]
            )

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward(
            collision_ignore_mask=[self.env.drones[0].Id, self.landing_pad_id]
        )

        # compute reward
        if not self.sparse_reward:
            # progress and distance to pad
            progress_to_pad = float(  # noqa
                np.linalg.norm(self.previous_distance[:2])
                - np.linalg.norm(self.distance[:2])
            )
            offset_to_pad = np.linalg.norm(self.distance[:2]) + 0.1  # noqa

            # deceleration as long as we're still falling
            deceleration_bonus = (  # noqa
                max(
                    (self.lin_vel[-1] < 0.0)
                    * (self.lin_vel[-1] - self.previous_lin_vel[-1]),
                    0.0,
                )
                / self.distance[-1]
            )

            # composite reward together
            self.reward += (
                -5.0  # negative offset to discourage staying in the air
                + (2.0 / offset_to_pad)  # encourage being near the pad
                + (100.0 * progress_to_pad)  # encourage progress to landing pad
                - (1.0 * abs(self.ang_vel[-1]))  # minimize spinning
                - (3.0 * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
                # + (5.0 * deceleration_bonus)  # reward deceleration when near pad
            )

            # -5.0  # negative offset to discourage staying in the air
            # + (2.0 / offset_to_pad)  # encourage being near the pad
            # + (100.0 * progress_to_pad)  # encourage progress to landing pad
            # -(1.0 * abs(self.ang_vel[-1]))  # minimize spinning
            # - (1.0 * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
            # # + (5.0 * deceleration_bonus)  # reward deceleration when near pad

        # check if we touched the landing pad
        if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
            self.landing_pad_contact = 1.0
            self.reward += 20
        else:
            self.landing_pad_contact = 0.0
            return

        # if collision has more than 0.35 rad/s angular velocity, we dead
        # truthfully, if collision has more than 0.55 m/s linear acceleration, we dead
        # number taken from here:
        # https://cosmosmagazine.com/space/launch-land-repeat-reusable-rockets-explained/
        # but doing so is kinda impossible for RL, so I've lessened the requirement to 1.0
        if (
            np.linalg.norm(self.previous_ang_vel) > 0.35
            or np.linalg.norm(self.previous_lin_vel) > 1.0
        ):
            self.info["fatal_collision"] = True
            self.termination |= True
            return

        # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
        if (
            np.linalg.norm(self.previous_ang_vel) < 0.02
            and np.linalg.norm(self.previous_lin_vel) < 0.02
            and np.linalg.norm(self.ang_pos[:2]) < 0.1
        ):
            self.reward += 500.0
            self.info["env_complete"] = True
            self.termination |= True
            return
