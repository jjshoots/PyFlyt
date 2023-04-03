"""Rocket Landing Environment."""
from __future__ import annotations

import os

import numpy as np
from gymnasium.spaces import Box

from .rocket_base_env import RocketBaseEnv


class RocketLandingEnv(RocketBaseEnv):
    """Rocket Landing Environment.

    Actions are finlet_x, finlet_z, roll, booster ignition, throttle, booster gimbal x, booster gimbal z

    The goal is to land the rocket on the landing pad

    Reward:
        -100 for collisions or out of bounds,
        -0.1 otherwise
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
            sparse_reward (bool): sparse_reward
            ceiling (float): the absolute ceiling of the flying area
            max_displacement (float): the maximum horizontal distance the rocket can go
            max_duration_seconds (float): maximum simulatiaon time of the environment
            angle_representation (str): can be "euler" or "quaternion"
            agent_hz (int): looprate of the agent to environment interaction
            render_mode (None | str): can be "human" or None
            render_resolution (tuple[int, int]): render_resolution
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
        options = dict(randomize_drop=False, accelerate_drop=True)
        drone_options = dict(starting_fuel_ratio=0.01)

        super().begin_reset(seed, options, drone_options)

        # reset the tracked parameters
        self.previous_ang_vel = np.zeros((3,))
        self.previous_lin_vel = np.zeros((3,))
        self.landing_pad_contact = 0.0
        self.distance_to_pad = np.array([100.0, 100.0, 100.0])

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
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # drone to target
        self.distance_to_pad = self.landing_pad_position - lin_pos

        # combine everything
        if self.angle_representation == 0:
            self.state = np.array(
                [
                    *ang_vel,
                    *ang_pos,
                    *lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    self.landing_pad_contact,
                    *self.distance_to_pad,
                ]
            )
        elif self.angle_representation == 1:
            self.state = np.array(
                [
                    *ang_vel,
                    *quarternion,
                    *lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    self.landing_pad_contact,
                    *self.distance_to_pad,
                ]
            )

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward(
            collision_ignore_mask=[self.env.drones[0].Id, self.landing_pad_id]
        )

        if not self.sparse_reward:
            # position, orientation, and angular velocity penalties
            distance_to_pad = np.linalg.norm(self.distance_to_pad[:2] + 0.1)
            self.reward += (
                +(0.1 / np.linalg.norm(distance_to_pad))
                - (0.1 * np.linalg.norm(distance_to_pad))
                - (1.0 * np.linalg.norm(self.state[3:6]))
                - (0.2 * np.linalg.norm(self.state[0:3]))
            )

            # velocity penalty
            # the closer we are to the landing pad, the more we care about velocity
            # distance_scalar = 0.3 / np.linalg.norm(self.distance_to_pad)
            # self.reward -= distance_scalar * np.linalg.norm(self.state[6:9])
            #

        # check if we touched the landing pad
        if not self.env.collision_array[self.env.drones[0].Id, self.landing_pad_id]:
            # track the velocity for the next time
            self.previous_ang_vel = self.state[0:3]
            self.previous_lin_vel = self.state[6:9]
            self.landing_pad_contact = 0.0
            return

        # update that we touched the landing pad
        self.landing_pad_contact = 1.0

        # if collision has more than 0.35 rad/s angular velocity, we dead
        # truthfully, if collision has more than 0.55 m/s linear acceleration, we dead
        # number taken from here:
        # https://cosmosmagazine.com/space/launch-land-repeat-reusable-rockets-explained/
        # but doing so is kinda impossible for RL, so I've lessened the requirement to 1.0
        if (
            np.linalg.norm(self.previous_ang_vel) > 0.35
            or np.linalg.norm(self.previous_lin_vel) > 1.0
        ):
            # self.reward = -20.0
            self.info["fatal_collision"] = True
            self.termination |= True
            return

        # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
        if (
            np.linalg.norm(self.previous_ang_vel) < 0.02
            and np.linalg.norm(self.previous_lin_vel) < 0.02
            and np.linalg.norm(self.state[3:5]) < 0.1
        ):
            self.reward = 100.0
            self.info["env_complete"] = True
            self.termination |= True
            return
