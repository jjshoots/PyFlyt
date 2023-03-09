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
        ceiling: float = 300.0,
        max_displacement: float = 300.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            ceiling (float): the absolute ceiling of the flying area
            max_displacement (float): the maximum horizontal distance the rocket can go
            max_duration_seconds (float): maximum simulatiaon time of the environment
            angle_representation (str): can be "euler" or "quaternion"
            agent_hz (int): looprate of the agent to environment interaction
            render_mode (None | str): can be "human" or None
        """
        super().__init__(
            ceiling=ceiling,
            max_displacement=max_displacement,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
        )

        """GYMNASIUM STUFF"""
        low = self.combined_space.low
        low = np.concatenate((low, np.array([0.0])))
        high = self.combined_space.high
        high = np.concatenate((high, np.array([1.0])))
        self.observation_space = Box(low=low, high=high, dtype=np.float64)

        # the landing pad
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, "../../models/landing_pad.urdf")

    def reset(self, seed=None, options=dict()):
        """Resets the environment.

        Args:
            seed: int
            options: None
        """
        options = dict(randomize_drop=True)

        super().begin_reset(seed, options)

        # impart some sidewards velocity
        start_lin_vel = self.np_random.uniform(-5.0, 5.0, size=(3,))
        start_ang_vel = self.np_random.uniform(-0.5, 0.5, size=(3,))
        self.env.resetBaseVelocity(self.env.drones[0].Id, start_lin_vel, start_ang_vel)

        # reset the tracked parameters
        self.previous_ang_vel = np.zeros((3,))
        self.previous_lin_vel = np.zeros((3,))
        self.landing_pad_contact = 0.0

        # randomly generate the target landing location
        theta = self.np_random.uniform(0.0, 2.0 * np.pi)
        distance = self.np_random.uniform(0.0, 0.1 * self.max_displacement)
        target = np.array([np.cos(theta), np.sin(theta), 0.1]) * distance
        self.landing_pad_id = self.env.loadURDF(
            self.targ_obj_dir,
            basePosition=target,
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
                ]
            )

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward(
            collision_ignore_mask=[self.env.drones[0].Id, self.landing_pad_id]
        )

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
            self.reward = -100.0
            self.info["fatal_collision"] = True
            self.termination |= True
            return

        # if our both velocities are less than 0.02 m/s, we LANDED!
        if (
            np.linalg.norm(self.previous_ang_vel) < 0.02
            or np.linalg.norm(self.previous_lin_vel) < 0.02
        ):
            self.reward = 100.0
            self.info["env_complete"] = True
            self.termination |= True
            return
