import math

import gym
import numpy as np
import pybullet as p
from gym import spaces

from PyFlyt.core.aviary import Aviary


class SimpleHoverEnv(gym.Env):
    """
    Simple Hover Environment

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is to not crash for the longest time possible
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps=10000,
        angle_representation="quaternion",
        flight_dome_size=3.0,
    ):

        """GYM STUFF"""
        # observation size increases by 2 for euler
        if angle_representation == "euler":
            obs_shape = 12
        elif angle_representation == "quaternion":
            obs_shape = 14
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, obs_shape), dtype=np.float64
        )

        high = np.array([[3.0, 3.0, 3.0, 1.0]])
        low = np.array([[-3.0, -3.0, -3.0, -1.0]])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        """ ENVIRONMENT CONSTANTS """
        self.enable_render = False
        self.flight_dome_size = flight_dome_size
        self.max_steps = max_steps
        self.ang_rep = 0
        if angle_representation == "euler":
            self.ang_rep = 0
        elif angle_representation == "quaternion":
            self.ang_rep = 1

        """ RUNTIME VARIABLES """
        self.env = None
        self.state = self.observation_space.sample()

    def render(self, mode="human"):
        self.enable_render = True

    def reset(self):
        # if we already have an env, disconnect from it
        if self.env is not None:
            self.env.disconnect()

        # reset step count
        self.step_count = 0

        # init env
        self.env = Aviary(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.enable_render,
        )

        # set flight mode
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        self.compute_state()
        return self.state

    def compute_state(self):
        """ This computes the observation as well as the distances to target """
        # ang_vel (3/4)
        # ang_pos (3/4)
        # lin_vel (3)
        # lin_pos (3)
        # body frame distance to targ (3)
        raw_state = self.env.states[0]

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # combine everything
        new_state = np.array([0])
        if self.ang_rep == 0:
            new_state = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos]
            )
        elif self.ang_rep == 1:
            # quarternion angles
            q_ang_vel = p.getQuaternionFromEuler(ang_vel)
            q_ang_pos = p.getQuaternionFromEuler(ang_pos)

            new_state = np.array(
                [*q_ang_vel, *q_ang_pos, *lin_vel, *lin_pos]
            )

        # expand dim to be consistent with obs
        new_state = np.expand_dims(new_state, axis=0)

        # this is our state
        self.state = new_state

    @property
    def reward(self):
        reward = 1.0 if not self.done else -10
        return reward

    @property
    def done(self):
        # exceed step count
        if self.step_count > self.max_steps:
            return True

        # exceed flight dome
        if np.linalg.norm(self.state[-3:]) > self.flight_dome_size:
            return True

        return False

    def step(self, action):
        """
        step the entire simulation
            output is states, reward, dones
        """
        self.env.set_setpoints(action)
        self.env.step()
        self.compute_state()

        self.step_count += 1

        return self.state, self.reward, self.done, None
