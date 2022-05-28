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

    Reward is 1.0 for each time step, and -10.0 for crashing
    or going outside the flight dome.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps=10000,
        angle_representation="quaternion",
        flight_dome_size=3.0,
    ):

        """GYM STUFF"""
        # observation size increases by 1 for quaternion
        if angle_representation == "euler":
            obs_shape = 12
        elif angle_representation == "quaternion":
            obs_shape = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        high = np.array([3.0, 3.0, 3.0, 1.0])
        low = np.array([-3.0, -3.0, -3.0, 0.0])
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

    def render(self, mode="human"):
        self.enable_render = True

    def reset(self):
        # if we already have an env, disconnect from it
        if self.env is not None:
            self.env.disconnect()

        # reset step count
        self.info = {}
        self.done = False
        self.step_count = 0

        # init env
        self.env = Aviary(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.enable_render,
        )

        # set flight mode
        self.env.set_mode(6)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        return self.compute_state()

    def compute_state(self):
        """This computes the observation as well as the distances to target"""
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
            new_state = np.array([*ang_vel, *ang_pos, *lin_vel, *lin_pos])
        elif self.ang_rep == 1:
            # quarternion angles
            q_ang_pos = p.getQuaternionFromEuler(ang_pos)

            new_state = np.array([*ang_vel, *q_ang_pos, *lin_vel, *lin_pos])

        return new_state

    @property
    def reward(self):
        reward = 1.0 if not self.done else -10
        return reward

    def compute_done(self):
        # we were already done
        if self.done:
            return True

        # exceed step count
        if self.step_count > self.max_steps:
            self.info["done"] = "step_limit"
            return True

        # exceed flight dome
        if np.linalg.norm(self.state[-3:]) > self.flight_dome_size:
            self.info["done"] = "out_of_range"
            return True

        # collision
        if len(self.env.getContactPoints()) > 0:
            self.info["done"] = "collision"
            return True

        return False

    def step(self, action: np.ndarray):
        """
        step the entire simulation
            output is states, reward, dones
        """
        # unsqueeze the action to be usable in aviary
        action = np.expand_dims(action, axis=0)

        # step through env
        self.env.set_setpoints(action)
        while self.env.drones[0].steps % self.env.drones[0].update_ratio != 0:
            self.env.step()

        # compute state and done
        self.state = self.compute_state()
        self.done = self.compute_done()

        self.step_count += 1

        return self.state, self.reward, self.done, None
