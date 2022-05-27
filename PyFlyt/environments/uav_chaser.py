import numpy as np
import math
import gym
from gym import spaces

from PyFlyt.core.aviary import Aviary


class Environment(gym.Env):
    """
    Wrapper for Aviary
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=10000):

        # Gym stuff
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 19), dtype=np.float32)
        self.action_space = spaces.Box(low=-360.0, high=360.0, shape=(1, 4), dtype=np.float32)

        # environment params
        self.to_render = False
        self.max_steps = max_steps

        # environment runtime variables
        self.env = None

    def render(self):
        self.to_render = True

    def reset(self):
        # if we already have an env, disconnect from it
        if self.env is not None:
            self.env.disconnect()

        # reset step count
        self.step_count = 0

        # init env
        self.env = Aviary(
            start_pos=np.array([0.0, 0.0, 1.0]), start_orn=np.array([0.0, 0.0, 0.0]), render=self.to_render
        )

        # set flight mode
        self.env.set_mode(4)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        return self.states

    @property
    def states(self):
        return self.env.states

    @property
    def done(self):
        if self.step_count > self.max_steps:
            return True

    @property
    def reward(self):
        return 0.

    def step(self, action):
        """
        step the entire simulation
            output is states, reward, dones
        """

        self.env.set_setpoints(action)
        self.env.step()
        self.step_count += 1

        return self.states, self.reward, self.done, None
