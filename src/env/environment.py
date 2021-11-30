import os
import math
import numpy as np

from env.aviary import *

class Environment():
    """
    Wrapper for Aviary
    """
    def __init__(self, drone_dir, num_envs, max_steps=math.inf, render=None):
        self.max_steps = max_steps
        self.drone_dir = drone_dir

        self.render = num_envs == 1 if render is None else render

        self.num_actions = 2

        self.reset()


    def reset(self):
        try:
            self.env.disconnect()
        except:
            pass

        self.step_count = 0

        self.env = Aviary(drone_dir=self.drone_dir, render=self.render)
        self.env.set_mode(5)

        # clear argsv[0] message, I don't know why it works don't ask me why it works
        print ("\033[A                             \033[A")

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        self.states = self.env.states


    def get_state(self):
        return self.env.states


    def step(self, setpoints):
        """
        step the entire simulation
            output is states, reward, dones, label
        """

        self.env.set_setpoints(setpoints)
        self.env.step()
        self.step_count += 1

        return self.env.states, 0, 0, None
