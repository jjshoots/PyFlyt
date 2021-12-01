import math
from pybullet_swarming.environment.aviary import *

class Environment():
    """
    Wrapper for Aviary
    """
    def __init__(self, start_pos, start_orn, num_envs=1, max_steps=math.inf, render=None):
        self.max_steps = max_steps

        self.start_pos = start_pos
        self.start_orn = start_orn

        self.render = num_envs == 1 if render is None else render

        self.num_actions = 2

        self.reset()


    def reset(self):
        try:
            self.env.disconnect()
        except:
            pass

        self.step_count = 0

        self.env = Aviary(start_pos=self.start_pos, start_orn=self.start_orn, render=self.render)

        # clear argsv[0] message, I don't know why it works don't ask me why it works
        print ("\033[A                             \033[A")

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()


    def set_mode(self, mode):
        self.env.set_mode(mode)


    def get_state(self):
        return self.env.states


    @property
    def states(self):
        return self.get_state()


    def step(self, setpoints):
        """
        step the entire simulation
            output is states, reward, dones, label
        """

        self.env.set_setpoints(setpoints)
        self.env.step()
        self.step_count += 1

        return self.env.states, 0, 0, None
