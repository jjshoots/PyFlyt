import numpy as np
import gym

import PyFlyt

env = gym.make("PyFlyt/SimpleWaypointEnv-v0")

env.render()
obs = env.reset()

for i in range(10000):
    obs, rew, dne, _ = env.step(env.action_space.sample())
