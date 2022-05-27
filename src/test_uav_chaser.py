import numpy as np
import gym

import PyFlyt

env = gym.make("PyFlyt/SimpleWaypointEnv-v0")

env.render()
obs = env.reset()

for i in range(10000):
    action = obs[0, -4:]
    action = np.array([[action[0], action[1], action[3], action[2]]])
    obs, rew, dne, _ = env.step(action)
