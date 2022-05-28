import numpy as np
import gym

import PyFlyt

env = gym.make("PyFlyt/SimpleHoverEnv-v0")

env.render()
obs = env.reset()

for i in range(10000):
    action = np.array([0, 0, 0, -0.1])
    obs, rew, dne, _ = env.step(action)
    print(dne, rew)
