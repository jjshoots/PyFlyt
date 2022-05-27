import gym

import PyFlyt

env = gym.make("PyFlyt/SimpleWaypoint-v0")

env.reset()

env.render()

for i in range(10000):
    env.step(env.action_space.sample())
