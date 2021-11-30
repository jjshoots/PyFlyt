import math
import numpy as np
import matplotlib.pyplot as plt

from env.environment import *

if __name__ == '__main__':
    env = Environment(
        drone_dir='models/vehicles/',
        num_envs=1,
        max_steps=1000,
        render=True
        )

    record_setpoints = []
    record_states = []
    for i in range(20000):
        states = env.get_state()

        s = math.sin(i / 200)
        c = math.cos(i / 200)
        s = c = 0

        setpoints = [np.array([s, c, 0., .2]) for _ in range(env.env.num_drones)]

        env.step(setpoints)

        # record_setpoints.append(setpoints[0][0])
        # record_states.append(states[0][2][0])

    # plt.plot(np.arange(len(record_setpoints)), np.array(record_setpoints), label='setpoint')
    # plt.plot(np.arange(len(record_setpoints)), np.array(record_states), label='state')
    # plt.legend()
    # plt.show()
