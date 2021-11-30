import math
import numpy as np

from env.environment import *

if __name__ == '__main__':
    env = Environment(
        drone_dir='models/vehicles/',
        num_envs=1,
        max_steps=1000,
        )

    for i in range(100000):
        states = env.get_state()

        setpoints = [np.array([0., 0., 0., 0.]) for _ in range(env.env.num_drones)]

        for j, setpoint in enumerate(setpoints):
            k = i / 400.
            setpoint[0] = 5 * math.sin(k)
            setpoint[1] = 5 * math.cos(k)
            setpoint[3] = states[j][-1][-1]

        env.step(setpoints)
