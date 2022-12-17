"""Spawn a single fixed wing UAV on x=0, y=0, z=50, with 0 rpy."""
import numpy as np

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 5.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True)

# set to position control
env.set_mode(7)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()