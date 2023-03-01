"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""
import os

import numpy as np

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    model_dir=os.path.dirname(os.path.abspath(__file__)),
    drone_model="new_uav",
    render=True,
)

# set to position control
env.set_mode(7)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()
