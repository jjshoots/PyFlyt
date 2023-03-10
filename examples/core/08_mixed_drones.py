"""Spawns three different types of drones, then gets all their states."""
import numpy as np

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[0.0, 5.0, 5.0], [3.0, 3.0, 1.0], [5.0, 0.0, 1.0]])
start_orn = np.zeros_like(start_pos)

# individual spawn options for each drone
rocket_options = dict()
quadx_options = dict(use_camera=True, drone_model="primitive_drone")
fixedwing_options = dict(starting_velocity=np.array([0.0, 0.0, 0.0]))

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type=["rocket", "quadx", "fixedwing"],
    drone_options=[rocket_options, quadx_options, fixedwing_options],
)

# set quadx to position control and fixedwing as nothing
env.set_mode([0, 7, 0])

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    states = env.all_states
    aux_states = env.all_aux_states
    env.step()
