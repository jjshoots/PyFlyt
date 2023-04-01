"""Implements a custom UAV in the Aviary."""
import numpy as np
from custom_uavs.rocket_brick import RocketBrick

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# define a new drone type
drone_type_mappings = dict()
drone_type_mappings["rocket_brick"] = RocketBrick

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type_mappings=drone_type_mappings,
    drone_type="rocket_brick",
)

# print out the links and their names in the urdf for debugging
env.drones[0].get_joint_info()

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()

    # ignite the rocket after ~1 seconds
    if i > 100:
        env.set_setpoint(0, np.array([1.0, 1.0]))
