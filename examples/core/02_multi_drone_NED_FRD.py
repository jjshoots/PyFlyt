"""Spawns three drones, then sets all drones to have different control looprates."""
import numpy as np

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[-1.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
start_orn = np.zeros_like(start_pos)

# modify the control hz of the individual drones
drone_options = []
drone_options.append(dict(control_hz=60))
drone_options.append(dict(control_hz=120))
drone_options.append(dict(control_hz=240))

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    darw_local_axis=True,
    drone_type="quadx",
    drone_options=drone_options,
    orn_conv="NED_FRD",
)

# set to position control
env.set_mode(7)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()
