"""Implements a simple time invariant, stateless wind model."""
import numpy as np

from PyFlyt.core import Aviary


# define the wind field
def simple_wind(time: float, position: np.ndarray):
    """Defines a simple wind updraft model.

    Args:
        time (float): time
        position (np.ndarray): position as an (n, 3) array
    """
    # the xy velocities are 0...
    wind = np.zeros_like(position)

    # and the vertical velocity is dependent on the logarithmic of height
    wind[:, -1] = np.log(position[:, -1])

    return wind


# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup, attach the windfield
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx")
env.register_wind_field_function(simple_wind)

# set the flight mode
env.set_mode(7)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()

env.disconnect()
