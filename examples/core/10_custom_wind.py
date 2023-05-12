"""Implements a custom stateful wind model."""
import numpy as np

from PyFlyt.core import Aviary
from PyFlyt.core.abstractions import WindFieldClass


# define the wind field
class MyWindField(WindFieldClass):
    """A stateful wind model."""

    def __init__(
        self, my_parameter=1.0, np_random: None | np.random.RandomState = None
    ):
        """__init__.

        Args:
            my_parameter: supports an arbitrary number of parameters
            np_random (None | np.random.RandomState): np random state
        """
        super().__init__(np_random)
        self.strength = my_parameter

    def __call__(self, time: float, position: np.ndarray):
        """__call__.

        Args:
            time (float): time
            position (np.ndarray): position as an (n, 3) array
        """
        wind = np.zeros_like(position)
        wind[:, -1] = np.log(position[:, -1]) * self.strength
        wind += self.np_random.randn(*wind.shape)
        return wind


# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup, attach the windfield
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type="quadx",
    wind_type=MyWindField,
    wind_options=dict(my_parameter=1.2),
)

# set the flight mode
env.set_mode(7)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()

env.disconnect()
