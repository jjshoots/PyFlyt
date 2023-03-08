"""Implement a controller that only wants the drone to be at x=1, y=1, z=1, while constantly spinning at yawrate=0.5, building off mode 6."""
import numpy as np

from PyFlyt.core import Aviary
from PyFlyt.core.abstractions import ControlClass


class CustomController(ControlClass):
    """A custom controller that inherits from the CtrlClass."""

    def __init__(self):
        """Initialize the controller here."""
        pass

    def reset(self):
        """Reset the internal state of the controller here."""
        pass

    def step(self, state: np.ndarray, setpoint: np.ndarray):
        """Step the controller here.

        Args:
            state (np.ndarray): Current state of the UAV
            setpoint (np.ndarray): Desired setpoint
        """
        # outputs a command to base flight mode 6 that makes the drone stay at x=1, y=1, z=1, yawrate=0.1
        target_velocity = np.array([1.0, 1.0, 1.0]) - state[-1]
        target_yaw_rate = 0.5
        output = np.array([*target_velocity[:2], target_yaw_rate, target_velocity[-1]])
        return output


# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx")

# register our custom controller for the first drone, this controller is id 8, and is based off 6
env.drones[0].register_controller(
    controller_constructor=CustomController, controller_id=8, base_mode=6
)

# set to our new custom controller
env.set_mode(8)

# run the sim
for i in range(1000):
    env.step()
