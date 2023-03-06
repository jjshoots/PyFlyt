"""Tests the the core API functionality."""
import numpy as np

from PyFlyt.core import Aviary
from PyFlyt.core.abstractions import CtrlClass
from custom_uavs.rocket_brick import RocketBrick

def test_simple_spawn():
    """Tests spawning a single drone."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # environment setup
    env = Aviary(start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx")

    # set to position control
    env.set_mode(7)

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()

def test_multi_spawn():
    """Tests spawning multiple drones."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    start_orn = np.zeros_like(start_pos)

    # environment setup
    env = Aviary(start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx")

    # set to position control
    env.set_mode(7)

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()

def test_default_control():
    """Tests spawning a single drone and sending control commands."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # environment setup
    env = Aviary(start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx")

    # set to position control
    env.set_mode(7)

    # for the first 500 steps, go to x=1, y=0, z=1
    setpoint = np.array([[1.0, 0.0, 0.0, 1.0]])
    env.set_setpoints(setpoint)

    for i in range(500):
        env.step()

    # for the next 500 steps, go to x=0, y=0, z=2, rotate 45 degrees
    setpoint = np.array([[0.0, 0.0, np.pi / 4, 2.0]])
    env.set_setpoints(setpoint)

    for i in range(500, 1000):
        env.step()

def test_camera():
    """Tests the camera module."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # environment setup
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        render=False,
        drone_type="quadx",
        drone_options=dict(use_camera=True),
    )

    # set to velocity control
    env.set_mode(6)

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(100):
        env.step()

def test_custom_controller():
    """Tests implementing a custom controller"""

    class CustomController(CtrlClass):
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
    env = Aviary(start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx")

    # register our custom controller for the first drone, this controller is id 8, and is based off 6
    env.drones[0].register_controller(
        controller_constructor=CustomController, controller_id=8, base_mode=6
    )

    # set to our new custom controller
    env.set_mode(8)

    # run the sim
    for i in range(1000):
        env.step()

def test_custom_uav():
    """Tests spawning in a custom UAV."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # define a new drone type
    drone_type_mappings = dict()
    drone_type_mappings["rocket_brick"] = RocketBrick

    # environment setup
    env = Aviary(start_pos=start_pos, start_orn=start_orn, render=False, drone_type_mappings=drone_type_mappings, drone_type="rocket_brick")

    # print out the links and their names in the urdf for debugging
    env.drones[0].get_joint_info()

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()

        # ignite the rocket after ~1 seconds
        if i > 100:
            env.set_setpoints(np.array([[1.0, 1.0]]))

