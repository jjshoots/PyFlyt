"""Tests the the core API functionality."""
from __future__ import annotations

import numpy as np
import pytest
from custom_uavs.rocket_brick import RocketBrick

from PyFlyt.core import Aviary
from PyFlyt.core.abstractions import ControlClass, WindFieldClass


def test_simple_spawn():
    """Tests spawning a single drone."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # environment setup
    env = Aviary(
        start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx"
    )

    # set to position control
    env.set_mode(7)

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()

    env.disconnect()


def test_multi_spawn():
    """Tests spawning multiple drones, and sets them all to have different control looprates."""
    # the starting position and orientations
    start_pos = np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
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
        render=False,
        drone_type="quadx",
        drone_options=drone_options,
    )

    # set to position control
    env.set_mode(7)

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()

    env.disconnect()


def test_default_control():
    """Tests spawning a single drone and sending control commands."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # environment setup
    env = Aviary(
        start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx"
    )

    # set to position control
    env.set_mode(7)

    # for the first 500 steps, go to x=1, y=0, z=1
    setpoint = np.array([1.0, 0.0, 0.0, 1.0])
    env.set_setpoint(0, setpoint)

    for i in range(500):
        env.step()

    # for the next 500 steps, go to x=0, y=0, z=2, rotate 45 degrees
    setpoint = np.array([0.0, 0.0, np.pi / 4, 2.0])
    env.set_setpoint(0, setpoint)

    for i in range(500, 1000):
        env.step()

    env.disconnect()


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

        # check the camera image
        assert isinstance(
            env.drones[0].rgbaImg, np.ndarray
        ), f"Expected camera image to be of type `np.ndarray`, got {type(env.drones[0].rgbaImg)}."
        assert isinstance(
            env.drones[0].depthImg, np.ndarray
        ), f"Expected depth image to be of type `np.ndarray`, got {type(env.drones[0].depthImg)}."
        assert isinstance(
            env.drones[0].segImg, np.ndarray
        ), f"Expected segmented image to be of type `np.ndarray`, got {type(env.drones[0].segImg)}."
        assert (
            env.drones[0].rgbaImg.shape[-1] == 4
        ), f"Expected 4 channels in the rendered image, got {env.drones[0].rgbaImg.shape[-1]}."
        assert (
            env.drones[0].depthImg.shape[-1] == 1
        ), f"Expected 1 channel in the depth image, got {env.drones[0].depthImg.shape[-1]}."
        assert (
            env.drones[0].segImg.shape[-1] == 1
        ), f"Expected 1 channel in the segmentated image, got {env.drones[0].segImg.shape[-1]}"

    env.disconnect()


def test_custom_controller():
    """Tests implementing a custom controller"""

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
            output = np.array(
                [*target_velocity[:2], target_yaw_rate, target_velocity[-1]]
            )
            return output

    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # environment setup
    env = Aviary(
        start_pos=start_pos, start_orn=start_orn, render=False, drone_type="quadx"
    )

    # register our custom controller for the first drone, this controller is id 8, and is based off 6
    env.drones[0].register_controller(
        controller_constructor=CustomController, controller_id=8, base_mode=6
    )

    # set to our new custom controller
    env.set_mode(8)

    # run the sim
    for i in range(1000):
        env.step()

    env.disconnect()


def test_custom_uav():
    """Tests spawning in a custom UAV."""
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
        render=False,
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
            env.set_all_setpoints(np.array([[1.0, 1.0]]))

    env.disconnect()


def test_mixed_drones():
    """Tests spawning multiple different UAVs, with one having a camera."""
    # the starting position and orientations
    start_pos = np.array([[0.0, 5.0, 5.0], [0.0, 0.0, 1.0], [5.0, 0.0, 1.0]])
    start_orn = np.zeros_like(start_pos)

    # rotate the rocket upright
    start_orn[0, 0] = np.pi / 2

    # individual spawn options for each drone
    rocket_options = dict()
    quadx_options = dict(use_camera=True)
    fixedwing_options = dict(starting_velocity=np.array([0.0, 0.0, 0.0]))

    # environment setup
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        render=False,
        drone_type=["rocket", "quadx", "fixedwing"],
        drone_options=[rocket_options, quadx_options, fixedwing_options],
    )

    # set quadx to position control and fixedwing as nothing
    env.set_mode([0, 7, 0])

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        _ = env.all_states
        env.step()

    env.disconnect()


@pytest.mark.parametrize(
    "model",
    ["fixedwing", "rocket"],
)
def test_simple_wind(model: str):
    """Tests the wind field functionality

    Args:
        model (str): model name
    """

    # define the wind field
    def simple_wind(time: float, position: np.ndarray):
        wind = np.zeros_like(position)
        wind[:, -1] = np.log(position[:, -1])
        return wind

    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # environment setup, attach the windfield
    env = Aviary(
        start_pos=start_pos, start_orn=start_orn, render=False, drone_type=model
    )
    env.register_wind_field_function(simple_wind)

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()

    env.disconnect()


@pytest.mark.parametrize(
    "model",
    ["fixedwing", "rocket"],
)
def test_custom_wind(model: str):
    """Tests the wind field functionality

    Args:
        model (str): model name
    """

    # define the wind field
    class MyWindField(WindFieldClass):
        def __init__(
            self, my_parameter=1.0, np_random: None | np.random.RandomState = None
        ):
            super().__init__(np_random)
            self.strength = my_parameter

        def __call__(self, time: float, position: np.ndarray):
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
        render=False,
        drone_type=model,
        wind_type=MyWindField,
    )

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()

    env.disconnect()
