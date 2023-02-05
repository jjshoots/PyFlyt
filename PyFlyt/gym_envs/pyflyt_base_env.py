from __future__ import annotations

import math

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.core.aviary import Aviary


class PyFlytBaseEnv(gymnasium.Env):
    """Base PyFlyt Environments using the Gymnasim API"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        drone_type: str = "quadx",
        drone_model: str = "cf2x",
        flight_dome_size: float = np.inf,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            max_duration_seconds (float): maximum simulatiaon time of the environment
            angle_representation (str): can be "euler" or "quaternion"
            agent_hz (int): looprate of the agent to environment interaction
            render_mode (None | str): can be "human" or None
        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if render_mode is not None:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode {render_mode}, only `human` allowed."
            self.enable_render = True
        else:
            self.enable_render = False

        """GYMNASIUM STUFF"""
        # attitude size increases by 1 for quaternion
        if angle_representation == "euler":
            attitude_shape = 16
        elif angle_representation == "quaternion":
            attitude_shape = 17
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.attitude_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(attitude_shape,), dtype=np.float64
        )

        control_surface_limit = np.inf
        thrust_limit = np.inf
        if drone_type == "quadx":
            control_surface_limit = math.pi
            thrust_limit = 0.8
        elif drone_type == "fixedwing":
            control_surface_limit = 1.0
            thrust_limit = 1.0

        high = np.array(
            [
                control_surface_limit,
                control_surface_limit,
                control_surface_limit,
                thrust_limit,
            ]
        )
        low = np.array(
            [
                -control_surface_limit,
                -control_surface_limit,
                -control_surface_limit,
                0.0,
            ]
        )
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        """ ENVIRONMENT CONSTANTS """
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.drone_type = drone_type
        self.drone_model = drone_model
        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

    def reset(self, seed=None, options=None):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options:
        """
        raise NotImplementedError

    def begin_reset(self, seed=None, options=None, aviary_options=dict()):
        """The first half of the reset function"""
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.state = None
        self.action = np.zeros((4,))
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        # init env
        self.env = Aviary(
            drone_type=self.drone_type,
            drone_model=self.drone_model,
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            render=self.enable_render,
            seed=seed,
            **aviary_options,
        )

    def end_reset(self, seed=None, options=None):
        """The tailing half of the reset function"""
        # register all new collision bodies
        self.env.register_all_new_bodies()

        # set flight mode
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        self.compute_state()

    def compute_state(self):
        raise NotImplementedError

    def compute_attitude(self):
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        """
        raw_state = self.env.states[0]

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        quarternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quarternion

    def compute_term_trunc_reward(self):
        raise NotImplementedError

    def compute_base_term_trunc_reward(self):
        """compute_base_term_trunc_reward."""
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation = self.truncation or True

        # collision
        if np.any(self.env.collision_array):
            self.reward = -100.0
            self.info["collision"] = True
            self.termination = self.termination or True

        # exceed flight dome
        if np.linalg.norm(self.env.states[0][-1]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination = self.termination or True

    def step(self, action: np.ndarray):
        """Steps the environment

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info
        """
        # unsqueeze the action to be usable in aviary
        self.action = action.copy()
        action = np.expand_dims(action, axis=0)

        # reset the reward and set the action
        self.reward = -0.1
        self.env.set_setpoints(action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self):
        """render."""
        assert (
            self.enable_render
        ), "Please set `render_mode='human' to use this function."

        camera_parameters = self.env.getDebugVisualizerCamera()

        camera_resolution = (camera_parameters[0], camera_parameters[1])

        _, _, rgbaImg, _, _ = self.env.getCameraImage(
            width=camera_parameters[0],
            height=camera_parameters[1],
            viewMatrix=camera_parameters[2],
            projectionMatrix=camera_parameters[3],
        )

        return np.array(rgbaImg).reshape(camera_resolution[1], camera_resolution[0], -1)
