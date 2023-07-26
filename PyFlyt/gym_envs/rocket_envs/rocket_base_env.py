"""Base PyFlyt Environment for the Rocket model using the Gymnasim API."""
from __future__ import annotations

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.core.aviary import Aviary
from PyFlyt.utils import check_numpy


class RocketBaseEnv(gymnasium.Env):
    """Base PyFlyt Environment for the Rocket model using the Gymnasim API."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        start_pos: np.ndarray = np.array([[0.0, 0.0, 10.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        ceiling: float = np.inf,
        max_displacement: float = np.inf,
        max_duration_seconds: float = 60.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 30,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            drone_type (str): drone_type
            drone_model (str): drone_model
            ceiling (float): ceiling
            max_displacement (float): max_displacement
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
            render_resolution (tuple[int, int]): render_resolution
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
            ), f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
        self.render_mode = render_mode
        self.render_resolution = render_resolution

        """GYMNASIUM STUFF"""
        # attitude size increases by 1 for quaternion
        if angle_representation == "euler":
            attitude_shape = 12
        elif angle_representation == "quaternion":
            attitude_shape = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.attitude_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(attitude_shape,), dtype=np.float64
        )
        self.auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64
        )

        # force_x, force_y, roll, ignition, throttle, booster_gimbal_1, booster_gimbal_2
        finlet_setpoint_limit = 1.0
        throttle_limit = 1.0
        booster_gimbal_limit = 1.0
        high = np.array(
            [
                finlet_setpoint_limit,
                finlet_setpoint_limit,
                finlet_setpoint_limit,
                1.0,
                throttle_limit,
                booster_gimbal_limit,
                booster_gimbal_limit,
            ]
        )
        low = np.array(
            [
                -finlet_setpoint_limit,
                -finlet_setpoint_limit,
                -finlet_setpoint_limit,
                0.0,
                0.0,
                -booster_gimbal_limit,
                -booster_gimbal_limit,
            ]
        )
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # the whole implicit state space = attitude + previous action + auxiliary information
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                attitude_shape
                + self.action_space.shape[0]
                + self.auxiliary_space.shape[0],
            ),
            dtype=np.float64,
        )

        """ ENVIRONMENT CONSTANTS """
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.ceiling = ceiling
        self.max_displacement = max_displacement
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

    def reset(self, seed=None, options=dict()):
        """Resets the environment.

        Args:
            seed: int
            options: None
        """
        raise NotImplementedError

    def close(self):
        """Disconnects the internal Aviary."""
        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

    def begin_reset(self, seed=None, options=dict(), drone_options=dict()):
        """The first half of the reset function."""
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.state = None
        self.action = np.zeros((7,))
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["fatal_collision"] = False
        self.info["env_complete"] = False

        # override the spawn location if needed
        if "randomize_drop" in options and options["randomize_drop"]:
            spawn_range = self.max_displacement * 0.1
            start_xy = self.np_random.uniform(-spawn_range, spawn_range, size=(2,))
            start_z = self.np_random.uniform(self.ceiling * 0.8, self.ceiling * 0.9)
            self.start_pos = np.expand_dims(np.array([*start_xy, start_z]), axis=0)

            # random rotation + make kind of upright
            rotation = self.np_random.uniform(-0.3, 0.3, size=(3,))
            rotation = np.expand_dims(rotation, axis=0)
            self.start_orn = rotation

        # camera handling
        if "use_camera" not in drone_options:
            drone_options["use_camera"] = self.render_mode is not None
        else:
            drone_options["use_camera"] |= self.render_mode is not None

        # init env
        self.env = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="rocket",
            render=self.render_mode is not None,
            drone_options=drone_options,
            seed=seed,
        )

        # add the random velocities to our base
        start_ang_vel = np.array([0.0, 0.0, 0.0])
        start_lin_vel = np.array([0.0, 0.0, 0.0])

        # impart some random velocities if randomize
        if options.get("randoimize_drop", False):
            start_lin_vel += self.np_random.uniform(-5.0, 5.0, size=(3,))
            start_ang_vel += self.np_random.uniform(-0.5, 0.5, size=(3,))

        # speed up the drop if required
        if options.get("accelerate_drop", False):
            start_lin_vel += np.array([0.0, 0.0, -100.0])

        self.env.resetBaseVelocity(self.env.drones[0].Id, start_lin_vel, start_ang_vel)

        if self.render_mode is not None:
            self.camera_parameters = self.env.getDebugVisualizerCamera()

    def end_reset(self, seed=None, options=dict()):
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.env.register_all_new_bodies()

        # set flight mode
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        self.compute_state()

    def compute_state(self):
        """Computes the state of the Rocket."""
        raise NotImplementedError

    def compute_auxiliary(self):
        """This returns the auxiliary state form the drone."""
        return self.env.aux_state(0)

    def compute_attitude(self):
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        """
        raw_state = self.env.state(0)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        quarternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quarternion

    def compute_term_trunc_reward(self):
        """compute_term_trunc_reward."""
        raise NotImplementedError

    def compute_base_term_trunc_reward(
        self, collision_ignore_mask: np.ndarray | list[int] = []
    ):
        """compute_base_term_trunc_reward.

        Args:
            collision_ignore_mask (np.ndarray | list[int]): list of ids to ignore collisions between
        """
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation = self.truncation or True

        # mask collisions if any
        collision_array = self.env.contact_array.copy()
        for i, j in zip(collision_ignore_mask[1:], collision_ignore_mask[:-1]):
            collision_array[i, j] = False
            collision_array[j, i] = False

        # fatal collision or below ground
        if np.any(collision_array) or self.env.state(0)[-1, -1] < 0.0:
            # self.reward = -100.0
            self.info["fatal_collision"] = True
            self.termination |= True

        # exceed flight dome
        if (
            np.linalg.norm(self.env.state(0)[-1, :2]) > self.max_displacement
            or self.env.state(0)[-1, 2] > self.ceiling
        ):
            # self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination |= True

    def step(self, action: np.ndarray):
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info
        """
        # unsqueeze the action to be usable in aviary
        self.action = action.copy()

        # reset the reward and set the action
        self.reward = 0.0
        self.env.set_setpoint(0, action)

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
        check_numpy()
        assert (
            self.render_mode is not None
        ), "Please set `render_mode='human'` or `render_mode='rgb_array'` to use this function."

        _, _, rgbaImg, _, _ = self.env.getCameraImage(
            width=self.render_resolution[1],
            height=self.render_resolution[0],
            viewMatrix=self.env.drones[0].camera.view_mat,
            projectionMatrix=self.env.drones[0].camera.proj_mat,
        )

        rgbaImg = np.asarray(rgbaImg).reshape(
            self.render_resolution[0], self.render_resolution[1], -1
        )

        return rgbaImg
