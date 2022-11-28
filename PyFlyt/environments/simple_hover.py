from __future__ import annotations
import math

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.core.aviary import Aviary


class SimpleHoverEnv(gymnasium.Env):
    """
    Simple Hover Environment

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is to not crash for the longest time possible

    Reward:
        -100 for collisions or out of bounds,
        -0.1 otherwise
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 300,
        angle_representation: str = "quaternion",
        flight_dome_size: float = 3.0,
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            max_steps (int): max_steps of the environment
            angle_representation (str): can be "euler" or "quaternion"
            flight_dome_size (float): size of the allowable flying area
            render_mode (None | str): can be "human" or None
        """

        if 120 % agent_hz != 0:
            lowest = int(120 / int(120 / agent_hz))
            highest = int(120 / (int(120 / agent_hz) + 1))
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
        # observation size increases by 1 for quaternion
        if angle_representation == "euler":
            obs_shape = 12
        elif angle_representation == "quaternion":
            obs_shape = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        high = np.array([2 * math.pi, 2 * math.pi, 2 * math.pi, 1.0])
        low = np.array([-2 * math.pi, -2 * math.pi, -2 * math.pi, 0.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        """ ENVIRONMENT CONSTANTS """
        self.cycle_steps = int(120 / agent_hz)
        self.flight_dome_size = flight_dome_size
        self.max_steps = max_steps
        self.ang_rep = 0
        if angle_representation == "euler":
            self.ang_rep = 0
        elif angle_representation == "quaternion":
            self.ang_rep = 1

    def reset(self, seed=None, options=None):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options:
        """
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False

        # init env
        self.env = Aviary(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.enable_render,
            seed=seed,
        )

        # set flight mode
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        self.compute_state()

        return self.state, self.info

    def compute_state(self):
        """state.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        """
        raw_state = self.env.states[0]

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # combine everything
        if self.ang_rep == 0:
            self.state = np.array([*ang_vel, *ang_pos, *lin_vel, *lin_pos])
        elif self.ang_rep == 1:
            # quarternion angles
            q_ang_pos = p.getQuaternionFromEuler(ang_pos)

            self.state = np.array([*ang_vel, *q_ang_pos, *lin_vel, *lin_pos])

    def compute_term_trunc_reward(self):
        """compute_term_trunc."""
        self.reward += -0.1

        # if we've already ended, just exit
        if self.termination or self.truncation:
            return

        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation = self.truncation or True

        # exceed flight dome
        if np.linalg.norm(self.env.states[0][-1]) > self.flight_dome_size:
            self.reward += -100.0
            self.info["out_of_bounds"] = True
            self.termination = self.termination or True

        # collision
        if np.any(self.env.collision_array):
            self.reward += -100.0
            self.info["collision"] = True
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
        self.reward = 0
        self.env.set_setpoints(action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.cycle_steps):
            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self):
        """render."""
        raise AssertionError(
            "This function is not meant to be called. Apply `render_mode='human'` on environment creation."
        )
