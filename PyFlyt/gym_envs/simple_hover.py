from __future__ import annotations

import numpy as np

from .pyflyt_env import PyFlytEnv


class SimpleHoverEnv(PyFlytEnv):
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
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            flight_dome_size (float): size of the allowable flying area
            max_duration_seconds (float): maximum simulatiaon time of the environment
            angle_representation (str): can be "euler" or "quaternion"
            agent_hz (int): looprate of the agent to environment interaction
            render_mode (None | str): can be "human" or None
        """
        super().__init__(
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
        )

        """GYMNASIUM STUFF"""
        self.observation_space = self.attitude_space

        """ ENVIRONMENT CONSTANTS """
        self.flight_dome_size = flight_dome_size

    def reset(self, seed=None, options=None):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options:
        """
        super().begin_reset(seed, options)
        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self):
        """state.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()

        # combine everything
        if self.angle_representation == 0:
            self.state = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action]
            )
        elif self.angle_representation == 1:
            self.state = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action]
            )

    def compute_term_trunc_reward(self):
        """compute_term_trunc."""
        super().compute_base_term_trunc_reward()

        # exceed flight dome
        if np.linalg.norm(self.env.states[0][-1]) > self.flight_dome_size:
            self.reward += -100.0
            self.info["out_of_bounds"] = True
            self.termination = self.termination or True
