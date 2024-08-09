"""QuadX Pole Waypoints Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.pole_handler import PoleHandler
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class QuadXPoleWaypointsEnv(QuadXBaseEnv):
    """QuadX Pole Waypoints Environment.

    Actions are direct motor PWM commands because any underlying controller introduces too much control latency.
    The target is to get to a set of `[x, y, z]` waypoints in space without dropping the pole.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        num_targets (int): number of waypoints in the environment.
        goal_reach_distance (float): distance to the waypoints for it to be considered reached.
        flight_mode (int): the flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.

    """

    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 4,
        goal_reach_distance: float = 0.2,
        flight_mode: int = -1,
        flight_dome_size: float = 10.0,
        max_duration_seconds: float = 20.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            num_targets (int): number of waypoints in the environment.
            goal_reach_distance (float): distance to the waypoints for it to be considered reached.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode.
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define waypoints
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=False,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=np.inf,
            flight_dome_size=flight_dome_size,
            min_height=1.3,
            np_random=self.np_random,
        )

        # init the pole
        self.pole = PoleHandler()
        combined_plus_pole_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.combined_space.shape[0] + self.pole.observation_space.shape[0],
            ),
            dtype=np.float64,
        )

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "attitude": combined_plus_pole_space,
                "target_deltas": spaces.Sequence(
                    space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                    stack=True,
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(
            seed,
            options,
            drone_options={
                "drone_model": "primitive_drone",
                "camera_position_offset": np.array([-3.0, 0.0, 1.0]),
            },
        )

        # spawn in a pole
        self.pole.reset(p=self.env, start_location=np.array([0.0, 0.0, 1.55]))

        # init some other metadata
        self.waypoints.reset(self.env, self.np_random)
        self.info["num_targets_reached"] = 0

        super().end_reset()

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the observation as well as the distances to target.
        - "attitude" (Box)
        ----- ang_vel (vector of 3 values)
        ----- ang_pos (vector of 3/4 values)
        ----- lin_vel (vector of 3 values)
        ----- lin_pos (vector of 3 values)
        ----- previous_action (vector of 4 values)
        ----- auxiliary information (vector of 4 values)
        ----- 12 values for the pole's positions relative to self:
        ---------- top position XYZ
        ---------- bottom position XYZ
        ---------- top velocity XYZ
        ---------- bottom velocity XYZ
        - "target_deltas" (Sequence)
        ----- list of body_frame distances to target (vector of 3/4 values)
        """
        # compute attitude of self
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()
        rotation = (
            np.array(self.env.getMatrixFromQuaternion(quaternion)).reshape(3, 3).T
        )

        # compute the pole's states
        (
            pole_top_pos,
            pole_top_vel,
            pole_bot_pos,
            pole_bot_vel,
        ) = self.pole.compute_state(
            rotation=rotation,
            uav_lin_pos=lin_pos,
            uav_lin_vel=lin_vel,
        )

        # combine everything
        new_state: dict[Literal["attitude", "target_deltas"], np.ndarray] = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                    pole_top_pos,
                    pole_bot_pos,
                    pole_top_vel,
                    pole_bot_vel,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                    pole_top_pos,
                    pole_bot_pos,
                    pole_top_vel,
                    pole_bot_vel,
                ],
                axis=-1,
            )

        new_state["target_deltas"] = self.waypoints.distance_to_targets(
            ang_pos, lin_pos, quaternion
        )

        self.state: dict[Literal["attitude", "target_deltas"], np.ndarray] = new_state

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            self.reward += max(15.0 * self.waypoints.progress_to_next_target, 0.0)
            self.reward += 0.5 / self.waypoints.distance_to_next_target
            self.reward += 0.5 - self.pole.leaningness

        # target reached
        if self.waypoints.target_reached:
            self.reward = 300.0

            # advance the targets
            self.waypoints.advance_targets()

            # update infos and dones
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached
