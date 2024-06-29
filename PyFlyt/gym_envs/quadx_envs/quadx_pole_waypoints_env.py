"""QuadX Waypoints Environment."""
from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class QuadXPoleWaypointsEnv(QuadXBaseEnv):
    """QuadX Pole Waypoints Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to get the tip of a pole to a set of `[x, y, z]` waypoints in space.

    Args:
    ----
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
        flight_mode: int = 0,
        flight_dome_size: float = 10.0,
        max_duration_seconds: float = 60.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
        ----
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
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
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

        # the pole urdf
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.pole_obj_dir = os.path.join(file_dir, "../../models/pole.urdf")

        # modify the state to take into account the pole's state
        pole_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.attitude_space.shape[0],),
            dtype=np.float64,
        )
        combined_plus_pole_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + pole_space.shape[0],),
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
        ----
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(
            seed, options, drone_options={"drone_model": "primitive_drone"}
        )

        # spawn in a pole and make it have enough friction
        self.poleId = self.env.loadURDF(
            self.pole_obj_dir,
            basePosition=np.array([0.0, 0.0, 2.1]),
            useFixedBase=False,
        )
        self.env.changeDynamics(
            self.poleId,
            linkIndex=1,
            anisotropicFriction=1e9,
            lateralFriction=1e9,
        )

        self.waypoints.reset(self.env, self.np_random)
        self.info["num_targets_reached"] = 0
        self.distance_to_immediate = np.inf
        self.pole_uprightness = 0.0
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
        - "target_deltas" (Sequence)
        ----- list of body_frame distances to target (vector of 3/4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        rotation = (
            np.array(self.env.getMatrixFromQuaternion(quaternion)).reshape(3, 3).T
        )
        aux_state = super().compute_auxiliary()

        # compute the attitude of the pole in global coords
        pole_lin_pos, pole_quaternion = self.env.getBasePositionAndOrientation(
            self.poleId
        )
        pole_lin_vel, pole_ang_vel = self.env.getBaseVelocity(self.poleId)
        pole_ang_pos = self.env.getEulerFromQuaternion(pole_quaternion)
        self.pole_uprightness = np.linalg.norm(pole_ang_pos[:2])

        # express everything relative to self
        # positions are global, hence the subtract
        rel_pole_lin_pos = np.matmul(rotation, (pole_lin_pos - lin_pos))
        rel_pole_ang_pos = np.matmul(rotation, (pole_ang_pos - ang_pos))
        # velocities of self are already rotated, pole isn't
        rel_pole_lin_vel = np.matmul(rotation, pole_lin_vel) - lin_vel
        rel_pole_ang_vel = np.matmul(rotation, pole_ang_vel) - ang_vel

        # regenerate the quaternion
        rel_pole_quaternion = self.env.getQuaternionFromEuler(rel_pole_ang_pos)

        # combine everything
        new_state: dict[Literal["attitude", "target_deltas"], np.ndarray] = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.array(
                [
                    *ang_vel,
                    *ang_pos,
                    *lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    *rel_pole_ang_vel,
                    *rel_pole_ang_pos,
                    *rel_pole_lin_vel,
                    *rel_pole_lin_pos,
                ]
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.array(
                [
                    *ang_vel,
                    *quaternion,
                    *lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    *rel_pole_ang_vel,
                    *rel_pole_quaternion,
                    *rel_pole_lin_vel,
                    *rel_pole_lin_pos,
                ]
            )

        new_state["target_deltas"] = self.waypoints.distance_to_target(
            pole_ang_pos, pole_lin_pos, pole_quaternion
        )
        self.distance_to_immediate = float(
            np.linalg.norm(new_state["target_deltas"][0])
        )

        self.state: dict[Literal["attitude", "target_deltas"], np.ndarray] = new_state

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, trunction, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            self.reward += max(3.0 * self.waypoints.progress_to_target(), 0.0)
            self.reward += 0.1 / self.distance_to_immediate
            self.reward -= 0.1 * self.pole_uprightness
            print(self.pole_uprightness)

        # target reached
        if self.waypoints.target_reached():
            self.reward = 100.0

            # advance the targets
            self.waypoints.advance_targets()

            # update infos and dones
            self.truncation |= self.waypoints.all_targets_reached()
            self.info["env_complete"] = self.waypoints.all_targets_reached()
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached()