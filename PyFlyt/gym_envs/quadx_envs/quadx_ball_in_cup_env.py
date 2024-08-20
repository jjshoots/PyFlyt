"""QuadX Ball-in-Cup Environment."""
from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv


class QuadXBallInCupEnv(QuadXBaseEnv):
    """QuadX Ball-in-Cup Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The goal is to swing up the pendulum to be above the UAV.

    Args:
    ----
        sparse_reward (bool): whether to use sparse rewards or not.
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
        goal_reach_distance: float = 0.3,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
        ----
            sparse_reward (bool): whether to use sparse rewards or not.
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
            start_pos=np.array([[0.0, 0.0, 2.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        self.ball_was_above = False
        self.ball_is_above = False

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + 6,),
            dtype=np.float64,
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward
        self.goal_reach_distance = goal_reach_distance
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.pendulum_filepath = os.path.join(
            file_dir, "./../../models/ball_and_string.urdf"
        )

    def add_ball_and_string(self) -> None:
        """Spawns in the ball and string."""
        # spawn in the pendulum
        self.pendulum_id = self.env.loadURDF(
            self.pendulum_filepath,
            # permanently attach the ball to the drone at a distance of 0.3
            basePosition=self.env.state(0)[-1] - np.array([0, 0, 0.3]),
            # randomly angle the ball
            baseOrientation=np.array(
                [
                    # (self.np_random.random() * 2.0) - 1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ),
            useFixedBase=False,
            globalScaling=1.0,
        )

        # register all new bodies with Aviary
        self.env.register_all_new_bodies()

        # create a constraint between the base of the pendulum object and the drone
        _zeros = np.zeros((3,), dtype=np.float32)
        self.env.createConstraint(
            parentBodyUniqueId=self.env.drones[0].Id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.pendulum_id,
            childLinkIndex=0,
            jointType=self.env.JOINT_POINT2POINT,
            jointAxis=_zeros,
            parentFramePosition=_zeros,
            childFramePosition=_zeros,
        )

        # disable motor on the prismatic joint
        self.env.setJointMotorControl2(
            self.pendulum_id,
            0,
            self.env.VELOCITY_CONTROL,
            force=0,
        )

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict]:
        """Resets the environment.

        Args:
        ----
            seed: seed to pass to the base environment.
            options: None

        """
        options = dict(
            drone_model="primitive_drone",
            use_camera=True,
            camera_fps=30,
            camera_angle_degrees=-90,
            camera_FOV_degrees=140,
        )
        super().begin_reset(seed, drone_options=options)
        self.add_ball_and_string()
        super().end_reset()
        self.compute_state()

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the current vehicle attitude as well as the state of the ball.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        - ball state (vector of 3 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        rotation = (
            np.array(self.env.getMatrixFromQuaternion(quaternion)).reshape(3, 3).T
        )
        aux_state = super().compute_auxiliary()

        # combine everything
        if self.angle_representation == 0:
            attitude = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            attitude = np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError

        # compute ball state
        ball_lin_pos, _ = self.env.getBasePositionAndOrientation(self.pendulum_id)
        ball_lin_vel, _ = self.env.getBaseVelocity(self.pendulum_id)

        # compute ball state relative to self
        ball_rel_lin_pos = np.matmul(rotation, ball_lin_pos - lin_pos)
        ball_rel_lin_vel = np.matmul(rotation, ball_lin_vel)

        # compute some stateful parameters
        self.ball_is_above = ball_lin_pos[2] > 0.0
        self.ball_upwards_vel = ball_rel_lin_vel[2]
        self.ball_drone_abs_dist = np.linalg.norm(ball_lin_pos)
        self.ball_drone_hor_dist = np.linalg.norm(ball_lin_pos[:2])

        # concat the attitude and ball state
        self.state = np.concatenate(
            [
                attitude,
                ball_rel_lin_pos,
                ball_rel_lin_vel,
            ],
            axis=0,
        )

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, trunction, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            # small reward [-0.1, 0.1] for staying close to origin
            self.reward += 0.1 * (
                (
                    np.linalg.norm(self.start_pos - self.env.state(0)[-1])
                    / self.flight_dome_size
                )
                + 0.5
            )

            if self.ball_is_above:
                # reward [0.38, 2](before scale) for bringing the ball close to self
                self.reward -= 2.0 * np.log(self.ball_drone_abs_dist + 1e-2)
            else:
                # reward for ball upwards velocity
                # reward [0, 0.4](before scale) for ball having sideways component
                # combined, these should encourage swinging behaviour
                self.reward += 0.5 * self.ball_upwards_vel
                self.reward += 0.5 * self.ball_drone_hor_dist

        # success
        if self.ball_is_above and (self.ball_drone_abs_dist < self.goal_reach_distance):
            self.reward = 300.0

            # update infos and dones
            self.termination = True
            self.info["env_complete"] = True
