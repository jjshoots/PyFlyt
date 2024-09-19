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
    The goal is to swing up a suspended ball onto the drone, and then bring it to the starting position.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        goal_reach_distance (float): minimum distance from the ending position that the UAV must reach for success.
        goal_reach_velocity (float): maximum velocity that the UAV must maintain at the ending position for success.
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
        goal_reach_distance: float = 1.0,
        goal_reach_velocity: float = 1.0,
        flight_mode: int = 0,
        flight_dome_size: float = 30.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            goal_reach_distance (float): minimum distance from the ending position that the UAV must reach for success.
            goal_reach_velocity (float): maximum velocity that the UAV must maintain at the ending position for success.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 4.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

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
        self.goal_reach_velocity = goal_reach_velocity
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.pendulum_filepath = os.path.join(
            file_dir, "./../../models/ball_and_string.urdf"
        )
        self.cup_filepath = os.path.join(file_dir, "./../../models/cup.urdf")

    def add_ball_and_string_and_cup(self) -> None:
        """Spawns in the ball and string and cup."""
        # spawn in the cup
        self.cup_id = self.env.loadURDF(
            self.cup_filepath,
            basePosition=self.env.state(0)[-1] + np.array([0.0, 0.0, 0.035]),
            baseOrientation=np.array([0.0, 0.0, 0.0, 1.0]),
            useFixedBase=False,
            globalScaling=1.0,
        )
        # spawn in the pendulum
        self.pendulum_id = self.env.loadURDF(
            self.pendulum_filepath,
            # permanently attach the ball to the drone
            basePosition=self.env.state(0)[-1] - np.array([0, 0, 0.5]),
            # randomly angle the ball
            baseOrientation=np.array(
                [
                    (self.np_random.random() * 2.0) - 1.0,
                    (self.np_random.random() * 2.0) - 1.0,
                    (self.np_random.random() * 2.0) - 1.0,
                    1.0,
                ]
            ),
            useFixedBase=False,
            globalScaling=1.0,
        )

        # register all new bodies with Aviary
        self.env.register_all_new_bodies()

        # create a constraint between the base of the cup object and the drone
        _zeros = np.zeros((3,), dtype=np.float32)
        self.env.createConstraint(
            parentBodyUniqueId=self.env.drones[0].Id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.cup_id,
            childLinkIndex=-1,
            jointType=self.env.JOINT_FIXED,
            jointAxis=_zeros,
            parentFramePosition=_zeros,
            childFramePosition=np.array([0.0, 0.0, -0.035]),
        )
        # create a constraint between the base of the pendulum object and the drone
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

        # add the ball
        self.add_ball_and_string_and_cup()

        # stateful params with history for the ball
        self.drone_state_error = np.zeros((4,), dtype=np.float32)
        self.drone_state_prev_error = np.zeros((4,), dtype=np.float32)

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
        self.ball_lin_pos, _ = self.env.getBasePositionAndOrientation(self.pendulum_id)
        self.ball_lin_vel, _ = self.env.getBaseVelocity(self.pendulum_id)

        # compute ball state relative to self
        self.ball_rel_lin_pos = np.matmul(rotation, self.ball_lin_pos - lin_pos)
        self.ball_rel_lin_vel = np.matmul(rotation, self.ball_lin_vel)

        # drone_state: [4, 3]
        # drone_state_error: [4,]
        self.drone_state_prev_error = self.drone_state_error.copy()
        self.drone_state_error = self.env.state(0).copy()
        self.drone_state_error[-1] -= np.array([0.0, 0.0, 1.0])
        self.drone_state_error = np.linalg.norm(self.drone_state_error, axis=-1) ** 2

        # concat the attitude and ball state
        self.state = np.concatenate(
            [
                attitude,
                self.ball_rel_lin_pos,
                self.ball_rel_lin_vel,
            ],
            axis=0,
        )

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        # compute some parameters of the ball
        # lin_pos: [3,], height: [1,], abs_dist: [1,]
        ball_rel_lin_pos = self.ball_lin_pos - self.env.state(0)[-1]
        ball_rel_height = ball_rel_lin_pos[2]
        ball_rel_abs_dist = np.linalg.norm(ball_rel_lin_pos)

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            # reward for staying alive
            self.reward += 0.4

            # penalty for aggressive maneuvres, and try to stay close to origin
            self.reward -= 0.01 * np.sum(self.drone_state_error)

            if ball_rel_height > 0.0:
                # reward for bringing the ball close to self
                self.reward -= 4.0 * np.log(0.45 * ball_rel_abs_dist + 1e-2)
            else:
                # penalty when ball below drone
                self.reward += ball_rel_height

        # when the ball hit the drone, either success or failure
        if self.env.contact_array[self.pendulum_id, self.env.drones[0].Id]:
            # hitting self is bad
            if ball_rel_height < 0.0:
                self.reward = -500.0
                self.termination = True
                self.info["self_collision"] = True
                return

            # if the ball is above us, we need to check the success criteria
            if (
                self.drone_state_error[-1] < self.goal_reach_distance
                and self.drone_state_error[-2] < self.goal_reach_velocity
            ):
                # hack: use truncation here, otherwise we need a HUGE reward to end
                self.reward += 1000.0
                self.truncation = True
                self.info["env_complete"] = True
                return

            # if it's up, but we're not at the winning criteria yet,
            # reward for approaching goal position
            if not self.sparse_reward:
                self.reward += 50.0 * (
                    self.drone_state_prev_error[-1] - self.drone_state_error[-1]
                )
                self.reward += 10.0 / (self.drone_state_error[-1] + 0.1)
