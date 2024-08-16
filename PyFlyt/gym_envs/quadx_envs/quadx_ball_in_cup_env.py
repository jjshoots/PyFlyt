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
    The target is a set of `[x, y, z, (optional) yaw]` waypoints in space.

    Args:
    ----
        sparse_reward (bool): whether to use sparse rewards or not.
        num_targets (int): number of waypoints in the environment.
        use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
        goal_reach_distance (float): distance to the waypoints for it to be considered reached.
        goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
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
            num_targets (int): number of waypoints in the environment.
            use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
            goal_reach_distance (float): distance to the waypoints for it to be considered reached.
            goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
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

        self.goal_reach_distance = goal_reach_distance

        # Define observation space
        self.observation_space = spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(3 + self.combined_space.shape[0],),
                        dtype=np.float64,
                    )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward
        
        self.ball_was_above_last = False
        self.ball_above_drone = False


    def add_ball_and_string(self) -> None:
        # spawn in the pendulum
        file_dir = os.path.dirname(os.path.realpath(__file__))
        targ_obj_dir = os.path.join(file_dir, "./../../models/ball_and_string.urdf")
        self.pendulum_id = self.env.loadURDF(
            targ_obj_dir,
            basePosition=self.env.state(0)[-1] - np.array([0,0,0.3]),
            baseOrientation=np.array([(np.random.rand()*2)-1., (np.random.rand()*2)-1., (np.random.rand()*2)-1., 1.0]),
            useFixedBase=False,
            globalScaling=1.0,
        )

        # register all new bodies with Aviary
        self.env.register_all_new_bodies()

        # create a constraint between the base of the pendulum object and the drone
        self.env.createConstraint(
            self.env.drones[0].Id,
            -1,
            self.pendulum_id,
            0,
            self.env.JOINT_POINT2POINT,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
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
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
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
        aux_state = super().compute_auxiliary()

        # combine everything
        new_state: dict[Literal["attitude", "ball"], np.ndarray] = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
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
            new_state["attitude"] = np.concatenate(
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

        new_state["ball"] = self.get_ball_state()

        self.state: dict[Literal["attitude", "ball"], np.ndarray] = new_state

        # concat the attitude and ball state
        self.state: np.array = np.concatenate(
            [new_state["attitude"], new_state["ball"]], axis=0
        )

    def get_ball_state(self) -> np.ndarray:
        """Get the state of the ball in the environment."""
        ball_pos = self.env.getBasePositionAndOrientation(self.pendulum_id)[0]
        drone_pos = self.env.state(0)[-1]
        self.ball_rel_to_drone = ball_pos - drone_pos
        self.ball_drone_dist = np.linalg.norm(self.ball_rel_to_drone)
        self.ball_was_above_last = self.ball_above_drone
        self.ball_above_drone = self.ball_rel_to_drone[2] > 0.
        return np.array(self.ball_rel_to_drone)

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, trunction, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            # small reward [-1, 0] for staying close to origin
            self.reward -= 1 *(np.linalg.norm(self.start_pos - self.env.state(0)[-1]) / self.flight_dome_size)
            if self.ball_above_drone:
                self.reward += -1 * np.log(self.ball_drone_dist)
            #else:
                # small penalty proportional to the distance
                #self.reward += np.log(1. - np.abs(self.ball_rel_to_drone[2]))
        
            if self.ball_was_above_last and not self.ball_above_drone:
                self.termination = True

        # target reached
        if self.ball_drone_dist < self.goal_reach_distance:
            self.reward = 100.0

            # update infos and dones
            self.truncation = True
            self.info["env_complete"] = True



if __name__ == "__main__":
    env = QuadXBallInCupEnv(render_mode="human")
    env.reset()
    for i in range(1000):
        obs, rew, done, _, _ = env.step(np.array([0.0, 0, 0, 0.53]))
        env.render()
        if done:
            break
    env.close()