import os
import random as rd

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.core import Aviary


class FixedwingWaypointsEnv(gym.Env):
    """Wrapper for PyFlyt fixed wing gymnasium environment"""

    metadata = {"render_modes": ["human", "gif"], "render_fps": 30}

    def __init__(self, render_mode: None | str = None):

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        # TODO: adjust observations to fit into this space
        # self.observation_space = spaces.Dict(
        #     {
        #         "attitude": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
        #         "target_deltas": spaces.Sequence(
        #             space=spaces.Box(
        #                 low=-2 * flight_dome_size,
        #                 high=2 * flight_dome_size,
        #                 shape=(4,) if use_yaw_targets else (3,),
        #                 dtype=np.float64,
        #             )
        #         ),
        #     }
        # )

        # Define action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Target visual file
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, f"../models/target.urdf")

        # Additional parameters (Could be updated to be inside __init__())
        self.start_pos = np.array([[0, 0, 10]])
        self.start_orn = np.array([[0, 0, 0]])
        self.goal_reach_distance = 2
        self.time_limit = 30  # Seconds
        self.agent_Hz = 30

        self.time_step_dur = 1 / self.agent_Hz
        # 120Hz is aviary step frequency
        self.max_steps = int(self.time_limit * 120)

        # Initialise render
        if render_mode == "human":
            self.enable_render = True
            self.use_camera = True
        elif render_mode == "gif":
            self.enable_render = False
            self.use_camera = True
        else:
            self.enable_render = False
            self.use_camera = False

        # Initialise PyFlyt env
        self.env = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            use_camera=self.use_camera,
            use_gimbal=False,
            render=self.enable_render,
            worldScale=20.0,
        )

        # Reset Aviary env
        self.env.set_mode(1)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset()
        self.done = False
        self.terminate = False
        self.truncate = False
        # TODO: fix these info to match those inside quadx_base_env, ie: use snake case
        self.info = {"Collide": False, "TargetReached": False, "NumTargetsReached": 0}
        self.curr_target_idx = 0
        self.collide = False
        self.target_reached = False
        self.all_targets_reached = False
        self.step_count = 0
        self.last_dist = 0
        self.last_error_from_UAV = [0, 0, 0]

        # Reset targets to follow
        x_arr = [rd.uniform(-50, 50) for i in range(1, 6)]
        y_arr = [rd.uniform(-50, 50) for i in range(1, 6)]
        z_arr = [rd.uniform(5, 50) for i in range(1, 6)]

        # Final 2 dummy targets not meant to be collected (Not rendered)
        self.num_targets = len(x_arr) - 2
        self.targets = []
        for i in range(len(x_arr)):
            self.targets.append([x_arr[i], y_arr[i], z_arr[i]])

        # if we are rendering, load in the targets
        if self.use_camera:
            self.target_visual = []
            for target in self.targets[0 : self.num_targets]:
                self.target_visual.append(
                    self.env.loadURDF(
                        self.targ_obj_dir, basePosition=target, useFixedBase=True
                    )
                )

            for i, visual in enumerate(self.target_visual):
                if i == 0:
                    # Recolour current target to Red
                    p.changeVisualShape(
                        visual,
                        linkIndex=-1,
                        rgbaColor=(1, 0, 0, 1),
                    )
                else:
                    p.changeVisualShape(
                        visual,
                        linkIndex=-1,
                        rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                    )

        return self.compute_state(), self.info

    def step(self, action):
        self.env.set_setpoints(np.array([action]))
        for i in range(4):
            self.env.step()
            self.step_count += 1
        self.obs = self.compute_state()
        self.rwd = self.compute_reward()
        self.terminate, self.truncate = self.compute_done()
        return self.obs, self.rwd, self.terminate, self.truncate, self.info

    def compute_state(self):
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - distance to target (scalar)
        - error_from_UAV (vector of 3 values)
        - next_targets (vector of 6 values)
        """
        # TODO: add current action to observation, this is usually after `lin_pos`, this allows the learning agent to have a notion of history
        raw_state = self.env.states[0]

        # State breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        quarternion = p.getQuaternionFromEuler(ang_pos)
        rotation = np.array(p.getMatrixFromQuaternion(quarternion)).reshape(3, 3).T

        # Compute distance to target
        error = self.targets[self.curr_target_idx] - lin_pos  # World frame
        self.error_from_UAV = np.matmul(rotation, error)  # UAV Body frame
        self.error_from_UAV_vel = (
            self.last_error_from_UAV - self.error_from_UAV
        ) / self.time_step_dur

        # Linear distance between UAV and target
        self.dist = np.linalg.norm(error)
        # Linear velocity between UAV and target
        self.vel_to_target = (self.last_dist - self.dist) / self.time_step_dur

        self.last_error_from_UAV = self.error_from_UAV.copy()
        self.last_dist = self.dist.copy()

        # Extract positions of next 2 targets
        next_targets = [
            *self.targets[self.curr_target_idx],
            *self.targets[self.curr_target_idx + 1],
        ]

        observation = np.array(
            [
                *ang_vel,
                *ang_pos,
                *lin_vel,
                *lin_pos,
                self.dist,
                self.vel_to_target,
                *self.error_from_UAV_vel,
                *self.error_from_UAV,
                *next_targets,
            ],
            dtype=np.float32,
        )

        return observation

    def compute_reward(self):
        """
        Returns reward for this step
            1 - Target not reached: -0.1
            2 - Target reached: 100
            3 - Collision: -100
        """
        self.info["Collide"] = False
        self.info["TargetReached"] = False

        self.collide = False
        self.target_reached = False
        self.all_targets_reached = False

        # Detect collision
        if np.any(self.env.getContactPoints(self.env.drones[0].Id)):
            self.info["Collide"] = True
            self.collide = True
            reward = -10

        # Check if target reached
        elif self.dist < self.goal_reach_distance:
            self.info["TargetReached"] = True
            # If target reached, advance to next set of targets, and end episode if all targets reached
            self.target_reached = True
            self.curr_target_idx += 1
            self.info["NumTargetsReached"] = self.curr_target_idx
            self.compute_state()
            self.recolour()
            if self.curr_target_idx == self.num_targets:
                self.all_targets_reached = True
                reward = 100
            else:
                reward = 10
        # Give -0.1 reward + velocity to target bonus if nothing happens
        else:

            reward = (0.5 * self.vel_to_target) * (self.vel_to_target > 0.0)

        reward = (1 * self.vel_to_target) * (self.vel_to_target > 0.0)
        return reward

    def compute_done(self):
        """
        Determines status of the UAV:
            4 - Collision (terminate env)
            5 - All targets collected (terminate env)
            3 - Exceed time window (truncate env)
            2 - Reach target (continue env)
            1 - Flying (continue env)
        """

        if self.collide:
            self.terminate = self.terminate or True
        elif self.all_targets_reached:
            self.truncate = self.truncate or True
        elif self.step_count >= self.max_steps:
            self.truncate = self.truncate or True

        return self.terminate, self.truncate

    def recolour(self):
        # delete the reached target and recolour the others
        if self.use_camera and len(self.target_visual) > 0:
            p.removeBody(self.target_visual[0])
            self.target_visual = self.target_visual[1:]

            # recolour
            for i, visual in enumerate(self.target_visual):
                if i == 0:
                    # Recolour current target to Red
                    p.changeVisualShape(
                        visual,
                        linkIndex=-1,
                        rgbaColor=(1, 0, 0, 1),
                    )
                else:
                    p.changeVisualShape(
                        visual,
                        linkIndex=-1,
                        rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                    )

    def render(self):
        # Take picture
        img = self.env.drones[0].capture_image()

        return img
