import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import time
import random as rd
from PyFlyt.core import Aviary, loadOBJ, obj_collision, obj_visual
import os
from wingman import cpuize, gpuize


class FWTargets(gym.Env):
    """Wrapper for PyFlyt fixed wing gymnasium environment"""
    metadata = {"render_modes": ["human", "gif"], "render_fps": 30}

    def __init__(
        self,
        render_mode: None | str = None
    ):

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        # Define action space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32)

        # Init targets
        x_arr = [rd.uniform(-50, 50) for i in range(1, 4)]
        y_arr = [rd.uniform(-50, 50) for i in range(1, 4)]
        z_arr = [rd.uniform(5, 50) for i in range(1, 4)]

        self.num_targets = len(x_arr)
        self.targets = []
        for i in range(len(x_arr)):
            self.targets.append([x_arr[i], y_arr[i], z_arr[i]])

        # Clone extra elements at the end of the list of targets
        for i in range(2):
            self.targets.append([0, 0, 0])
        # Target visual file
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, f"../models/target.urdf")

        # Additional parameters (Could be updated to be inside __init__())
        self.start_pos = np.array([[0, 0, 10]])
        self.start_orn = np.array([[0, 0, 0]])
        self.start_vel = np.array([[0, 20, 0]])
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
        self.env = Aviary(start_pos=self.start_pos, start_orn=self.start_orn,
                          start_vel=self.start_vel, use_camera=self.use_camera, use_gimbal=False, render=self.enable_render)

        # Reset Aviary env
        self.env.set_mode(1)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset()
        self.done = False
        self.terminate = False
        self.truncate = False
        self.info = {"Collide": False,
                     "TargetReached": False,
                     "NumTargetsReached": 0}
        self.curr_target_idx = 0
        self.collide = False
        self.target_reached = False
        self.all_targets_reached = False
        self.step_count = 0
        self.last_dist = 0
        self.last_error_from_UAV = [0, 0, 0]

        # Reset targets to follow
        x_arr = [rd.uniform(-50, 50) for i in range(1, 4)]
        y_arr = [rd.uniform(-50, 50) for i in range(1, 4)]
        z_arr = [rd.uniform(5, 50) for i in range(1, 4)]

        self.num_targets = len(x_arr)
        self.targets = []
        for i in range(len(x_arr)):
            self.targets.append([x_arr[i], y_arr[i], z_arr[i]])

        # Clone extra elements at the end of the list of targets
        for i in range(2):
            self.targets.append([0, 0, 0])

        # if we are rendering, load in the targets
        if self.use_camera:
            self.target_visual = []
            for target in self.targets:
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

        return self._compute_obs(), self.info

    def step(self, action):
        self.env.set_setpoints([action])
        for i in range(4):
            self.env.step()
            self.step_count += 1
        self.obs = self._compute_obs()
        self.rwd = self._compute_rwd()
        self.terminate, self.truncate = self._compute_done()
        return self.obs, self.rwd, self.terminate, self.truncate, self.info

    def _compute_obs(self):
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
        raw_state = self.env.states[0]

        # State breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        quarternion = p.getQuaternionFromEuler(ang_pos)
        rotation = np.array(
            p.getMatrixFromQuaternion(quarternion)).reshape(3, 3).T

        # Compute distance to target
        error = self.targets[self.curr_target_idx] - lin_pos  # World frame
        self.error_from_UAV = np.matmul(rotation, error)  # UAV Body frame
        self.error_from_UAV_vel = (
            self.last_error_from_UAV - self.error_from_UAV) / self.time_step_dur

        # Linear distance between UAV and target
        self.dist = np.linalg.norm(error)
        # Linear velocity between UAV and target
        self.vel_to_target = (self.last_dist - self.dist) / self.time_step_dur

        self.last_error_from_UAV = self.error_from_UAV.copy()
        self.last_dist = self.dist.copy()

        # Extract positions of next 2 targets
        next_targets = [*self.targets[self.curr_target_idx],
                        *self.targets[self.curr_target_idx+1]]

        observation = np.array([*ang_vel, *ang_pos, *lin_vel, *lin_pos,
                               self.dist, self.vel_to_target, *self.error_from_UAV_vel, *self.error_from_UAV, *next_targets], dtype=np.float32)

        return observation

    def _compute_rwd(self):
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
            self._compute_obs()
            self._recolour()
            if self.curr_target_idx == self.num_targets:
                self.all_targets_reached = True
                reward = 100
            else:
                reward = 10
        # Give -0.1 reward + velocity to target bonus if nothing happens 
        else:

            reward = -0.01 + (0.01 * self.vel_to_target) * (self.vel_to_target > 0.0)

        return reward

    def _compute_done(self):
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

    def _recolour(self):
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
