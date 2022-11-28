from __future__ import annotations
import copy
import math
import os

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.core.aviary import Aviary


class AdvancedGatesEnv(gymnasium.Env):
    """
    Advanced Gates Env

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is a set of `[x, y, z, yaw]` targets in space

    Reward is -(distance from waypoint + angle error) for each timestep,
    and -100.0 for hitting the ground.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 5000,
        angle_representation: str = "quaternion",
        num_targets: int = 5,
        goal_reach_distance: float = 0.21,
        max_gate_angles: list[float] = [0.0, 0.3, 1.0],
        min_gate_distance: float = 1.0,
        max_gate_distance: float = 4.0,
        camera_frame_size: tuple[int, int] = (128, 128),
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            max_steps (int): max_steps
            angle_representation (str): angle_representation
            num_targets (int): num_targets
            goal_reach_distance (float): goal_reach_distance
            max_gate_angles (list[float]): max_gate_angles
            min_gate_distance (float): min_gate_distance
            max_gate_distance (float): max_gate_distance
            camera_frame_size (tuple[int, int]): camera_frame_size
            render_mode (None | str): render_mode
        """
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
            state_size = 12
        elif angle_representation == "quaternion":
            state_size = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.observation_space = spaces.Dict(
            {
                "attitude": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float64
                ),
                "rgba_cam": spaces.Box(
                    low=0.0, high=255.0, shape=(4, *camera_frame_size), dtype=np.float64
                ),
            }
        )

        high = np.array([2 * math.pi, 2 * math.pi, 2 * math.pi, 1.0])
        low = np.array([-2 * math.pi, -2 * math.pi, -2 * math.pi, 0.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        """ ENVIRONMENT CONSTANTS """
        self.min_gate_distance = min_gate_distance
        self.max_gate_distance = max_gate_distance
        self.max_gate_angles = np.array([max_gate_angles])
        self.max_steps = max_steps
        self.goal_reach_distance = goal_reach_distance
        self.ang_rep = 0
        if angle_representation == "euler":
            self.ang_rep = 0
        elif angle_representation == "quaternion":
            self.ang_rep = 1
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.gate_obj_dir = os.path.join(file_dir, f"../models/race_gate.urdf")
        self.camera_frame_size = camera_frame_size
        self.num_targets = num_targets

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

        # reset step count
        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        self.dis_error = -100.0

        # init env
        self.env = Aviary(
            start_pos=np.array([[0.0, 0.0, 3.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.enable_render,
            use_camera=True,
            camera_frame_size=self.camera_frame_size,
            seed=seed,
        )

        # generate gates
        self.gates = []
        self.targets = []
        self.generate_gates()

        # set flight mode
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        return self.state, self.info

    def generate_gates(self):
        """generate_gates."""
        # sample a bunch of distances for gate distances
        distances = self.np_random.uniform(
            self.min_gate_distance, self.max_gate_distance, size=(self.num_targets,)
        )
        angles = self.np_random.uniform(-1.0, 1.0, size=(self.num_targets, 3))
        angles *= self.max_gate_angles

        # starting position and angle
        gate_pos = np.array([0.0, 0.0, 3.0])
        gate_ang = np.array([0.0, 0.0, 0.0])

        for new_distance, new_angle in zip(distances, angles):

            # if we're below a certain height, only go up
            if gate_pos[2] < self.max_gate_distance * math.cos(
                self.max_gate_angles[0, 1]
            ):
                new_angle[1] = -abs(new_angle[1])

            # old rotation matrix and quat
            old_quat = p.getQuaternionFromEuler(gate_ang)
            old_mat = np.array(p.getMatrixFromQuaternion(old_quat)).reshape(3, 3)

            # new rotation matrix and quat
            new_quat = p.getQuaternionFromEuler(new_angle)
            new_mat = np.array(p.getMatrixFromQuaternion(new_quat)).reshape(3, 3)

            # rotate new distance by old angle and then new angle
            new_distance = np.array([new_distance, 0.0, 0.0])
            new_distance = new_mat @ old_mat @ new_distance

            # new position
            gate_pos += new_distance
            gate_ang += new_angle

            # get new gate quaternion
            gate_quat = p.getQuaternionFromEuler(gate_ang)

            # store the new target and gates
            self.targets.append(copy.copy(gate_pos))
            self.gates.append(
                self.env.loadURDF(
                    self.gate_obj_dir,
                    basePosition=gate_pos,
                    baseOrientation=gate_quat,
                    useFixedBase=True,
                )
            )

        # colour the first gate
        self.colour_first_gate()
        self.colour_other_gate()

        # update the gates
        self.info["target"] = self.targets[0]

    def colour_dead_gate(self, gate):
        """colour_dead_gate.

        Args:
            gate:
        """
        # colour the dead gates red
        for i in range(p.getNumJoints(gate)):
            p.changeVisualShape(
                gate,
                linkIndex=i,
                rgbaColor=(1, 0, 0, 1),
            )

    def colour_first_gate(self):
        """colour_first_gate."""
        # colour the first gate green
        for i in range(p.getNumJoints(self.gates[0])):
            p.changeVisualShape(
                self.gates[0],
                linkIndex=i,
                rgbaColor=(0, 1, 0, 1),
            )

    def colour_other_gate(self):
        """colour_other_gate."""
        # colour all other gates yellow
        for gate in self.gates[1:]:
            for i in range(p.getNumJoints(gate)):
                p.changeVisualShape(
                    gate,
                    linkIndex=i,
                    rgbaColor=(1, 1, 0, 1),
                )

    @property
    def state(self):
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

        # quarternion angles
        q_ang_pos = p.getQuaternionFromEuler(ang_pos)

        # drone to target
        self.dis_error_scalar = np.linalg.norm(self.targets[0] - lin_pos)

        obs = dict()

        # combine everything
        if self.ang_rep == 0:
            obs["attitude"] = np.array([*ang_vel, *ang_pos, *lin_vel, *lin_pos])
        elif self.ang_rep == 1:
            obs["attitude"] = np.array([*ang_vel, *q_ang_pos, *lin_vel, *lin_pos])

        # grab the image
        obs["rgba_cam"] = self.env.drones[0].rgbImg

        return obs

    @property
    def target_reached(self):
        """target_reached."""
        if self.dis_error_scalar < self.goal_reach_distance:
            return True
        else:
            return False

    @property
    def reward(self):
        """reward."""
        if self.info["collision"]:
            return -100.0
        else:
            # normal reward
            return self.dis_error_scalar

    def compute_term_trunc(self):
        """compute_term_trunc."""
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation = self.truncation or True

        # out of range of next gate
        if self.dis_error_scalar > 2 * self.max_gate_distance:
            self.info["out_of_bounds"] = True
            self.termination = self.termination or True

        # collision
        if len(self.env.getContactPoints(bodyA=self.env.drones[0].Id)) > 0:
            self.info["collision"] = True
            self.termination = self.termination or True

        # target reached
        if self.target_reached:
            if len(self.targets) > 1:
                # still have targets to go
                self.targets = self.targets[1:]
            else:
                self.info["env_complete"] = True
                self.termination = self.termination or True

            # shift the gates and recolour the reached one
            self.colour_dead_gate(self.gates[0])
            self.gates = self.gates[1:]

            # colour the new target
            self.colour_first_gate()

    def step(self, action: np.ndarray):
        """
        step the entire simulation
            output is states, reward, termination, truncation, info
        """
        # unsqueeze the action to be usable in aviary
        action = np.expand_dims(action, axis=0)

        # step env
        self.env.set_setpoints(action)
        self.env.step()

        # compute state and done
        self.compute_term_trunc()

        # increment step count
        self.step_count += 1

        # show where the next target is
        self.info["target"] = self.targets[0]

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self):
        """render."""
        raise AssertionError(
            "This function is not meant to be called. Apply `render_mode='human'` on environment creation."
        )
