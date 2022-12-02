from __future__ import annotations

import copy
import math
import os

import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.spaces import GraphInstance

from .pyflyt_env import PyFlytEnv


class AdvancedGatesEnv(PyFlytEnv):
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
        num_targets: int = 5,
        goal_reach_distance: float = 0.21,
        min_gate_height: float = 1.0,
        max_gate_angles: list[float] = [0.0, 0.3, 1.0],
        min_gate_distance: float = 1.0,
        max_gate_distance: float = 4.0,
        camera_resolution: tuple[int, int] = (128, 128),
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            goal_reach_distance (float): goal_reach_distance
            min_gate_height (float): the minimum height of any gate
            max_gate_angles (list[float]): max_gate_angles
            min_gate_distance (float): min_gate_distance
            max_gate_distance (float): max_gate_distance
            camera_resolution (tuple[int, int]): camera_resolution
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
        self.observation_space = spaces.Dict(
            {
                "attitude": self.attitude_space,
                "rgba_cam": spaces.Box(
                    low=0.0, high=255.0, shape=(4, *camera_resolution), dtype=np.uint8
                ),
                "target_deltas": spaces.Graph(
                    node_space=spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                    edge_space=None,
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
        self.min_gate_height = min_gate_height
        self.min_gate_distance = min_gate_distance
        self.max_gate_distance = max_gate_distance
        self.max_gate_angles = np.array([max_gate_angles])
        self.goal_reach_distance = goal_reach_distance

        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.gate_obj_dir = os.path.join(file_dir, f"../models/race_gate.urdf")
        self.camera_resolution = camera_resolution
        self.num_targets = num_targets

    def reset(self, seed=None, options=None):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options:
        """
        aviary_options = dict()
        aviary_options["use_camera"] = True
        aviary_options["use_gimbal"] = False
        aviary_options["camera_resolution"] = self.camera_resolution
        aviary_options["camera_angle_degrees"] = 15.0
        super().begin_reset(seed, options, aviary_options)

        """GATES GENERATION"""
        self.gates = []
        self.targets = []
        self.generate_gates()

        super().end_reset(seed, options)

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
        gate_pos = np.array([0.0, 0.0, 1.0])
        gate_ang = np.array([0.0, 0.0, 0.0])

        for new_distance, new_angle in zip(distances, angles):

            # check that the gate we're about to make doesn't go below the minimum height
            vertical_offset = 0.0
            if (
                gate_pos[2]
                + self.max_gate_distance * math.cos(self.max_gate_angles[0, 1])
                < self.min_gate_height
            ):
                vertical_offset = gate_pos[2] + self.max_gate_distance * math.cos(
                    self.max_gate_angles[0, 1]
                )

            # old rotation matrix and quat
            old_quat = p.getQuaternionFromEuler(gate_ang)
            old_mat = np.array(p.getMatrixFromQuaternion(old_quat)).reshape(3, 3)

            # new rotation matrix and quat
            new_quat = p.getQuaternionFromEuler(new_angle)
            new_mat = np.array(p.getMatrixFromQuaternion(new_quat)).reshape(3, 3)

            # rotate new distance by old angle and then new angle
            new_distance = np.array([new_distance, 0.0, vertical_offset])
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

    def compute_state(self):
        """state.

        This returns the observation as well as the distances to target.
        - "attitude" (Box)
            - ang_vel (vector of 3 values)
            - ang_pos (vector of 3/4 values)
            - lin_vel (vector of 3 values)
            - lin_pos (vector of 3 values)
        - "target_deltas" (Graph)
            - list of body_frame distances to target (vector of 3/4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()

        # rotation matrix
        rotation = np.array(p.getMatrixFromQuaternion(quarternion)).reshape(3, 3).T

        # drone to target
        target_deltas = np.matmul(rotation, (self.targets - lin_pos).T).T
        self.dis_error_scalar = np.linalg.norm(target_deltas[0])

        # combine everything
        new_state = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action]
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action]
            )

        # grab the image
        img = self.env.drones[0].rgbaImg.astype(np.uint8)
        new_state["rgba_cam"] = np.moveaxis(img, -1, 0)

        # distances to targets
        new_state["target_deltas"] = GraphInstance(
            nodes=target_deltas, edge_links=None, edges=None
        )

        self.state = new_state

    @property
    def target_reached(self):
        """target_reached."""
        if self.dis_error_scalar < self.goal_reach_distance:
            return True
        else:
            return False

    def compute_term_trunc_reward(self):
        """compute_term_trunc_reward."""
        super().compute_base_term_trunc_reward()

        # out of range of next gate
        if self.dis_error_scalar > 2 * self.max_gate_distance:
            self.reward += -100.0
            self.info["out_of_bounds"] = True
            self.termination = self.termination or True

        # target reached
        if self.target_reached:
            self.reward += 100.0
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
