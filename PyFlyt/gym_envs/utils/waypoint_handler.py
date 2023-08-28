"""Handler for Waypoints in the environments."""
from __future__ import annotations

import math
import os

import numpy as np
from pybullet_utils import bullet_client


class WaypointHandler:
    """Handler for Waypoints in the environments."""

    def __init__(
        self,
        enable_render: bool,
        num_targets: int,
        use_yaw_targets: bool,
        goal_reach_distance: float,
        goal_reach_angle: float,
        flight_dome_size: float,
        np_random: np.random.Generator,
    ):
        """__init__.

        Args:
            enable_render (bool): enable_render
            num_targets (int): num_targets
            use_yaw_targets (bool): use_yaw_targets
            goal_reach_distance (float): goal_reach_distance
            goal_reach_angle (float): goal_reach_angle
            flight_dome_size (float): flight_dome_size
            np_random (np.random.Generator): np_random
        """
        # constants
        self.enable_render = enable_render
        self.num_targets = num_targets
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.flight_dome_size = flight_dome_size
        self.np_random = np_random

        # the target visual
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, "../../models/target.urdf")

    def reset(
        self,
        p: bullet_client.BulletClient,
        np_random: None | np.random.Generator = None,
    ):
        """Resets the waypoints."""
        # store the client
        self.p = p

        # update the random state
        if np_random is not None:
            self.np_random = np_random

        # reset the error
        self.new_distance = 0.0
        self.old_distance = 0.0

        # we sample from polar coordinates to generate linear targets
        self.targets = np.zeros(shape=(self.num_targets, 3))
        thetas = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        phis = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        for i, theta, phi in zip(range(self.num_targets), thetas, phis):
            dist = self.np_random.uniform(low=1.0, high=self.flight_dome_size * 0.9)
            x = dist * math.sin(phi) * math.cos(theta)
            y = dist * math.sin(phi) * math.sin(theta)
            z = abs(dist * math.cos(phi))

            # check for floor of z
            self.targets[i] = np.array([x, y, z if z > 0.1 else 0.1])

        # yaw targets
        if self.use_yaw_targets:
            self.yaw_targets = self.np_random.uniform(
                low=-math.pi, high=math.pi, size=(self.num_targets,)
            )

        # if we are rendering, load in the targets
        if self.enable_render:
            self.target_visual = []
            for target in self.targets:
                self.target_visual.append(
                    self.p.loadURDF(
                        self.targ_obj_dir,
                        basePosition=target,
                        useFixedBase=True,
                        globalScaling=self.goal_reach_distance / 4.0,
                    )
                )

            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )

    def distance_to_target(
        self,
        ang_pos: np.ndarray,
        lin_pos: np.ndarray,
        quarternion: np.ndarray,
    ):
        """distance_to_target.

        Args:
            ang_pos (np.ndarray): ang_pos
            lin_pos (np.ndarray): lin_pos
            quarternion (np.ndarray): quarternion
        """
        # rotation matrix
        rotation = np.array(self.p.getMatrixFromQuaternion(quarternion)).reshape(3, 3)

        # drone to target
        target_deltas = np.matmul((self.targets - lin_pos), rotation)

        # record distance to the next target
        self.old_distance = self.new_distance
        self.new_distance = float(np.linalg.norm(target_deltas[0]))

        if self.use_yaw_targets:
            yaw_errors = self.yaw_targets - ang_pos[-1]

            # rollover yaw
            yaw_errors[yaw_errors > math.pi] -= 2.0 * math.pi
            yaw_errors[yaw_errors < -math.pi] += 2.0 * math.pi
            yaw_errors = np.expand_dims(yaw_errors, axis=-1)

            # add the yaw delta to the target deltas
            target_deltas = np.concatenate([target_deltas, yaw_errors], axis=-1)

            # compute the yaw error scalar
            self.yaw_error_scalar = np.abs(yaw_errors[0])

        return target_deltas

    def progress_to_target(self):
        """progress_to_target."""
        return self.old_distance - self.new_distance

    def target_reached(self):
        """target_reached."""
        if not self.new_distance < self.goal_reach_distance:
            return False

        if not self.use_yaw_targets:
            return True

        if self.yaw_error_scalar < self.goal_reach_angle:
            return True

        return False

    def advance_targets(self):
        """advance_targets."""
        if len(self.targets) > 1:
            # still have targets to go
            self.targets = self.targets[1:]
            if self.use_yaw_targets:
                self.yaw_targets = self.yaw_targets[1:]
        else:
            self.targets = []
            self.yaw_targets = []

        # delete the reached target and recolour the others
        if self.enable_render and len(self.target_visual) > 0:
            self.p.removeBody(self.target_visual[0])
            self.target_visual = self.target_visual[1:]

            # recolour
            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )

    def num_targets_reached(self):
        """num_targets_reached."""
        return self.num_targets - len(self.targets)

    def all_targets_reached(self):
        """all_targets_reached."""
        return len(self.targets) == 0
