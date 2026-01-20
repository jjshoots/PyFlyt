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
        min_height: float,
        np_random: np.random.Generator,
        unordered: bool = False,  # <--- NEW ARGUMENT
    ):
        """__init__.

        Args:
            enable_render (bool): enable_render
            num_targets (int): num_targets
            use_yaw_targets (bool): use_yaw_targets
            goal_reach_distance (float): goal_reach_distance
            goal_reach_angle (float): goal_reach_angle
            flight_dome_size (float): flight_dome_size
            min_height (float): min_height
            np_random (np.random.Generator): np_random

        """
        # constants
        self.enable_render = enable_render
        self.num_targets = num_targets
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.flight_dome_size = flight_dome_size
        self.min_height = min_height
        self.np_random = np_random
        self.unordered = unordered  # <--- STORE IT
        self.captured_index = -1    # <--- TRACKER FOR UNORDERED CAPTURE

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
        self.new_distance = np.inf
        self.old_distance = np.inf

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
            self.targets[i] = np.array(
                [x, y, z if z > self.min_height else self.min_height]
            )

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

    @property
    def distance_to_next_target(self) -> float:
        """distance_to_next_target.

        Returns:
            float:
        """
        return self.new_distance

    def distance_to_targets(
        self,
        ang_pos: np.ndarray,
        lin_pos: np.ndarray,
        quaternion: np.ndarray,
    ):
        # rotation matrix
        rotation = np.array(self.p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

        # drone to target
        target_deltas = np.matmul((self.targets - lin_pos), rotation)

        # --- MODIFIED SECTION START ---
        if self.unordered:
            # UNORDERED: Track distance to the CLOSEST target
            if len(target_deltas) > 0:
                dists = np.linalg.norm(target_deltas, axis=1)
                self.new_distance = float(np.min(dists))
                
                # Check if we hit anything
                hits = np.where(dists < self.goal_reach_distance)[0]
                if len(hits) > 0:
                    self.captured_index = hits[0]
                else:
                    self.captured_index = -1
            else:
                self.new_distance = 0.0
                self.captured_index = -1
        else:
            # ORIGINAL: Track distance to FIRST target only
            self.new_distance = float(np.linalg.norm(target_deltas[0]))
            self.captured_index = 0 if self.new_distance < self.goal_reach_distance else -1
        
        self.old_distance = self.new_distance # Update old distance for reward calculation
        # --- MODIFIED SECTION END ---

        if self.use_yaw_targets:
            # ... (keep existing yaw logic, but be careful it usually assumes index 0) ...
            # For simplicity, calculate yaw error based on the 'active' target
            target_idx = 0 if not self.unordered else (self.captured_index if self.captured_index != -1 else np.argmin(np.linalg.norm(target_deltas, axis=1)))
            
            yaw_errors = self.yaw_targets - ang_pos[-1]
            yaw_errors[yaw_errors > math.pi] -= 2.0 * math.pi
            yaw_errors[yaw_errors < -math.pi] += 2.0 * math.pi
            yaw_errors = yaw_errors[..., None]

            target_deltas = np.concatenate([target_deltas, yaw_errors], axis=-1)
            self.yaw_error_scalar = np.abs(yaw_errors[target_idx])

        return target_deltas

    @property
    def progress_to_next_target(self):
        """progress_to_target."""
        if np.any(np.isinf(self.old_distance + self.new_distance)):
            return 0.0
        return self.old_distance - self.new_distance

    @property
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
        # Determine which index to remove
        # If unordered, use the captured one. If ordered, always 0.
        remove_idx = self.captured_index if self.unordered else 0
        
        if remove_idx == -1 or len(self.targets) == 0:
            return

        # Delete from numpy arrays
        self.targets = np.delete(self.targets, remove_idx, axis=0)
        if self.use_yaw_targets:
            self.yaw_targets = np.delete(self.yaw_targets, remove_idx, axis=0)

        # Delete visual body
        if self.enable_render and len(self.target_visual) > 0:
            self.p.removeBody(self.target_visual[remove_idx])
            del self.target_visual[remove_idx]

            # Recolour remaining
            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / max(len(self.target_visual), 1)), 0, 1),
                )
        
        self.captured_index = -1

    @property
    def num_targets_reached(self):
        """num_targets_reached."""
        return self.num_targets - len(self.targets)

    @property
    def all_targets_reached(self):
        """all_targets_reached."""
        return len(self.targets) == 0
