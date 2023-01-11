from __future__ import annotations

import math
import os

import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.spaces import GraphInstance

from .pyflyt_env import PyFlytEnv


class SimpleWaypointEnv(PyFlytEnv):
    """
    Simple Waypoint Environment

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is a set of `[x, y, z, yaw]` targets in space

    Reward:
        100.0 for reaching target,
        -100 for collisions or out of bounds,
        -0.1 otherwise
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_targets: int = 4,
        use_yaw_targets: bool = True,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            num_targets (int): num_targets
            use_yaw_targets (bool): use_yaw_targets
            goal_reach_distance (float): goal_reach_distance
            goal_reach_angle (float): goal_reach_angle
            flight_dome_size (float): size of the allowable flying area
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
                "target_deltas": spaces.Graph(
                    node_space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float64,
                    ),
                    edge_space=None,
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
        self.flight_dome_size = flight_dome_size
        self.num_targets = num_targets
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle

        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, f"../models/target.urdf")

    def reset(self, seed=None, options=None):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options:
        """
        super().begin_reset(seed, options)

        """TARGET GENERATION"""
        # we sample from polar coordinates to generate linear targets
        self.targets = np.zeros(shape=(self.num_targets, 3))
        thts = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        phis = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        for i, tht, phi in zip(range(self.num_targets), thts, phis):
            dist = self.np_random.uniform(low=1.0, high=self.flight_dome_size * 0.9)
            x = dist * math.sin(phi) * math.cos(tht)
            y = dist * math.sin(phi) * math.sin(tht)
            z = abs(dist * math.cos(phi))

            # check for floor of z
            self.targets[i] = np.array([x, y, z if z > 0.1 else 0.1])

        # yaw targets
        if self.use_yaw_targets:
            self.yaw_targets = self.np_random.uniform(
                low=-math.pi, high=math.pi, size=(self.num_targets,)
            )

        # if we are rendering, laod in the targets
        if self.enable_render:
            self.target_visual = []
            for target in self.targets:
                self.target_visual.append(
                    self.env.loadURDF(
                        self.targ_obj_dir, basePosition=target, useFixedBase=True
                    )
                )

            for i, visual in enumerate(self.target_visual):
                p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )

        super().end_reset()

        return self.state, self.info

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

        if self.use_yaw_targets:
            yaw_errors = self.yaw_targets - ang_pos[-1]

            # rollover yaw
            yaw_errors[yaw_errors > math.pi] -= 2.0 * math.pi
            yaw_errors[yaw_errors < -math.pi] += 2.0 * math.pi

            # add the yaw delta to the target deltas
            target_deltas = np.concatenate([target_deltas, yaw_errors], axis=-1)

            # compute the yaw error scalar
            self.yaw_error_scalar = np.abs(yaw_errors[0])

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

        new_state["target_deltas"] = GraphInstance(
            nodes=target_deltas, edge_links=None, edges=None
        )

        self.state = new_state

    @property
    def target_reached(self):
        """target_reached."""
        if self.dis_error_scalar < self.goal_reach_distance:
            if self.use_yaw_targets:
                if self.yaw_error_scalar < self.goal_reach_angle:
                    return True
            else:
                return True

        return False

    def compute_term_trunc_reward(self):
        """compute_term_trunc."""
        super().compute_base_term_trunc_reward()

        # exceed flight dome
        if np.linalg.norm(self.env.states[0][-1]) > self.flight_dome_size:
            self.reward += -100.0
            self.info["out_of_bounds"] = True
            self.termination = self.termination or True

        # target reached
        if self.target_reached:
            self.reward += 100.0
            if len(self.targets) > 1:
                # still have targets to go
                self.targets = self.targets[1:]
                if self.use_yaw_targets:
                    self.yaw_targets = self.yaw_targets[1:]
            else:
                self.info["env_complete"] = True
                self.termination = self.termination or True

            # delete the reached target and recolour the others
            if self.enable_render and len(self.target_visual) > 0:
                p.removeBody(self.target_visual[0])
                self.target_visual = self.target_visual[1:]

                # recolour
                for i, visual in enumerate(self.target_visual):
                    p.changeVisualShape(
                        visual,
                        linkIndex=-1,
                        rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                    )
