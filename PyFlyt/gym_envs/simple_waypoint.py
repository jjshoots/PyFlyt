from __future__ import annotations
import math
import os

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.spaces import GraphInstance

from PyFlyt.core.aviary import Aviary


class SimpleWaypointEnv(gymnasium.Env):
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
        max_steps: int = 300,
        angle_representation: str = "quaternion",
        num_targets: int = 4,
        use_yaw_targets: bool = True,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_dome_size: float = 5.0,
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            max_steps (int): max_steps of the environment
            angle_representation (str): can be "euler" or "quaternion"
            num_targets (int): num_targets
            use_yaw_targets (bool): use_yaw_targets
            goal_reach_distance (float): goal_reach_distance
            goal_reach_angle (float): goal_reach_angle
            flight_dome_size (float): size of the allowable flying area
            agent_hz (int): looprate of the agent to environment interaction
            render_mode (None | str): can be "human" or None
        """

        if 120 % agent_hz != 0:
            lowest = int(120 / int(120 / agent_hz))
            highest = int(120 / (int(120 / agent_hz) + 1))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

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
            obs_shape = 16
        elif angle_representation == "quaternion":
            obs_shape = 17
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        # if we have yaw targets, then the obs has yaw targets as well
        if use_yaw_targets:
            obs_shape += 1

        self.observation_space = spaces.Dict(
            {
                "attitude": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
                ),
                "target_deltas": spaces.Graph(
                    node_space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                    edge_space=None,
                ),
            }
        )

        a_lim = math.pi / 2.0
        t_lim = 0.8
        high = np.array([a_lim, a_lim, a_lim, t_lim])
        low = np.array([-a_lim, -a_lim, -a_lim, 0.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        """ ENVIRONMENT CONSTANTS """
        self.cycle_steps = int(120 / agent_hz)
        self.flight_dome_size = flight_dome_size
        self.max_steps = max_steps
        self.num_targets = num_targets
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        if angle_representation == "euler":
            self.ang_rep = 0
        elif angle_representation == "quaternion":
            self.ang_rep = 1
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, f"../models/target.urdf")

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

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.action = np.zeros((4,))
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        # init env
        self.env = Aviary(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.enable_render,
            seed=seed,
        )

        # set flight mode
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

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

        self.compute_state()

        return self.state, self.info

    def compute_state(self):
        """state.

        This returns the observation as well as the distances to target.
        - "attitude" (Box)
            - ang_vel (vector of 3 values)
            - ang_pos (vector of 3/4 values)
            - lin_vel (vector of 3 values)
            - lin_pos (vector of 3 values)
        - "targets" (Sequence)
            - list of body_frame distances to target (vector of 3/4 values)
        """
        raw_state = self.env.states[0]

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        q_ang_pos = p.getQuaternionFromEuler(ang_pos)

        # rotation matrix
        rotation = np.array(p.getMatrixFromQuaternion(q_ang_pos)).reshape(3, 3).T

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
        if self.ang_rep == 0:
            new_state["attitude"] = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action]
            )
        elif self.ang_rep == 1:
            new_state["attitude"] = np.array(
                [*ang_vel, *q_ang_pos, *lin_vel, *lin_pos, *self.action]
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
        self.reward += -0.1

        # if we've already ended, just exit
        if self.termination or self.truncation:
            return

        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation = self.truncation or True

        # exceed flight dome
        if np.linalg.norm(self.env.states[0][-1]) > self.flight_dome_size:
            self.reward += -100.0
            self.info["out_of_bounds"] = True
            self.termination = self.termination or True

        # collision
        if np.any(self.env.collision_array):
            self.reward += -100.0
            self.info["collision"] = True
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

    def step(self, action: np.ndarray):
        """step.

        Args:
            action (np.ndarray): action
        """
        # unsqueeze the action to be usable in aviary
        self.action = action.copy()
        action = np.expand_dims(action, axis=0)

        # reset the reward and set the action
        self.reward = 0
        self.env.set_setpoints(action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.cycle_steps):
            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self):
        """render."""
        raise AssertionError(
            "This function is not meant to be called. Apply `render_mode='human'` on environment creation."
        )
