import math

import gym
import numpy as np
import pybullet as p
from gym import spaces

from PyFlyt.core.aviary import Aviary


class SimpleWaypointEnv(gym.Env):
    """
    Simple Waypoint Environment

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is a set of `[x, y, z, yaw]` targets in space

    Reward is -(distance from waypoint + angle error) for each timestep,
    and -100.0 for hitting the ground.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps=10000,
        angle_representation="quaternion",
        num_targets=4,
        use_yaw_targets=True,
        goal_reach_distance=0.2,
        goal_reach_angle=0.1,
        flight_dome_size=10.0,
    ):

        """GYM STUFF"""
        # observation size increases by 2 for euler
        if angle_representation == "euler":
            obs_shape = 15
        elif angle_representation == "quaternion":
            obs_shape = 17
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        # if we have yaw targets, then the obs has yaw targets as well
        if use_yaw_targets:
            obs_shape += 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        high = np.array([3.0, 3.0, 3.0, 1.0])
        low = np.array([-3.0, -3.0, -3.0, -1.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        """ ENVIRONMENT CONSTANTS """
        self.enable_render = False
        self.flight_dome_size = flight_dome_size
        self.max_steps = max_steps
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.ang_rep = 0
        if angle_representation == "euler":
            self.ang_rep = 0
        elif angle_representation == "quaternion":
            self.ang_rep = 1

        """ RUNTIME VARIABLES """
        self.env = None
        self.state = self.observation_space.sample()
        self.done = False
        self.dis_error = -100.0
        self.yaw_error = -100.0

        # we sample from polar coordinates to generate linear targets
        self.targets = np.zeros(shape=(num_targets, 3))
        thts = np.random.uniform(0.0, 2.0 * math.pi, size=(num_targets,))
        phis = np.random.uniform(0.0, 2.0 * math.pi, size=(num_targets,))
        for i, tht, phi in zip(range(num_targets), thts, phis):
            x = 1.0 * math.sin(phi) * math.cos(tht)
            y = 1.0 * math.sin(phi) * math.sin(tht)
            z = np.abs(1.0 * math.cos(phi))
            self.targets[i] = np.array([x, y, z])

        # yaw targets
        if self.use_yaw_targets:
            self.yaw_targets = np.random.uniform(
                low=-math.pi, high=math.pi, size=(num_targets,)
            )

    def render(self, mode="human"):
        self.enable_render = True

    def reset(self):
        # if we already have an env, disconnect from it
        if self.env is not None:
            self.env.disconnect()

        # reset step count
        self.step_count = 0

        # init env
        self.env = Aviary(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=self.enable_render,
        )

        # set flight mode
        self.env.set_mode(6)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        return self.compute_state()

    def compute_state(self):
        """This computes the observation as well as the distances to target"""
        # ang_vel (3/4)
        # ang_pos (3/4)
        # lin_vel (3)
        # lin_pos (3)
        # body frame distance to targ (3)
        raw_state = self.env.states[0]

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        q_ang_vel = p.getQuaternionFromEuler(ang_vel)
        q_ang_pos = p.getQuaternionFromEuler(ang_pos)

        # rotation matrix
        rotation = np.array(p.getMatrixFromQuaternion(q_ang_pos)).reshape(3, 3).T

        # drone to target
        self.dis_error = np.matmul(rotation, self.targets[0] - lin_pos)
        self.yaw_error = self.yaw_targets[0] - ang_pos[-1]

        # rollover yaw
        if self.yaw_error > math.pi:
            self.yaw_error -= 2.0 * math.pi
        if self.yaw_error < -math.pi:
            self.yaw_error += 2.0 * math.pi

        # precompute the scalars so we can use it later
        self.dis_error_scalar = np.linalg.norm(self.dis_error)
        self.yaw_error_scalar = np.abs(self.yaw_error)

        # use targ_yaw if necessary
        if self.use_yaw_targets:
            error = np.array([*self.dis_error, self.yaw_error])
        else:
            error = self.dis_error

        # combine everything
        new_state = np.array([0])
        if self.ang_rep == 0:
            new_state = np.array([*ang_vel, *ang_pos, *lin_vel, *lin_pos, *error])
        elif self.ang_rep == 1:
            new_state = np.array([*q_ang_vel, *q_ang_pos, *lin_vel, *lin_pos, *error])

        return new_state

    @property
    def target_reached(self):
        if self.dis_error_scalar < self.goal_reach_distance:
            if self.use_yaw_targets:
                if self.yaw_error_scalar < self.goal_reach_angle:
                    return True
            else:
                return True

        return False

    @property
    def reward(self):
        if len(self.env.getContactPoints()) > 0:
            # collision with ground
            return -100.0
        else:
            # normal reward
            error = self.dis_error_scalar
            if self.use_yaw_targets:
                error += self.yaw_error_scalar
            return -error

    def compute_done(self):
        # exceed step count
        if self.step_count > self.max_steps:
            return True

        # exceed flight dome
        if np.linalg.norm(self.state[-3:]) > self.flight_dome_size:
            return True

        # collision
        if len(self.env.getContactPoints()) > 0:
            return True

        # target reached
        if self.target_reached:
            if len(self.targets) > 1:
                # still have targets to go
                self.targets = self.targets[1:]
                self.yaw_targets = self.yaw_targets[1:]
            else:
                return True

        return False

    def step(self, action: np.ndarray):
        """
        step the entire simulation
            output is states, reward, dones
        """
        # unsqueeze the action to be usable in aviary
        action = np.expand_dims(action, axis=0)

        # step env
        self.env.set_setpoints(action)
        self.env.step()

        # compute state and dones
        self.state = self.compute_state()
        self.done = self.compute_done()

        self.step_count += 1

        return self.state, self.reward, self.done, None
