"""Handler for the pole."""

from __future__ import annotations

import os

import numpy as np
from gymnasium import spaces
from pybullet_utils import bullet_client


class PoleHandler:
    """PoleHandler."""

    def __init__(self):
        """__init__."""
        # the pole urdf
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.pole_obj_dir = os.path.join(file_dir, "../../models/pole.urdf")

        # modify the state to take into account the pole's state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float64,
        )

    def reset(
        self,
        p: bullet_client.BulletClient,
        start_location: np.ndarray,
    ):
        """reset.

        Args:
            p (bullet_client.BulletClient): p
            start_location (np.ndarray): start_location

        """
        # store the client
        self.p = p

        # spawn in a pole and make it have enough friction
        self.Id = self.p.loadURDF(
            self.pole_obj_dir,
            basePosition=start_location,
            useFixedBase=False,
        )
        self.p.changeDynamics(
            self.Id,
            linkIndex=1,
            lateralFriction=1.0e5,
            restitution=0.01,
        )

        self._leaningness = 0.0

    @property
    def leaningness(self) -> float:
        """The lean of the pole, minimum of 0.0 and maximum of 1.0.

        Returns:
            float:

        """
        return float(self._leaningness)

    def compute_state(
        self, rotation: np.ndarray, uav_lin_pos: np.ndarray, uav_lin_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """compute_state.

        Args:
            rotation (np.ndarray): rotation
            uav_lin_pos (np.ndarray): uav_lin_pos
            uav_lin_vel (np.ndarray): uav_lin_vel

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: top_pos, top_vel, bot_pos, bot_vel

        """
        # we measure the top and bottom linear position and velocity of the pole
        # compute the attitude of the pole in global coords
        pole_top_pos, *_, pole_top_vel, _ = self.p.getLinkState(
            self.Id, linkIndex=0, computeLinkVelocity=True
        )
        pole_bot_pos, *_, pole_bot_vel, _ = self.p.getLinkState(
            self.Id, linkIndex=1, computeLinkVelocity=True
        )

        # convert to np_array
        pole_top_pos = np.array(pole_top_pos)
        pole_top_vel = np.array(pole_top_vel)
        pole_bot_pos = np.array(pole_bot_pos)
        pole_bot_vel = np.array(pole_bot_vel)

        # compute the uprightness of the pole BEFORE we do axis transforms
        if pole_top_pos[-1] > pole_bot_pos[-1]:
            self._leaningness = np.linalg.norm(pole_top_pos[:2] - pole_bot_pos[:2])
        else:
            self._leaningness = 1.0

        # get everything relative to the drone's position
        pole_top_pos: np.ndarray = np.matmul(rotation, (pole_top_pos - uav_lin_pos))
        pole_bot_pos: np.ndarray = np.matmul(rotation, (pole_bot_pos - uav_lin_pos))
        pole_top_vel: np.ndarray = np.matmul(rotation, pole_top_vel) - uav_lin_vel
        pole_bot_vel: np.ndarray = np.matmul(rotation, pole_bot_vel) - uav_lin_vel

        return (
            pole_top_pos,
            pole_top_vel,
            pole_bot_pos,
            pole_bot_vel,
        )
