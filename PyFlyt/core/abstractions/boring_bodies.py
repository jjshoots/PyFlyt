"""A component to simulate bodies moving through the air."""
from __future__ import annotations

import warnings

import numpy as np
from pybullet_utils import bullet_client


class BoringBodies:
    """BoringBodies."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        physics_period: float,
        np_random: np.random.RandomState,
        uav_id: int,
        body_ids: np.ndarray,
        drag_coefs: np.ndarray,
        normal_areas: np.ndarray,
    ):
        """Used for simulating a body moving through the air.

        Args:
            p (bullet_client.BulletClient): p
            physics_period (float): physics_period
            np_random (np.random.RandomState): np_random
            uav_id (int): uav_id
            body_ids (np.ndarray): (n,) array of ids for the links representing the bodies
            drag_coefs (np.ndarray): (n, 3) array of drag coefs in the x, y, z
            normal_areas (np.ndarray): (n, 3) array of frontal areas in the x, y, z
        """
        self.p = p
        self.physics_period = physics_period
        self.np_random = np_random

        # store IDs
        self.uav_id = uav_id
        self.body_ids = body_ids

        # some checks
        assert -1 not in body_ids
        assert drag_coefs.shape == (len(body_ids), 3)
        assert normal_areas.shape == (len(body_ids), 3)

        # constants
        self.drag_consts = 0.5 * 1.225 * drag_coefs * normal_areas

        # runtime parameters
        self.local_body_velocities = np.zeros((len(self.body_ids), 3))

    def reset(self):
        """Reset the boring bodies."""
        self.local_body_velocities = np.zeros((len(self.body_ids), 3))

    def update_local_surface_velocity(
        self, rotation_matrices: np.ndarray
    ):
        """Updates the local surface velocity of the boring body.

        Args:
            rotation_matrices (np.ndarray): (n, 3, 3) array of rotation matrices of the bodies
        """
        # get all the states for all the bodies
        body_velocities = self.p.getLinkStates(
            self.uav_id, self.body_ids, computeLinkVelocity=True
        )

        # get all the velocities
        body_velocities = np.array([item[-2] for item in body_velocities])
        body_velocities = np.expand_dims(body_velocities, axis=-1)

        # rotate all velocities to be in local frame
        body_velocities = np.matmul(rotation_matrices, body_velocities)
        body_velocities = np.squeeze(body_velocities, axis=-1)

        # update the variable
        self.local_body_velocities = body_velocities

    def update_forces(self):
        """Applies a force to the boring bodies depending on their local surface velocities."""
        forces = -self.local_body_velocities * self.drag_consts
        print(forces)
        for i, force in enumerate(forces):
            self.p.applyExternalForce(
                self.uav_id,
                self.body_ids[i],
                force,
                [0.0, 0.0, 0.0],
                self.p.LINK_FRAME,
            )
