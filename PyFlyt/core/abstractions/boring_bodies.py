"""A component to simulate bodies moving through the air."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from pybullet_utils import bullet_client


class BoringBodies:
    """Vectorized implementation of a series of plain bodies affected by aerodynamics.

    The `BoringBodies` component is used to represent a normal body moving through the air.

    Args:
        p (bullet_client.BulletClient): PyBullet physics client ID.
        physics_period (float): physics period of the simulation.
        np_random (np.random.RandomState): random number generator of the simulation.
        uav_id (int): ID of the drone.
        body_ids (np.ndarray | Sequence[int]): (n,) array of IDs for the links representing the bodies.
        drag_coefs (np.ndarray): (n, 3) array of drag coefficients for each body in the link-referenced XYZ directions.
        normal_areas (np.ndarray): (n, 3) array of frontal areas in the link-referenced XYZ directions.
    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        physics_period: float,
        np_random: np.random.RandomState,
        uav_id: int,
        body_ids: np.ndarray | Sequence[int],
        drag_coefs: np.ndarray,
        normal_areas: np.ndarray,
    ):
        """Used for simulating a body moving through the air.

        Args:
            p (bullet_client.BulletClient): PyBullet physics client ID.
            physics_period (float): physics period of the simulation.
            np_random (np.random.RandomState): random number generator of the simulation.
            uav_id (int): ID of the drone.
            body_ids (np.ndarray | Sequence[int]): (n,) array of IDs for the links representing the bodies.
            drag_coefs (np.ndarray): (n, 3) array of drag coefficients for each body in the link-referenced XYZ directions.
            normal_areas (np.ndarray): (n, 3) array of frontal areas in the link-referenced XYZ directions.
        """
        self.p = p
        self.physics_period = physics_period
        self.np_random = np_random

        # store IDs
        self.uav_id = uav_id
        self.body_ids = body_ids

        # some checks
        assert -1 not in body_ids
        assert drag_coefs.shape == (len(body_ids), 3), f"got {drag_coefs.shape}."
        assert normal_areas.shape == (len(body_ids), 3), f"got {normal_areas.shape}."

        # constants
        self.drag_consts = 0.5 * 1.225 * drag_coefs * normal_areas

        # runtime parameters
        self.local_body_velocities = np.zeros((len(self.body_ids), 3))

    def reset(self):
        """Reset the boring bodies."""
        self.local_body_velocities = np.zeros((len(self.body_ids), 3))

    def get_states(self):
        """`get_states` does not exist for boring bodies, they're boring."""
        raise NotImplementedError(
            "`get_states` does not exist for boring bodies, they're boring."
        )

    def state_update(self, rotation_matrix: np.ndarray):
        """Updates the local surface velocity of the boring body.

        Args:
            rotation_matrix (np.ndarray): (3, 3) rotation_matrix of the main body
        """
        # get all the states for all the bodies
        link_states = self.p.getLinkStates(
            self.uav_id, self.body_ids, computeLinkVelocity=True
        )

        # get all the velocities
        body_velocities = np.array([item[-2] for item in link_states])

        # query for wind if available and add to surface velocities
        if self.p.wind_field is not None:
            body_positions = np.array([item[0] for item in link_states])
            body_velocities -= self.p.wind_field(self.p.elapsed_time, body_positions)

        # rotate all velocities to be in local frame
        body_velocities = np.matmul(rotation_matrix, body_velocities.T).T

        if rotation_matrix.shape == (len(self.body_ids), 3, 3):
            body_velocities = np.matmul(
                rotation_matrix, np.expand_dims(body_velocities, -1)
            ).squeeze(-1)
        elif rotation_matrix.shape == (3, 3):
            body_velocities = np.matmul(rotation_matrix, body_velocities.T).T
        else:
            raise ValueError(
                f"Only accept (num_bodies, 3, 3) or (3, 3) array for `rotation_matrix`, got {rotation_matrix.shape}."
            )

        # update the variable
        self.local_body_velocities = body_velocities

    def physics_update(self):
        """Applies a force to the boring bodies depending on their local surface velocities."""
        forces = (
            -np.sign(self.local_body_velocities)
            * self.drag_consts
            * self.local_body_velocities**2
        )
        for i, force in enumerate(forces):
            self.p.applyExternalForce(
                self.uav_id,
                self.body_ids[i],
                force,
                [0.0, 0.0, 0.0],
                self.p.LINK_FRAME,
            )
