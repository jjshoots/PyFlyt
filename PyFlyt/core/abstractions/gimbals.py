"""A component to simulate an array of gimbals on vehicle."""
from __future__ import annotations

import warnings

import numpy as np
from pybullet_utils import bullet_client

from PyFlyt.core.utils.compile_helpers import jitter


class Gimbals:
    """A set of actuated gimbals.

    The `Gimbals` component simulates a series of servo actuated gimbals.
    Each gimbal can rotate about two arbitrary axis that may not be orthogonal to each other.

    Args:
        p (bullet_client.BulletClient): PyBullet physics client ID.
        physics_period (float): physics period of the simulation.
        np_random (np.random.RandomState): random number generator of the simulation.
        gimbal_unit_1 (np.ndarray): first unit vector that the gimbal rotates around.
        gimbal_unit_2 (np.ndarray): second unit vector that the gimbal rotates around.
        gimbal_tau (np.ndarray): gimbal actuation time constant.
        gimbal_range_degrees (np.ndarray): gimbal actuation range in degrees.
    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        physics_period: float,
        np_random: np.random.RandomState,
        gimbal_unit_1: np.ndarray,
        gimbal_unit_2: np.ndarray,
        gimbal_tau: np.ndarray,
        gimbal_range_degrees: np.ndarray,
    ):
        """Used for simulating an array of gimbals.

        Args:
            p (bullet_client.BulletClient): PyBullet physics client ID.
            physics_period (float): physics period of the simulation.
            np_random (np.random.RandomState): random number generator of the simulation.
            gimbal_unit_1 (np.ndarray): first unit vector that the gimbal rotates around.
            gimbal_unit_2 (np.ndarray): second unit vector that the gimbal rotates around.
            gimbal_tau (np.ndarray): gimbal actuation time constant.
            gimbal_range_degrees (np.ndarray): gimbal actuation range in degrees.
        """
        self.p = p
        self.physics_period = physics_period
        self.np_random = np_random

        assert (
            len(gimbal_unit_1.shape) == 2 and gimbal_unit_1.shape[-1] == 3
        ), f"Expected `gimbal_unit_1` to be of shape (n, 3), got {gimbal_unit_1.shape}"
        assert (
            len(gimbal_unit_2.shape) == 2 and gimbal_unit_2.shape[-1] == 3
        ), f"Expected `gimbal_unit_2` to be of shape (n, 3), got {gimbal_unit_1.shape}"
        assert (
            gimbal_unit_1.shape[0] == gimbal_unit_2.shape[0]
        ), f"Expected both gimbal_units to have equal number of elements, got {gimbal_unit_1.shape} and {gimbal_unit_2.shape}"
        assert gimbal_tau.shape == (gimbal_unit_1.shape[0],)
        assert gimbal_range_degrees.shape == (gimbal_unit_1.shape[0], 2)

        self.num_gimbals = gimbal_unit_1.shape[0]

        # check that the gimbal_axis is normalized
        for i, axis in enumerate(gimbal_unit_1):
            if np.linalg.norm(axis) != 1.0:
                warnings.warn(
                    f"Norm of `gimbal_unit_1` at element {i}: {gimbal_unit_1[i]=} is not 1.0, normalizing..."
                )
                gimbal_unit_1[i] /= np.linalg.norm(gimbal_unit_1[i])
        for i, axis in enumerate(gimbal_unit_2):
            if np.linalg.norm(axis) != 1.0:
                warnings.warn(
                    f"Norm of `gimbal_unit_2` at element {i}: {gimbal_unit_2[i]=} is not 1.0, normalizing..."
                )
                gimbal_unit_2[i] /= np.linalg.norm(gimbal_unit_2[i])

        # check that gimbal axes are orthogonal
        for i, (ax1, ax2) in enumerate(zip(gimbal_unit_1, gimbal_unit_2)):
            if np.dot(ax1, ax2) != 0.0:
                warnings.warn(
                    f"gimbal units at element {i} ({gimbal_unit_1[i], gimbal_unit_2[i]}) are not orthogonal, you have been warned."
                )

        # constants
        self.gimbal_tau = gimbal_tau
        self.gimbal_range_radians = np.deg2rad(gimbal_range_degrees)

        # runtime variables
        # rotation matrices using
        # https://math.stackexchange.com/questions/142821/matrix-for-rotation-around-a-vector
        self.w1 = np.zeros((self.num_gimbals, 3, 3))
        self.w1[:, 2, 1] = gimbal_unit_1[:, 0]
        self.w1[:, 1, 2] = -gimbal_unit_1[:, 0]
        self.w1[:, 0, 2] = gimbal_unit_1[:, 1]
        self.w1[:, 2, 0] = -gimbal_unit_1[:, 1]
        self.w1[:, 1, 0] = gimbal_unit_1[:, 2]
        self.w1[:, 0, 1] = -gimbal_unit_1[:, 2]
        self.w1_squared = self.w1 @ self.w1

        self.w2 = np.zeros((self.num_gimbals, 3, 3))
        self.w2[:, 2, 1] = gimbal_unit_2[:, 0]
        self.w2[:, 1, 2] = -gimbal_unit_2[:, 0]
        self.w2[:, 0, 2] = gimbal_unit_2[:, 1]
        self.w2[:, 2, 0] = -gimbal_unit_2[:, 1]
        self.w2[:, 1, 0] = gimbal_unit_2[:, 2]
        self.w2[:, 0, 1] = -gimbal_unit_2[:, 2]
        self.w2_squared = self.w2 @ self.w2

    def reset(self):
        """Reset the gimbals."""
        self.gimbal_state = np.zeros((self.num_gimbals, 2), dtype=np.float64)
        self.rotation1 = np.array([np.eye(3)] * self.num_gimbals, dtype=np.float64)
        self.rotation2 = np.array([np.eye(3)] * self.num_gimbals, dtype=np.float64)

    def get_states(self) -> np.ndarray:
        """Gets the current state of the components.

        Returns:
            np.ndarray: a (2 * num_gimbals, ) array where every pair of values represents the current state of the gimbal
        """
        return np.concatenate(
            [
                self.gimbal_state.flatten(),  # [n, 2]
            ]
        )

    def state_update(self):
        """This does not need to be called for gimbals."""
        warnings.warn("`state_update` does not need to be called for gimbals.")

    def physics_update(self):
        """This does not need to be called for gimbals, call `compute_rotation` instead."""
        raise NameError(
            "`state_update` does not need to be called for gimbals, call `compute_rotation` instead."
        )

    def compute_rotation(self, gimbal_command: np.ndarray) -> np.ndarray:
        """Returns a rotation vector after the gimbal rotation.

        Args:
            gimbal_command (np.ndarray): (num_gimbals, 2) array of floats between [-1, 1].

        Returns:
            rotation_vector (np.ndarray): (num_gimbals, 3, 3) rotation matrices for all gimbals.
        """
        assert np.all(gimbal_command >= -1.0) and np.all(
            gimbal_command <= 1.0
        ), f"`{gimbal_command=} has values out of bounds of -1.0 and 1.0.`"

        # model the gimbal using first order ODE, y' = T/tau * (setpoint - y)
        self.gimbal_state += (self.physics_period / self.gimbal_tau) * (
            gimbal_command - self.gimbal_state
        )

        # compute gimbal euler angles
        gimbal_angles = self.gimbal_state * self.gimbal_range_radians
        gimbal_angles = gimbal_angles.reshape(*gimbal_angles.shape, 1, 1)

        # compute gimbal rotation matrix
        (rotation1, rotation2) = self._jitted_compute_rotation(
            gimbal_angles,
            self.w1,
            self.w2,
            self.w1_squared,
            self.w2_squared,
        )
        return rotation1 @ rotation2

    @staticmethod
    @jitter
    def _jitted_compute_rotation(
        gimbal_angles: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        w1_squared: np.ndarray,
        w2_squared: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the rotation matrix given the gimbal action values.

        Args:
            gimbal_angles (np.ndarray): gimbal_angles
            w1 (np.ndarray): w1 from self
            w2 (np.ndarray): w2 from self
            w1_squared (np.ndarray): w1_squared from self
            w2_squared (np.ndarray): w2_squared from self

        Returns:
            tuple[np.ndarray, np.ndarray]:
        """
        # precompute some things
        sin_angles = np.sin(gimbal_angles)
        sin_half_angles = np.sin(gimbal_angles / 2.0)

        # start calculating rotation matrices
        # https://math.stackexchange.com/questions/142821/matrix-for-rotation-around-a-vector
        rotation1 = (
            np.eye(3)
            + sin_angles[:, 0, ...] * w1
            + 2 * (sin_half_angles[:, 0, ...] ** 2) * w1_squared
        )
        rotation2 = (
            np.eye(3)
            + sin_angles[:, 1, ...] * w2
            + 2 * (sin_half_angles[:, 1, ...] ** 2) * w2_squared
        )

        return rotation1, rotation2
