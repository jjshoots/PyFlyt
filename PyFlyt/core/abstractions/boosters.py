from __future__ import annotations

import warnings

import numpy as np
from pybullet_utils import bullet_client


class Boosters:
    """Boosters."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        physics_period: float,
        np_random: np.random.RandomState,
        uav_id: int,
        booster_ids: np.ndarray | list[int],
        fueltank_ids: np.ndarray | list[int],
        total_fuel_mass: np.ndarray,
        max_fuel_rate: np.ndarray,
        max_inertia: np.ndarray,
        min_thrust: np.ndarray,
        max_thrust: np.ndarray,
        thrust_unit: np.ndarray,
        reignitable: np.ndarray | list[bool],
        booster_tau: np.ndarray,
        gimbal_tau: np.ndarray,
        gimbal_range_degrees: np.ndarray,
    ):
        """Used for simulating an array of boosters.

        Args:
            p (bullet_client.BulletClient): p
            physics_period (float): physics_period
            np_random (np.random.RandomState): np_random
            uav_id (int): uav_id
            booster_ids (list[int]): booster_ids
            fueltank_ids (list[int]): fueltank_ids
            total_fuel_mass (np.ndarray): total_fuel_mass
            max_fuel_rate (np.ndarray): max_fuel_rate
            max_inertia (np.ndarray): diagonal elements of the inertia tensor
            min_thrust (np.ndarray): min_thrust
            max_thrust (np.ndarray): max_thrust
            thrust_unit (np.ndarray):
            reignitable (list[bool]): whether we can turn off and on the booster
            booster_tau (np.ndarray): booster ramp time constant
            gimbal_tau (np.ndarray): gimbal ramp time constant
            gimbal_range_degrees (np.ndarray): gimbal_range_degrees
        """
        self.p = p
        self.physics_period = physics_period
        self.np_random = np_random

        # store IDs
        self.uav_id = uav_id
        self.booster_ids = booster_ids
        self.fueltank_ids = fueltank_ids

        # get number of motors and assert shapes
        self.num_boosters = len(booster_ids)
        assert len(fueltank_ids) == self.num_boosters
        assert total_fuel_mass.shape == (self.num_boosters,)
        assert max_fuel_rate.shape == (self.num_boosters,)
        assert max_inertia.shape == (self.num_boosters, 3)
        assert max_thrust.shape == (self.num_boosters,)
        assert thrust_unit.shape == (self.num_boosters, 3)
        assert len(reignitable) == self.num_boosters
        assert min_thrust.shape == (self.num_boosters,)
        assert booster_tau.shape == (self.num_boosters,)
        assert gimbal_tau.shape == (self.num_boosters,)
        assert gimbal_range_degrees.shape == (self.num_boosters,)

        # check that the thrust_axis is normalized
        if np.linalg.norm(thrust_unit) != 1.0:
            warnings.warn(f"Norm of `{thrust_unit=}` is not 1.0, normalizing...")
            thrust_unit /= np.linalg.norm(thrust_unit)

        # constants
        self.total_fuel_mass = total_fuel_mass
        self.max_fuel_rate = max_fuel_rate
        self.max_inertia = max_inertia
        self.max_thrust = max_thrust
        self.thrust_unit = thrust_unit
        self.reignitable = np.array(reignitable, dtype=bool)
        self.min_thrust = min_thrust
        self.booster_tau = booster_tau
        self.gimbal_tau = gimbal_tau
        self.gimbal_range_radians = np.deg2rad(gimbal_range_degrees)

        # some precomputed constants
        self.ratio_min_throttle = self.min_thrust / self.max_thrust
        self.ratio_throttleable = 1.0 - self.ratio_min_throttle
        self.ratio_fuel_rate = self.max_fuel_rate / self.total_fuel_mass

    def reset(self):
        """reset_motors."""
        # deal with everything in percents
        self.ratio_fuel_remaining = np.ones((self.num_boosters,), dtype=np.float64)
        self.throttle_setting = np.zeros((self.num_boosters,), dtype=np.float64)
        self.ignition_state = np.zeros((self.num_boosters,), dtype=bool)
        self.gimbal_state = np.zeros((self.num_boosters, 2), dtype=np.float64)

        # store the rotation matrix to make compute faster
        self.rotation1 = np.array([np.eye(3)] * self.num_boosters, dtype=np.float64)
        self.rotation2 = np.array([np.eye(3)] * self.num_boosters, dtype=np.float64)

    def settings2forces(
        self,
        ignition: np.ndarray,
        pwm: np.ndarray,
        gimbal_x: np.ndarray,
        gimbal_y: np.ndarray,
    ):
        """settings2forces.

        Args:
            ignition (np.ndarray): (num_boosters,) array of booleans for engine on or off
            pwm (np.ndarray): (num_boosters,) array of floats between [0, 1] for min or max thrust
            gimbal_x (np.ndarray): (num_boosters,) array of floats between [-1, 1]
            gimbal_y (np.ndarray): (num_boosters,) array of floats between [-1, 1]
        """
        (thrust, mass, inertia) = self._compute_thrust_mass_inertia(ignition, pwm)
        thrust_unit = self._compute_thrust_vector(gimbal_x, gimbal_y)

        # final thrust vector is unit vector * scalar
        thrust_vector = thrust_unit * thrust

        # apply the forces
        for i in range(self.num_boosters):
            self.p.applyExternalForce(
                self.uav_id,
                self.booster_ids[i],
                thrust_vector[i],
                [0.0, 0.0, 0.0],
                self.p.LINK_FRAME,
            )
            self.p.changeDynamics(
                self.uav_id,
                self.fueltank_ids[i],
                mass=mass[i],
                localInertiaDiagonal=inertia[i],
            )

    def _compute_thrust_mass_inertia(self, ignition: np.ndarray, pwm: np.ndarray):
        """_compute_thrust_mass_inertia.

        Args:
            ignition (np.ndarray): (num_boosters,) array of booleans for engine on or off
            pwm (np.ndarray): (num_boosters,) array of floats between [0, 1] for min or max thrust
        """
        # if not reignitable, logical or ignition_state with ignition
        # otherwise, just follow ignition
        self.ignition_state = ((not self.reignitable) & self.ignition_state) | bool(
            ignition
        )

        # target throttle depends on ignition status and pwm
        target_throttle = self.ignition_state * (
            pwm * self.ratio_throttleable + self.ratio_min_throttle
        )

        # model the booster using first order ODE, y' = T/tau * (setpoint - y)
        self.throttle_setting += (self.physics_period / self.booster_tau) * (
            target_throttle - self.throttle_setting
        )

        # if no fuel, hard cutoff
        self.throttle_setting *= self.ratio_fuel_remaining > 0.0

        # compute fuel remaining
        self.ratio_fuel_remaining -= (
            self.throttle_setting * self.ratio_fuel_rate * self.physics_period
        )

        # compute mass properties based on remaining fuel
        mass = self.ratio_fuel_remaining * self.total_fuel_mass
        inertia = self.ratio_fuel_remaining * self.max_inertia

        # compute thrust
        thrust = self.throttle_setting * self.max_thrust

        return thrust, mass, inertia

    def _compute_thrust_vector(self, gimbal_x: np.ndarray, gimbal_y: np.ndarray):
        """_compute_thrust_vector.

        Args:
            gimbal_x (np.ndarray): (num_boosters,) array of floats between [-1, 1]
            gimbal_y (np.ndarray): (num_boosters,) array of floats between [-1, 1]
        """
        # store the command as an (n, 2) array of floats
        gimbal_command = np.stack([gimbal_x, gimbal_y], axis=-1)

        # model the gimbal using first order ODE, y' = T/tau * (setpoint - y)
        self.gimbal_state += (self.physics_period / self.gimbal_tau) * (
            gimbal_command - self.gimbal_state
        )

        gimbal_angles = self.gimbal_state * self.gimbal_range_radians

        c = np.cos(gimbal_angles)
        s = np.sin(gimbal_angles)

        # start plugging things into the rotation matrices
        self.rotation1[:, 1, 1] = c[:, 0]
        self.rotation1[:, 2, 2] = c[:, 0]
        self.rotation1[:, 1, 2] = -s[:, 0]
        self.rotation1[:, 2, 1] = s[:, 0]
        self.rotation2[:, 0, 0] = c[:, 1]
        self.rotation2[:, 1, 1] = c[:, 1]
        self.rotation2[:, 0, 1] = -s[:, 1]
        self.rotation2[:, 1, 0] = s[:, 1]

        # get the final thrust vector
        return (
            self.rotation1 @ self.rotation2 @ np.expand_dims(self.thrust_unit, axis=-1)
        )
