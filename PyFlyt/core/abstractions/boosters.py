"""A component to simulate an array of boosters on vehicle."""
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
        fueltank_ids: np.ndarray | list[None | int],
        total_fuel_mass: np.ndarray,
        max_fuel_rate: np.ndarray,
        max_inertia: np.ndarray,
        min_thrust: np.ndarray,
        max_thrust: np.ndarray,
        thrust_unit: np.ndarray,
        reignitable: np.ndarray | list[bool],
        tau: np.ndarray,
    ):
        """Used for simulating an array of boosters.

        Args:
            p (bullet_client.BulletClient): p
            physics_period (float): physics_period
            np_random (np.random.RandomState): np_random
            uav_id (int): uav_id
            booster_ids (list[int]): booster_ids
            fueltank_ids (list[None | int]): fueltank_ids
            total_fuel_mass (np.ndarray): total_fuel_mass
            max_fuel_rate (np.ndarray): max_fuel_rate
            max_inertia (np.ndarray): diagonal elements of the inertia tensor
            min_thrust (np.ndarray): min_thrust
            max_thrust (np.ndarray): max_thrust
            thrust_unit (np.ndarray): unit vector of the direction thrust is pointing
            reignitable (list[bool]): whether we can turn off and on the booster
            tau (np.ndarray): booster ramp time constant
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
        assert tau.shape == (self.num_boosters,)

        # check that the thrust_axis is normalized
        if np.linalg.norm(thrust_unit) != 1.0:
            warnings.warn(f"Norm of `{thrust_unit=}` is not 1.0, normalizing...")
            thrust_unit /= np.linalg.norm(thrust_unit)

        # constants
        self.total_fuel_mass = total_fuel_mass
        self.max_fuel_rate = max_fuel_rate
        self.max_inertia = max_inertia
        self.max_thrust = max_thrust
        self.thrust_unit = np.expand_dims(thrust_unit, axis=-1)
        self.reignitable = np.array(reignitable, dtype=bool)
        self.min_thrust = min_thrust
        self.tau = tau
        self.ratio_min_throttle = self.min_thrust / self.max_thrust
        self.ratio_throttleable = 1.0 - self.ratio_min_throttle
        self.ratio_fuel_rate = self.max_fuel_rate / self.total_fuel_mass

    def reset(self):
        """Reset the boosters."""
        # deal with everything in percents
        self.ratio_fuel_remaining = np.ones((self.num_boosters,), dtype=np.float64)
        self.throttle = np.zeros((self.num_boosters,), dtype=np.float64)
        self.ignition_state = np.zeros((self.num_boosters,), dtype=bool)

    def get_states(self) -> np.ndarray:
        """Gets the current state of the components.

        Returns:
            np.ndarray:
        """
        return np.concatenate(
            [
                self.ignition_state.flatten(),  # [n]
                self.ratio_fuel_remaining.flatten(),  # [n]
                self.throttle.flatten(),  # [n]
            ]
        )

    def settings2forces(
        self, ignition: np.ndarray, pwm: np.ndarray, rotation: None | np.ndarray = None
    ):
        """Converts booster settings into forces on the booster and inertia change on fuel tank.

        Args:
            ignition (np.ndarray): (num_boosters,) array of booleans for engine on or off
            pwm (np.ndarray): (num_boosters,) array of floats between [0, 1] for min or max thrust
            rotation (np.ndarray): (num_boosters, 3, 3) rotation matrices to rotate each booster's thrust axis around
        """
        assert np.all(ignition >= 0.0) and np.all(
            ignition <= 1.0
        ), f"{ignition=} has values out of bounds of 0.0 and 1.0."
        assert np.all(pwm >= 0.0) and np.all(
            pwm <= 1.0
        ), f"{pwm=} has values out of bounds of 0.0 and 1.0."
        if rotation is not None:
            assert rotation.shape == (
                self.num_boosters,
                3,
                3,
            ), f"`rotation` should be of shape (num_boosters, 3, 3), got {rotation.shape}"

        # compute thrust mass inertia
        (thrust, mass, inertia) = self._compute_thrust_mass_inertia(ignition, pwm)

        # handle rotation
        thrust_unit = (
            self.thrust_unit if rotation is None else rotation @ self.thrust_unit
        )

        # final thrust vector is unit vector * scalar
        thrust_vector = thrust_unit * thrust

        # apply the forces and fueltanks
        for i in range(self.num_boosters):
            self.p.applyExternalForce(
                self.uav_id,
                self.booster_ids[i],
                thrust_vector[i],
                [0.0, 0.0, 0.0],
                self.p.LINK_FRAME,
            )

            # no need to update inertia if no fueltank
            if self.fueltank_ids[i] is None:
                continue

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
        if self.tau == 0.0:
            self.throttle = target_throttle
        else:
            self.throttle += (self.physics_period / self.tau) * (
                target_throttle - self.throttle
            )

        # if no fuel, hard cutoff
        self.throttle *= self.ratio_fuel_remaining > 0.0

        # compute fuel remaining, clip if less than 0
        self.ratio_fuel_remaining -= (
            self.throttle * self.ratio_fuel_rate * self.physics_period
        )
        self.ratio_fuel_remaining = np.clip(self.ratio_fuel_remaining, 0.0, 1.0)

        # compute mass properties based on remaining fuel
        mass = self.ratio_fuel_remaining * self.total_fuel_mass
        inertia = self.ratio_fuel_remaining * self.max_inertia

        # compute thrust
        thrust = self.throttle * self.max_thrust

        return thrust, mass, inertia
