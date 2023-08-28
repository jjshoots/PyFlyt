"""A component to simulate an array of electric propeller motors on vehicle."""
from __future__ import annotations

import warnings

import numpy as np
from pybullet_utils import bullet_client

from PyFlyt.core.utils.compile_helpers import jitter


class Motors:
    """Simulates an array of brushless motor driven propellers.

    The `Motors` component is used to simulate a series of brushless motor driven propellers at arbitrary locations of the drone.
    Counter rotating motors (producing reversed torque) can be represented using negative `torque_coef` values.
    The maximum RPM can be easily computed using `max_rpm = max_thrust / thrust_coef`.

    Args:
        p (bullet_client.BulletClient): PyBullet physics client ID.
        physics_period (float): physics period of the simulation.
        np_random (np.random.RandomState): random number generator of the simulation.
        uav_id (int): ID of the drone.
        motor_ids (list[int]): a (n,) list of integers representing the link IDs for n motors.
        tau (np.ndarray): an (n,) of floats array representing the ramp time constant of each motor.
        max_rpm (np.ndarray): an (n,) array of floats representing the maximum RPM of each motor.
        thrust_coef (np.ndarray): an (n,) array of floats representing all motor thrust coefficients.
        torque_coef (np.ndarray): an (n,) array of floats representing all motor torque coefficients, uses right hand rotation rule around the `thrust_unit` axis.
        thrust_unit (np.ndarray): an (n, 3) array of floats representing n unit vectors along with the thrust of each motor acts.
        noise_ratio (np.ndarray): an (n,) array of floats representing the ratio amount of noise fluctuation present in each motor.
    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        physics_period: float,
        np_random: np.random.RandomState,
        uav_id: np.ndarray | int,
        motor_ids: np.ndarray | list[int],
        tau: np.ndarray,
        max_rpm: np.ndarray,
        thrust_coef: np.ndarray,
        torque_coef: np.ndarray,
        thrust_unit: np.ndarray,
        noise_ratio: np.ndarray,
    ):
        """Used for simulating an array of motors.

        Args:
            p (bullet_client.BulletClient): PyBullet physics client ID.
            physics_period (float): physics period of the simulation.
            np_random (np.random.RandomState): random number generator of the simulation.
            uav_id (int): ID of the drone.
            motor_ids (list[int]): a (n,) list of integers representing the link IDs for n motors.
            tau (np.ndarray): an (n,) of floats array representing the ramp time constant of each motor.
            max_rpm (np.ndarray): an (n,) array of floats representing the maximum RPM of each motor.
            thrust_coef (np.ndarray): an (n,) array of floats representing all motor thrust coefficients.
            torque_coef (np.ndarray): an (n,) array of floats representing all motor torque coefficients, uses right hand rotation rule around the `thrust_unit` axis.
            thrust_unit (np.ndarray): an (n, 3) array of floats representing n unit vectors along with the thrust of each motor acts.
            noise_ratio (np.ndarray): an (n,) array of floats representing the ratio amount of noise fluctuation present in each motor.
        """
        self.p = p
        self.physics_period = physics_period
        self.np_random = np_random

        # store IDs
        self.uav_id = uav_id
        self.motor_ids = motor_ids

        # get number of motors and assert shapes
        self.num_motors = len(motor_ids)
        assert tau.shape == (self.num_motors,)
        assert max_rpm.shape == (self.num_motors,)
        assert thrust_coef.shape == (self.num_motors,)
        assert torque_coef.shape == (self.num_motors,)
        assert thrust_unit.shape == (self.num_motors, 3)
        assert noise_ratio.shape == (self.num_motors,)
        assert all(
            tau >= 0.0 / physics_period
        ), f"Setting `tau = 1 / physics_period` is equivalent to 0, 0 is not a valid option, got {tau}."

        # motor constants
        self.tau = tau
        self.max_rpm = max_rpm
        self.thrust_coef = np.expand_dims(thrust_coef, axis=-1)
        self.torque_coef = np.expand_dims(torque_coef, axis=-1)
        self.thrust_unit = np.expand_dims(thrust_unit, axis=-1)
        self.noise_ratio = noise_ratio

    def reset(self):
        """Reset the motors."""
        self.throttle = np.zeros((self.num_motors,))

    def get_states(self) -> np.ndarray:
        """Gets the current state of the components.

        Returns:
            np.ndarray: an (num_motors, ) array for the current throttle level of each motor
        """
        return self.throttle.flatten()

    def state_update(self):
        """This does not need to be called for motors."""
        warnings.warn("`state_update` does not need to be called for motors.")

    def physics_update(self, pwm: np.ndarray, rotation: None | np.ndarray = None):
        """Converts motor PWM values to forces, this motor allows negative thrust.

        Args:
            pwm (np.ndarray): [num_motors, ] array defining the pwm values of each motor from -1 to 1.
            rotation (np.ndarray): (num_motors, 3, 3) rotation matrices to rotate each booster's thrust axis around, this is readily obtained from the `gimbals` component.
        """
        assert np.all(pwm >= -1.0) and np.all(
            pwm <= 1.0
        ), f"`{pwm=} has values out of bounds of -1.0 and 1.0.`"
        if rotation is not None:
            assert rotation.shape == (
                self.num_motors,
                3,
                3,
            ), f"`rotation` should be of shape (num_motors, 3, 3), got {rotation.shape}"

        # model the motor using first order ODE, y' = T/tau * (setpoint - y)
        self.throttle += (self.physics_period / self.tau) * (pwm - self.throttle)

        # noise in the motor
        self.throttle += (
            self.np_random.randn(*self.throttle.shape)
            * self.throttle
            * self.noise_ratio
        )

        # compute thrust and torque in jitted manner
        (thrust, torque) = self._jitted_compute_thrust_torque(
            rotation,
            self.throttle,
            self.max_rpm,
            self.thrust_unit,
            self.thrust_coef,
            self.torque_coef,
        )

        # apply the forces
        for idx, thr, tor in zip(self.motor_ids, thrust, torque):
            self.p.applyExternalForce(
                self.uav_id, idx, thr, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.uav_id, idx, tor, self.p.LINK_FRAME)

    @staticmethod
    @jitter
    def _jitted_compute_thrust_torque(
        rotation: None | np.ndarray,
        throttle: np.ndarray,
        max_rpm: np.ndarray,
        thrust_unit: np.ndarray,
        thrust_coef: np.ndarray,
        torque_coef: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the thrusts and torques for the array of motors.

        Args:
            rotation (None | np.ndarray): rotation
            throttle (np.ndarray): throttle from self
            max_rpm (np.ndarray): max_rpm from self
            thrust_unit (np.ndarray): thrust_unit from self
            thrust_coef (np.ndarray): thrust_coef from self
            torque_coef (np.ndarray): torque_coef from self

        Returns:
            tuple[np.ndarray, np.ndarray]:
        """
        # throttle to rpm
        rpm = throttle * max_rpm
        rpm = np.expand_dims(rpm, axis=-1)

        # handle rotation, `[..., 0]` is basically squeeze but numba friendly
        if rotation is not None:
            thrust_unit = (rotation @ thrust_unit)[..., 0]
        else:
            thrust_unit = thrust_unit[..., 0]

        # rpm to thrust and torque
        thrust = (rpm**2) * thrust_coef * thrust_unit
        torque = (rpm**2) * torque_coef * thrust_unit

        return thrust, torque
