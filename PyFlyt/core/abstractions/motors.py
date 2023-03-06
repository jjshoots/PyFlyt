"""A component to simulate an array of electric propeller motors on vehicle."""
from __future__ import annotations

import numpy as np
from pybullet_utils import bullet_client


class Motors:
    """Motors."""

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
            p (bullet_client.BulletClient): p
            physics_period (float): physics_period
            np_random (np.random.RandomState): np_random
            uav_id (int): uav_id
            motor_ids (list[int]): motor_ids
            tau (np.ndarray): motor ramp time constant
            max_rpm (np.ndarray): max_rpm
            thrust_coef (np.ndarray): thrust_coef
            torque_coef (np.ndarray): torque_coef
            thrust_unit (np.ndarray): axis on which the thrust acts on
            noise_ratio (np.ndarray): noise_ratio
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
            np.ndarray:
        """
        return self.throttle.flatten()

    def pwm2forces(self, pwm: np.ndarray, rotation: None | np.ndarray = None):
        """Converts motor PWM values to forces, this motor allows negative thrust.

        Args:
            pwm (np.ndarray): [num_motors, ] array defining the pwm values of each motor from -1 to 1
            rotation (np.ndarray): (num_motors, 3, 3) rotation matrices to rotate each booster's thrust axis around
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

        # throttle to rpm
        rpm = self.throttle * self.max_rpm
        rpm = np.expand_dims(rpm, axis=-1)

        # handle rotation
        thrust_unit = (
            self.thrust_unit if rotation is None else rotation @ self.thrust_unit
        ).squeeze(axis=-1)

        # rpm to thrust and torque
        thrust = (rpm**2) * self.thrust_coef * thrust_unit
        torque = (rpm**2) * self.torque_coef * thrust_unit

        # apply the forces
        for idx, thr, tor in zip(self.motor_ids, thrust, torque):
            self.p.applyExternalForce(
                self.uav_id, idx, thr, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.uav_id, idx, tor, self.p.LINK_FRAME)
