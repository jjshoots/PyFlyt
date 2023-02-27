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
        noise_ratio: np.ndarray,
    ):
        """Used for simulating an array of motors.

        Args:
            p (bullet_client.BulletClient): p
            physics_period (float): physics_period
            uav_id (int): uav_id
            motor_ids (list[int]): motor_ids
            tau (np.ndarray): motor ramp time constant
            max_rpm (np.ndarray): max_rpm
            thrust_coef (np.ndarray): thrust_coef
            torque_coef (np.ndarray): torque_coef
            noise_ratio (np.ndarray): noise_ratio
            np_random (np.random.RandomState): np_random
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
        assert thrust_coef.shape == (self.num_motors, 3)
        assert torque_coef.shape == (self.num_motors, 3)
        assert noise_ratio.shape == (self.num_motors,)

        # motor constants
        self.tau = np.expand_dims(tau, axis=-1)
        self.max_rpm = np.expand_dims(max_rpm, axis=-1)
        self.thrust_coef = thrust_coef
        self.torque_coef = torque_coef
        self.noise_ratio = np.expand_dims(noise_ratio, axis=-1)

    def reset(self):
        """reset the motors."""
        self.throttle = np.zeros((self.num_motors, 1))

    def get_states(self) -> np.ndarray:
        """Gets the current state of the components.

        Returns:
            np.ndarray:
        """
        return self.throttle.flatten()

    def pwm2forces(self, pwm):
        """pwm2forces.

        Args:
            pwm:
        """
        assert np.all(pwm >= -1.0) and np.all(
            pwm <= 1.0
        ), f"`{pwm=} has values out of bounds of -1.0 and 1.0.`"

        pwm = np.expand_dims(pwm, 1)

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

        # rpm to thrust and torque
        thrust = (rpm**2) * self.thrust_coef
        torque = (rpm**2) * self.torque_coef

        # apply the forces
        for idx, thr, tor in zip(self.motor_ids, thrust, torque):
            self.p.applyExternalForce(
                self.uav_id, idx, thr, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.uav_id, idx, tor, self.p.LINK_FRAME)
