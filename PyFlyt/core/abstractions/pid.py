"""A simple class implementing the PID algorithm that works on numpy arrays."""
import numpy as np


class PID:
    """PID."""

    def __init__(
        self,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        limits: np.ndarray,
        period: float,
    ):
        """Defines a simple PID controller that works on numpy arrays.

        Kp, Ki, and Kd must be equal shaped numpy arrays.

        Args:
            Kp (np.ndarray): Kp
            Ki (np.ndarray): Ki
            Kd (np.ndarray): Kd
            limits (np.ndarray): limits
            period (float): period
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.limits = limits
        self.period = period

        # runtime variables
        self._integral = np.zeros_like(self.Kp)
        self._prev_error = np.zeros_like(self.Kp)

    def reset(self):
        """Resets the internal state of the PID controller."""
        self._integral *= 0.0
        self._prev_error *= 0.0

    def step(self, state: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        """Steps the PID controller.

        Args:
            state (np.ndarray): the state of the system, must be same shape as controller parameters.
            setpoint (np.ndarray): the setpoint of the system, must be same shape as controller parameters.

        Returns:
            np.ndarray:
        """
        error = setpoint - state

        self._integral = np.clip(
            self._integral + self.Ki * error * self.period, -self.limits, self.limits
        )

        derivative = self.Kd * (error - self._prev_error) / self.period
        self._prev_error = error

        proportional = self.Kp * error

        return np.clip(
            proportional + self._integral + derivative, -self.limits, self.limits
        )
