import numpy as np


class PID:
    def __init__(
        self,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        limits: np.ndarray,
        period: float,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.limits = limits
        self.period = period

        # runtime variables
        self._integral = np.zeros_like(self.Kp)
        self._prev_error = np.zeros_like(self.Kp)

    def reset(self):
        self._integral *= 0.0
        self._prev_error *= 0.0

    def step(self, state: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
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
