"""A simple class implementing the PID algorithm that works on numpy arrays."""
import numba as nb
import numpy as np

_pid_spec = [
    ("kp", nb.float64[:]),
    ("ki", nb.float64[:]),
    ("kd", nb.float64[:]),
    ("limits", nb.float64[:]),
    ("period", nb.float64),
    ("_integral", nb.float64[:]),
    ("_prev_error", nb.float64[:]),
]


@nb.experimental.jitclass(_pid_spec)  # type: ignore [reportGeneralTypeIssues]
class PID:
    """PID."""

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        limits: np.ndarray,
        period: float,
    ):
        """Defines a simple PID controller that works on numpy arrays.

        This function is jitted to achieve 1.3x speedup.
        Because of this, all arguments passed into this function, except `period`, must be a 1D np array and have the same shape.

        Example:
            Invalid implementation:
            >>> controller = PID(0.5, 0.4, 0.3, 1.0, 0.01)
            >>> controller.step(5.0, 2.0)

            Valid implementation:
            >>> kp = np.array([0.5])
            >>> ki = np.array([0.4])
            >>> kd = np.array([0.3])
            >>> limits = np.array([1.0])
            >>> controller = PID(kp, ki, kd, 0.01)
            >>> controller.step(np.array([5.0]), np.array([2.0]))

        Args:
            kp (np.ndarray): kp
            ki (np.ndarray): ki
            kd (np.ndarray): kd
            limits (np.ndarray): limits
            period (float): period
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = limits
        self.period = period

        # runtime variables
        self._integral = np.zeros_like(self.kp)
        self._prev_error = np.zeros_like(self.kp)

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

        proportional = self.kp * error

        self._integral = np.clip(
            self._integral + self.ki * error * self.period, -self.limits, self.limits
        )

        derivative = self.kd * (error - self._prev_error) / self.period
        self._prev_error = error

        return np.clip(
            proportional + self._integral + derivative, -self.limits, self.limits
        )
