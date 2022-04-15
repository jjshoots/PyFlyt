import numpy as np


class PID:
    def __init__(self, Kp, Ki, Kd, limits, period):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.limits = limits

        self.period = period

        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, state, setpoint):
        error = setpoint - state

        self.integral = np.clip(
            self.integral + self.Ki * error * self.period, -self.limits, self.limits
        )

        derivative = self.Kd * (error - self.prev_error) / self.period
        self.prev_error = error

        proportional = self.Kp * error

        return np.clip(
            proportional + self.integral + derivative, -self.limits, self.limits
        )
