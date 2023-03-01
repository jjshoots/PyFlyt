"""Defines a basic controller class to inherit from when building custom controllers."""
from abc import ABC, abstractmethod

import numpy as np


class CtrlClass(ABC):
    """Basic Controller class to implement custom controllers."""

    @abstractmethod
    def reset(self):
        """Reset the internal state of the controller."""
        pass

    @abstractmethod
    def step(self, state: np.ndarray, setpoint: np.ndarray):
        """Step the controller.

        Args:
            state (np.ndarray): state
            setpoint (np.ndarray): setpoint
        """
        pass
