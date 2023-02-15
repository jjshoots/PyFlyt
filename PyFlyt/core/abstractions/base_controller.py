from abc import ABC, abstractmethod

import numpy as np


class CtrlClass(ABC):
    """Basic Controller class to implement custom controllers."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, state: np.ndarray, setpoint: np.ndarray):
        pass
