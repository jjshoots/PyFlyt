import os
from abc import ABC, abstractmethod

import numpy as np
from pybullet_utils import bullet_client


class CtrlClass(ABC):
    """Basic Controller class to implement custom controllers."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, state: np.ndarray, setpoint: np.ndarray):
        pass


class DroneClass(ABC):
    """Basic Drone class for all drone models to inherit from."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model: str,
        model_dir: None | str = None,
        np_random: None | np.random.RandomState = None,
    ):
        """DEFAULT CONFIGURATION FOR DRONES"""
        if physics_hz != 240.0:
            raise UserWarning(
                f"Physics_hz is currently {physics_hz}, not the 240.0 that is recommended by pybullet. There may be physics errors."
            )

        self.p = p
        self.np_random = np.random.RandomState() if np_random is None else np_random
        self.physics_hz = 1.0 / physics_hz
        self.ctrl_period = 1.0 / ctrl_hz
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../models/vehicles/"
            )
        self.drone_dir = os.path.join(model_dir, f"{drone_model}/{drone_model}.urdf")
        self.param_path = os.path.join(model_dir, f"{drone_model}/{drone_model}.yaml")

        """ SPAWN """
        self.start_pos = start_pos
        self.start_orn = self.p.getQuaternionFromEuler(start_orn)
        self.Id = self.p.loadURDF(
            self.drone_dir,
            basePosition=self.start_pos,
            baseOrientation=self.start_orn,
            useFixedBase=False,
        )

        """ DEFINE SETPOINT """
        self.setpoint = np.zeros((4,))

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_mode(self, mode):
        pass

    @abstractmethod
    def update_avionics(self):
        pass

    @abstractmethod
    def update_physics(self):
        pass
