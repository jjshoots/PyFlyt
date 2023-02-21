import os
from abc import ABC, abstractmethod
from pprint import pprint

import numpy as np
from pybullet_utils import bullet_client

from .base_controller import CtrlClass
from .camera import Camera


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
        self.physics_period = 1.0 / physics_hz
        self.ctrl_period = 1.0 / ctrl_hz
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../../models/vehicles/"
            )
        self.drone_dir = os.path.join(model_dir, f"{drone_model}/{drone_model}.urdf")
        self.param_path = os.path.join(model_dir, f"{drone_model}/{drone_model}.yaml")
        self.camera: Camera

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
        self.state: np.ndarray
        self.setpoint: np.ndarray

        """ CUSTOM CONTROLLERS """
        # dictionary mapping of controller_id to controller objects
        self.registered_controllers = dict()
        self.instanced_controllers = dict()
        self.registered_base_modes = dict()

    @abstractmethod
    def reset(self):
        """reset."""
        pass

    @abstractmethod
    def update_avionics(self):
        """update_avionics."""
        pass

    @abstractmethod
    def update_physics(self):
        """update_physics."""
        pass

    def set_mode(self, mode):
        """
        Default set_mode.

        By default, mode 0 defines the following setpoint behaviour:
        Mode 0 - [Pitch, Roll, Yaw, Thrust]
        """
        if (mode != 0) and (mode not in self.registered_controllers.keys()):
            raise ValueError(
                f"`mode` must be either 0 or be registered in {self.registered_controllers.keys()=}, got {mode}."
            )

        self.mode = mode

        # for custom modes
        if mode in self.registered_controllers.keys():
            self.instanced_controllers[mode] = self.registered_controllers[mode]()
            mode = self.registered_base_modes[mode]

    def register_controller(
        self,
        controller_id: int,
        controller_constructor: type[CtrlClass],
        base_mode: int,
    ):
        """
        Default register_controller.

        Args:
            controller_id (int): controller_id
            controller_constructor (type[CtrlClass]): controller_constructor
            base_mode (int): base_mode
        """
        assert (
            controller_id > 0
        ), f"`controller_id` must be more than 0, got {controller_id}."
        assert (
            base_mode == 0
        ), f"`base_mode` must be 0, no other controllers available, got {base_mode}."
        self.registered_controllers[controller_id] = controller_constructor
        self.registered_base_modes[controller_id] = base_mode

    def get_joint_info(self):
        """Debugging function for displaying all joint ids and names as defined in urdf."""
        # read out all infos
        infos = dict()
        for idx in range(self.p.getNumJoints(self.Id)):
            info = self.p.getJointInfo(self.Id, idx)
            infos[idx] = info[12]

        # add the base
        infos[-1] = "base"

        pprint(infos)
