from __future__ import annotations

import math

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions import CtrlClass, DroneClass
from ..pid import PID


class FixedWing(DroneClass):
    """FixedWing instance that handles everything about a FixedWing."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model: str = "FW_UAV",
        model_dir: None | str = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 20,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        np_random: None | np.random.RandomState = None,
    ):
        """Creates a drone in the QuadX configuration and handles all relevant control and physics.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            ctrl_hz (int): ctrl_hz
            physics_hz (int): physics_hz
            model_dir (None | str): model_dir
            drone_model (str): drone_model
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
            np_random (None | np.random.RandomState): np_random
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            ctrl_hz=ctrl_hz,
            physics_hz=physics_hz,
            model_dir=model_dir,
            drone_model=drone_model,
            np_random=np_random,
        )

    def reset(self):
        pass

    def set_mode(self, mode):
        pass

    def update_avionics(self):
        pass

    def update_physics(self):
        pass
