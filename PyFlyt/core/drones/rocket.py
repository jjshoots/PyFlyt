from __future__ import annotations

import math

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions.base_drone import DroneClass
from ..abstractions.camera import Camera
from ..abstractions.lifting_surface import LiftingSurface


class Rocket(DroneClass):
    """Rocket instance that handles everything about a thrust vectored rocket with mildly throttleable boosters."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model: str = "rocket",
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

        """Reads fixedwing.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)
            booster_params = all_params["booster_params"]

            # add all finlets
            self.lifting_surfaces = []
            for finlet_id in [1, 2]:
                # pitching fins
                self.lifting_surfaces.append(
                    LiftingSurface(
                        id=finlet_id,
                        command_id=0,
                        command_sign=+1.0,
                        lifting_axis="+y",
                        forward_axis="-z",
                        aerofoil_params=all_params["finlet_params"],
                    )
                )
            # for finlet_id in [3, 4]:
            #     # rolling fins
            #     self.lifting_surfaces.append(
            #         LiftingSurface(
            #             id=finlet_id,
            #             command_id=1,
            #             command_sign=+1.0,
            #             lifting_axis="+y",
            #             forward_axis="-z",
            #             aerofoil_params=all_params["finlet_params"],
            #         )
            #     )

        """ CAMERA """
        self.use_camera = use_camera
        if self.use_camera:
            self.camera = Camera(
                p=self.p,
                uav_id=self.Id,
                camera_id=0,
                use_gimbal=use_gimbal,
                camera_FOV_degrees=camera_FOV_degrees,
                camera_angle_degrees=camera_angle_degrees,
                camera_resolution=camera_resolution,
            )

        """ CUSTOM CONTROLLERS """
        # dictionary mapping of controller_id to controller objects
        self.registered_controllers = dict()
        self.instanced_controllers = dict()
        self.registered_base_modes = dict()

        self.reset()

    def reset(self):
        """reset."""
        self.set_mode(0)
        self.setpoint = np.zeros((4))
        self.pwm = np.zeros((4))

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.update_state()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()

    def update_state(self):
        """ang_vel, ang_pos, lin_vel, lin_pos"""
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)

        # create the state
        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

        # update all lifting surface velocities
        for surface in self.lifting_surfaces:
            surface_velocity = self.p.getLinkState(
                self.Id, surface.id, computeLinkVelocity=True
            )[-2]
            surface.update_local_surface_velocity(rotation, surface_velocity)

    def update_control(self):
        """runs through controllers"""
        # the default mode
        if self.mode == 0:
            self.cmd = self.setpoint
            return

        # otherwise, check that we have a custom controller
        if self.mode not in self.registered_controllers.keys():
            raise ValueError(
                f"Don't have other modes aside from 0, received {self.mode}."
            )

        # custom controllers run if any
        self.cmd = self.instanced_controllers[self.mode].step(self.state, self.setpoint)

    def update_forces(self):
        """Calculates and applies forces acting on Rocket"""
        for surface in self.lifting_surfaces[0:1]:
            actuation = (
                0.0
                if surface.command_id is None
                else float(self.cmd[surface.command_id] * surface.command_sign)
            )

            force, torque = surface.compute_force_torque(actuation)

            self.p.applyExternalForce(
                self.Id,
                surface.id,
                force,
                [0.0, 0.0, 0.0],
                self.p.LINK_FRAME,
            )
            self.p.applyExternalTorque(self.Id, surface.id, torque, self.p.LINK_FRAME)

    def update_physics(self):
        """update_physics."""
        self.update_state()
        self.update_forces()

    def update_avionics(self):
        """
        updates state and control
        """
        self.update_control()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
