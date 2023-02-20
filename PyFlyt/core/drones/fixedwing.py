from __future__ import annotations

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions.base_drone import DroneClass
from ..abstractions.camera import Camera
from ..abstractions.lifting_surface import LiftingSurface
from ..abstractions.motors import Motors


class FixedWing(DroneClass):
    """FixedWing instance that handles everything about a FixedWing."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model: str = "fixedwing",
        model_dir: None | str = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 0,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        np_random: None | np.random.RandomState = None,
    ):
        """Creates a fixed wing UAV and handles all relevant control and physics.

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

            # all lifting surfaces
            self.lifting_surfaces: list[LiftingSurface] = []
            self.lifting_surfaces.append(
                LiftingSurface(
                    id=3,
                    command_id=1,
                    command_sign=+1.0,
                    z_axis_lift=True,
                    aerofoil_params=all_params["left_wing_flapped_params"],
                )
            )
            self.lifting_surfaces.append(
                LiftingSurface(
                    id=4,
                    command_id=1,
                    command_sign=-1.0,
                    z_axis_lift=True,
                    aerofoil_params=all_params["right_wing_flapped_params"],
                )
            )
            self.lifting_surfaces.append(
                LiftingSurface(
                    id=1,
                    command_id=0,
                    command_sign=-1.0,
                    z_axis_lift=True,
                    aerofoil_params=all_params["horizontal_tail_params"],
                )
            )
            self.lifting_surfaces.append(
                LiftingSurface(
                    id=5,
                    command_id=None,
                    command_sign=+1.0,
                    z_axis_lift=True,
                    aerofoil_params=all_params["main_wing_params"],
                )
            )
            self.lifting_surfaces.append(
                LiftingSurface(
                    id=2,
                    command_id=2,
                    command_sign=-1.0,
                    z_axis_lift=False,
                    aerofoil_params=all_params["vertical_tail_params"],
                )
            )
            self.surface_ids: list[int] = [
                surface.id for surface in self.lifting_surfaces
            ]

            # motor
            motor_params = all_params["motor_params"]
            tau = np.array([[motor_params["tau"]]])
            max_rpm = np.array([[1.0]]) * np.sqrt(
                (motor_params["thrust_to_weight"] * 9.81) / motor_params["thrust_coef"]
            )
            thrust_coef = np.array([[0.0, 1.0, 0.0]]) * motor_params["thrust_coef"]
            torque_coef = np.array([[0.0, 1.0, 0.0]]) * motor_params["torque_coef"]
            noise_ratio = np.array([[motor_params["noise_ratio"]]])
            self.motors = Motors(
                p=self.p,
                physics_period=self.physics_period,
                uav_id=self.Id,
                motor_ids=[0],
                tau=tau,
                max_rpm=max_rpm,
                thrust_coef=thrust_coef,
                torque_coef=torque_coef,
                noise_ratio=noise_ratio,
                np_random=self.np_random,
            )

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
                camera_position_offset=np.array([0.0, -3.0, 1.0]),
                is_tracking_camera=True,
                cinematic=True,
            )

        """ CUSTOM CONTROLLERS """
        # dictionary mapping of controller_id to controller objects
        self.registered_controllers = dict()
        self.instanced_controllers = dict()
        self.registered_base_modes = dict()

        self.reset()

    def update_forces(self):
        """Calculates and applies forces acting on UAV"""
        # Motor thrust
        self.motors.pwm2forces(self.cmd[[3]])

        for surface in self.lifting_surfaces:
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
        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

        # update all lifting surface velocities
        for surface in self.lifting_surfaces:
            surface_velocity = self.p.getLinkState(
                self.Id, surface.id, computeLinkVelocity=True
            )[-2]
            surface.update_local_surface_velocity(rotation, surface_velocity)

    def update_control(self):
        """runs through controllers"""
        # Final cmd, [Roll, Pitch, Yaw, Throttle] from [-1, 1]
        self.cmd = self.setpoint

    def reset(self):
        self.set_mode(1)
        self.setpoint = np.zeros((4))
        self.cmd = np.zeros((4))

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.p.resetBaseVelocity(self.Id, [0, 20, 0], [0, 0, 0])
        self.motors.reset()
        self.update_state()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()

    def set_mode(self, mode):
        """
        Mode 1 - [Roll, Pitch, Yaw, Throttle]
        """
        # WIP, copied and pasted from quadx
        if (mode < -1 or mode > 7) and mode not in self.registered_controllers.keys():
            raise ValueError(
                f"`mode` must be between -1 and 7 or be registered in {self.registered_controllers.keys()=}, got {mode}."
            )

        self.mode = mode

        # for custom modes
        if mode in self.registered_controllers.keys():
            self.instanced_controllers[mode] = self.registered_controllers[mode]()
            mode = self.registered_base_modes[mode]

    def update_physics(self):
        self.update_state()
        self.update_forces()

    def update_avionics(self):
        """
        updates state and control
        """
        self.update_control()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
