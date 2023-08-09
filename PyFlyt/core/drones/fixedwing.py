"""Implementation of a Fixedwing UAV."""
from __future__ import annotations

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions.base_drone import DroneClass
from ..abstractions.camera import Camera
from ..abstractions.lifting_surfaces import LiftingSurface, LiftingSurfaces
from ..abstractions.motors import Motors


class Fixedwing(DroneClass):
    """Fixedwing instance that handles everything about a FixedWing."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        control_hz: int = 120,
        physics_hz: int = 240,
        drone_model: str = "fixedwing",
        model_dir: None | str = None,
        np_random: None | np.random.RandomState = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 0,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        camera_position_offset: np.ndarray = np.array([-3.0, 0.0, 1.0]),
        starting_velocity: np.ndarray = np.array([20.0, 0.0, 0.0]),
    ):
        """Creates a Fixedwing UAV and handles all relevant control and physics.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            control_hz (int): control_hz
            physics_hz (int): physics_hz
            drone_model (str): drone_model
            model_dir (None | str): model_dir
            np_random (None | np.random.RandomState): np_random
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
            camera_position_offset (np.ndarray): offset position of the camera
            starting_velocity (np.ndarray): vector representing the velocity at spawn
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            control_hz=control_hz,
            physics_hz=physics_hz,
            model_dir=model_dir,
            drone_model=drone_model,
            np_random=np_random,
        )

        # constants
        self.starting_velocity = starting_velocity

        """Reads fixedwing.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)

            # all lifting surfaces
            surfaces = list()
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=3,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["left_wing_flapped_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=4,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["right_wing_flapped_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=1,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["horizontal_tail_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=2,
                    lifting_unit=np.array([0.0, 1.0, 0.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["vertical_tail_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=5,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["main_wing_params"],
                )
            )
            self.lifting_surfaces = LiftingSurfaces(lifting_surfaces=surfaces)

            # mapping for RPYT -> LeftAil, RightAil, HorStab, VertStab, MainWing, Motor
            # signs for each control surface when under assist
            self.surface_assist_ids = np.array([0, 0, 1, 1, 2, 3])
            self.surface_assist_signs = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 1.0])

            # motor
            motor_params = all_params["motor_params"]
            tau = np.array([motor_params["tau"]])
            max_rpm = np.array([1.0]) * np.sqrt(
                (motor_params["total_thrust"]) / motor_params["thrust_coef"]
            )
            thrust_coef = np.array([motor_params["thrust_coef"]])
            torque_coef = np.array([motor_params["torque_coef"]])
            thrust_unit = np.array([[1.0, 0.0, 0.0]])
            noise_ratio = np.array([motor_params["noise_ratio"]])
            self.motors = Motors(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                motor_ids=[0],
                tau=tau,
                max_rpm=max_rpm,
                thrust_coef=thrust_coef,
                torque_coef=torque_coef,
                thrust_unit=thrust_unit,
                noise_ratio=noise_ratio,
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
                camera_position_offset=camera_position_offset,
                is_tracking_camera=True,
            )

    def reset(self):
        """Resets the vehicle to the initial state."""
        self.set_mode(0)
        self.setpoint: np.ndarray
        self.cmd = np.zeros(6)

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.p.resetBaseVelocity(self.Id, self.starting_velocity, [0, 0, 0])
        self.disable_artificial_damping()
        self.lifting_surfaces.reset()
        self.motors.reset()

    def set_mode(self, mode):
        """Sets the current flight mode of the vehicle.

        flight modes:
            - -1: Left Aileron, Right Aileron, Horizontal Tail, Vertical Tail, Main Wing, Thrust
            - 0: Pitch, Roll, Yaw, Thrust

        Args:
            mode (int): flight mode
        """
        if (mode < -1 or mode > 0) and mode not in self.registered_controllers.keys():
            raise ValueError(
                f"`mode` must be between -1 and 0 or be registered in {self.registered_controllers.keys()=}, got {mode}."
            )

        self.mode = mode

        if mode == -1:
            self.setpoint = np.zeros(6)
        elif mode == 0:
            self.setpoint = np.zeros(4)

    def update_control(self):
        """Runs through controllers."""
        # full control over all surfaces
        if self.mode == -1:
            self.cmd = self.setpoint
            return

        # the default mode
        elif self.mode == 0:
            self.cmd = (
                self.setpoint[self.surface_assist_ids] * self.surface_assist_signs
            )
            return

        # otherwise, check that we have a custom controller
        if self.mode not in self.registered_controllers.keys():
            raise ValueError(
                f"Don't have other modes aside from 0 and -1, received {self.mode}."
            )

        # custom controllers run if any
        self.cmd = self.instanced_controllers[self.mode].step(self.state, self.setpoint)

    def update_physics(self):
        """Updates the physics of the vehicle."""
        self.lifting_surfaces.physics_update(self.cmd[:-1])
        self.motors.physics_update(self.cmd[[5]])

    def update_state(self):
        """Updates the current state of the UAV.

        This includes: ang_vel, ang_pos, lin_vel, lin_pos.
        """
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
        self.lifting_surfaces.state_update(rotation)

        # update auxiliary information
        self.aux_state = np.concatenate(
            (self.lifting_surfaces.get_states(), self.motors.get_states())
        )

    def update_last(self):
        """Updates things only at the end of `Aviary.step()`."""
        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
