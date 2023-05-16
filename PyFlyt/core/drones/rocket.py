"""Implementation of a 1:10 scale SpaceX Rocket UAV."""
from __future__ import annotations

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions.base_drone import DroneClass
from ..abstractions.boosters import Boosters
from ..abstractions.boring_bodies import BoringBodies
from ..abstractions.camera import Camera
from ..abstractions.gimbals import Gimbals
from ..abstractions.lifting_surfaces import LiftingSurface, LiftingSurfaces


class Rocket(DroneClass):
    """Rocket instance that handles everything about a thrust vectored rocket with throttleable boosters and controllable finlets.

    The setpoint for this model has 7 values:
        - finlet x deflection
        - finlet y deflection
        - finlet yaw
        - ignition
        - throttle
        - booster gimbal axis 1
        - booster gimbal axis 2
    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        control_hz: int = 120,
        physics_hz: int = 240,
        drone_model: str = "rocket",
        model_dir: None | str = None,
        np_random: None | np.random.RandomState = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 30,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        camera_position_offset: np.ndarray = np.array([-1.0, 0.0, 3.0]),
        starting_fuel_ratio: float = 0.05,
    ):
        """Creates a drone in the QuadX configuration and handles all relevant control and physics.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            physics_hz (int): physics_hz
            control_hz (int): control_hz
            drone_model (str): drone_model
            model_dir (None | str): model_dir
            np_random (None | np.random.RandomState): np_random
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
            camera_position_offset (np.ndarray): offset position of the camera
            starting_fuel_ratio (float): amount of fuel that the rocket has to beginwith
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
        self.starting_fuel_ratio = starting_fuel_ratio

        """Reads fixedwing.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)
            booster_params = all_params["booster_params"]
            body_params = all_params["body_params"]

            # add the main body
            self.bodies = BoringBodies(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                body_ids=np.array([0]),
                drag_coefs=np.array(
                    [
                        [
                            body_params["drag_coef_x"],
                            body_params["drag_coef_y"],
                            body_params["drag_coef_z"],
                        ]
                    ]
                ),
                normal_areas=np.array(
                    [
                        [
                            body_params["area_x"],
                            body_params["area_y"],
                            body_params["area_z"],
                        ]
                    ]
                ),
            )

            # add all finlets
            surfaces = list()
            for finlet_id, command_id in zip([2, 3], [0, 1]):
                # x axis fins
                surfaces.append(
                    LiftingSurface(
                        p=self.p,
                        physics_period=self.physics_period,
                        np_random=self.np_random,
                        uav_id=self.Id,
                        surface_id=finlet_id,
                        command_id=command_id,
                        command_sign=+1.0,
                        lifting_unit=np.array([0.0, 1.0, 0.0]),
                        forward_unit=np.array([0.0, 0.0, -1.0]),
                        **all_params["finlet_params"],
                    )
                )
            for finlet_id, command_id in zip([4, 5], [2, 3]):
                # y axis fins
                surfaces.append(
                    LiftingSurface(
                        p=self.p,
                        physics_period=self.physics_period,
                        np_random=self.np_random,
                        uav_id=self.Id,
                        surface_id=finlet_id,
                        command_id=command_id,
                        command_sign=+1.0,
                        lifting_unit=np.array([1.0, 0.0, 0.0]),
                        forward_unit=np.array([0.0, 0.0, -1.0]),
                        **all_params["finlet_params"],
                    )
                )
            self.lifting_surfaces = LiftingSurfaces(lifting_surfaces=surfaces)

            # mixing matrix to map finlet force command to finlet movement
            # force_x, force_y, yaw
            self.finlet_map = np.array(
                [
                    [+0.0, +1.0, +1.0],  # pos_x fin
                    [+0.0, +1.0, -1.0],  # neg_x fin
                    [+1.0, +0.0, -1.0],  # pos_y fin
                    [+1.0, +0.0, +1.0],  # neg_y fin
                ]
            )

            # add the booster
            self.boosters = Boosters(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                booster_ids=np.array([1], dtype=int),
                fueltank_ids=np.array([0], dtype=int),
                tau=np.array([booster_params["booster_tau"]]),
                total_fuel_mass=np.array([booster_params["total_fuel"]]),
                max_fuel_rate=np.array([booster_params["max_fuel_rate"]]),
                max_inertia=np.array(
                    [
                        [
                            booster_params["inertia_ixx"],
                            booster_params["inertia_iyy"],
                            booster_params["inertia_izz"],
                        ]
                    ]
                ),
                min_thrust=np.array([booster_params["min_thrust"]]),
                max_thrust=np.array([booster_params["max_thrust"]]),
                thrust_unit=np.array([[0.0, 0.0, 1.0]]),
                reignitable=np.array([booster_params["reignitable"]], dtype=bool),
                noise_ratio=np.array([booster_params["noise_ratio"]]),
            )

            # add the gimbal for the booster
            self.booster_gimbal = Gimbals(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                gimbal_unit_1=np.array([[1.0, 0.0, 0.0]]),
                gimbal_unit_2=np.array([[0.0, 1.0, 0.0]]),
                gimbal_tau=np.array([booster_params["gimbal_tau"]]),
                gimbal_range_degrees=np.array(
                    [[booster_params["gimbal_range_degrees"]] * 2]
                ),
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
        self.setpoint = np.zeros(7)
        self.cmd = np.zeros(8)

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.disable_artificial_damping()
        self.bodies.reset()
        self.lifting_surfaces.reset()
        self.booster_gimbal.reset()
        self.boosters.reset(starting_fuel_ratio=self.starting_fuel_ratio)

    def set_mode(self, mode):
        """Sets the current flight mode of the vehicle.

        flight modes:
            - 0: finlet x deflection, finlet y deflection, finlet yaw, ignition, throttle, booster gimbal axis 1, booster gimbal axis 2

        Args:
            mode (int): flight mode
        """
        super().set_mode(mode)

    def update_control(self):
        """Runs through controllers."""
        # the default mode
        if self.mode == 0:
            # finlet mapping
            finlet_cmd = self.finlet_map @ np.expand_dims(self.setpoint[:3], axis=-1)
            finlet_cmd = np.clip(finlet_cmd, -1.0, 1.0)

            # prepend the finlet mapping to the command itself
            self.cmd = np.concatenate((finlet_cmd.flatten(), self.setpoint[3:]))
            return

        # otherwise, check that we have a custom controller
        if self.mode not in self.registered_controllers.keys():
            raise ValueError(
                f"Don't have other modes aside from 0, received {self.mode}."
            )

        # custom controllers run if any
        self.cmd = self.instanced_controllers[self.mode].step(self.state, self.setpoint)

    def update_physics(self):
        """Updates the physics of the vehicle."""
        # update the forces on the main body
        self.bodies.physics_update()

        # actuate lifting surfaces
        self.lifting_surfaces.physics_update(self.cmd)

        # move the booster gimbal
        rotation = self.booster_gimbal.compute_rotation(
            np.array([self.cmd[6], self.cmd[7]])
        )

        # update booster
        self.boosters.physics_update(
            ignition=self.cmd[[4]],
            pwm=self.cmd[[5]],
            rotation=rotation,
        )

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

        # update all bodies, which is just the booster here
        self.bodies.state_update(rotation)

        # update all lifting surface velocities
        self.lifting_surfaces.state_update(rotation)

        # update auxiliary information
        self.aux_state = np.concatenate(
            (
                self.lifting_surfaces.get_states(),
                self.boosters.get_states(),
                self.booster_gimbal.get_states(),
            )
        )

    def update_last(self):
        """Updates things only at the end of `Aviary.step()`."""
        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
