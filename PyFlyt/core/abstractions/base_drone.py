"""Basic Drone class for all drone models to inherit from."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
from pybullet_utils import bullet_client

from .base_controller import ControlClass
from .camera import Camera


class DroneClass(ABC):
    """The `DroneClass` is an abstract class that all drones should inherit from.

    It serves as a template class for interfacing with the `Aviary`.
    Each drone inheriting from this class must have several attributes and methods implemented before they can be considered usable.

    Args:
        p (bullet_client.BulletClient): PyBullet physics client ID.
        start_pos (np.ndarray): an `(3,)` array for the starting X, Y, Z position for the drone.
        start_orn (np.ndarray): an `(3,)` array for the starting X, Y, Z orientation for the drone.
        control_hz (int): an integer representing the control looprate of the drone.
        physics_hz (int): an integer representing the physics looprate of the `Aviary`.
        drone_model (str): name of the drone itself, must be the same name as the folder where the URDF and YAML files are located.
        model_dir (None | str = None): directory where the drone model folder is located, if none is provided, defaults to the directory of the default drones.
        np_random (None | np.random.RandomState = None): random number generator of the simulation.

    Example Implementation:
        >>> def __init__(
        >>>     self,
        >>>     p: bullet_client.BulletClient,
        >>>     start_pos: np.ndarray,
        >>>     start_orn: np.ndarray,
        >>>     control_hz: int = 120,
        >>>     physics_hz: int = 240,
        >>>     drone_model: str = "rocket_brick",
        >>>     model_dir: None | str = os.path.dirname(os.path.realpath(__file__)),
        >>>     np_random: None | np.random.RandomState = None,
        >>>     use_camera: bool = False,
        >>>     use_gimbal: bool = False,
        >>>     camera_angle_degrees: int = 0,
        >>>     camera_FOV_degrees: int = 90,
        >>>     camera_resolution: tuple[int, int] = (128, 128),
        >>> ):
        >>>     super().__init__(
        >>>         p=p,
        >>>         start_pos=start_pos,
        >>>         start_orn=start_orn,
        >>>         control_hz=control_hz,
        >>>         physics_hz=physics_hz,
        >>>         model_dir=model_dir,
        >>>         drone_model=drone_model,
        >>>         np_random=np_random,
        >>>     )
        >>>     with open(self.param_path, "rb") as f:
        >>>         # load all params from yaml into components
        >>>         all_params = yaml.safe_load(f)
        >>>
        >>>         self.lifting_surfaces = LiftingSurfaces(...)
        >>>         self.boosters = Boosters(...)
        >>>
        >>>     self.use_camera = use_camera
        >>>     if self.use_camera:
        >>>         self.camera = Camera(...)
    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        control_hz: int,
        physics_hz: int,
        drone_model: str,
        model_dir: None | str = None,
        np_random: None | np.random.RandomState = None,
    ):
        """Defines the default configuration for UAVs, to be used in conjunction with the Aviary class.

        Args:
            p (bullet_client.BulletClient): PyBullet physics client ID.
            start_pos (np.ndarray): an `(3,)` array for the starting X, Y, Z position for the drone.
            start_orn (np.ndarray): an `(3,)` array for the starting X, Y, Z orientation for the drone.
            control_hz (int): an integer representing the control looprate of the drone.
            physics_hz (int): an integer representing the physics looprate of the `Aviary`.
            drone_model (str): name of the drone itself, must be the same name as the folder where the URDF and YAML files are located.
            model_dir (None | str = None): directory where the drone model folder is located, if none is provided, defaults to the directory of the default drones.
            np_random (None | np.random.RandomState = None): random number generator of the simulation.
        """
        if physics_hz % control_hz != 0:
            raise ValueError(
                f"`physics_hz` ({physics_hz}) must be multiple of `control_hz` ({control_hz})."
            )

        self.p = p
        self.np_random = np.random.RandomState() if np_random is None else np_random
        self.physics_control_ratio = int(physics_hz / control_hz)
        self.physics_period = 1.0 / physics_hz
        self.control_period = 1.0 / control_hz
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../../models/vehicles/"
            )
        self.drone_path = os.path.join(model_dir, f"{drone_model}/{drone_model}.urdf")
        self.param_path = os.path.join(model_dir, f"{drone_model}/{drone_model}.yaml")
        self.camera: Camera

        """DEFINE SPAWN"""
        self.start_pos = start_pos
        self.start_orn = self.p.getQuaternionFromEuler(start_orn)
        self.Id = self.p.loadURDF(
            self.drone_path,
            basePosition=self.start_pos,
            baseOrientation=self.start_orn,
            useFixedBase=False,
        )

        """DEFINE STATE AND SETPOINT"""
        self.state: np.ndarray
        self.aux_state: np.ndarray
        self.setpoint: np.ndarray

        """DEFINE CONTROLLERS"""
        self.registered_controllers: dict[int, type[ControlClass]] = dict()
        self.instanced_controllers: dict[int, ControlClass] = dict()
        self.registered_base_modes: dict[int, int] = dict()

        """DEFINE CAMERA IMAGES"""
        self.rgbaImg: np.ndarray
        self.depthImg: np.ndarray
        self.segImg: np.ndarray

    @abstractmethod
    def reset(self):
        """Resets the vehicle to the initial state.

        Example Implementation:
            >>> def reset(self):
            >>>     # set the flight mode and reset the setpoint and actuator command
            >>>     self.set_mode(0)
            >>>     self.setpoint = np.zeros(2)
            >>>     self.cmd = np.zeros(2)
            >>>
            >>>     # reset the drone position and orientation and damping
            >>>     self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
            >>>     self.disable_artificial_damping()
            >>>
            >>>     # reset the components to their default states
            >>>     self.lifting_surfaces.reset()
            >>>     self.boosters.reset()
        """
        raise NotImplementedError

    @abstractmethod
    def update_control(self):
        """Updates onboard flight control laws at a rate specified by `control_hz`.

        Example Implementation:
            >>> def update_control(self):
            >>>     # depending on the flight control mode, do things
            >>>     if self.mode == 0:
            >>>         # assign the actuator commands to be some function of the setpoint
            >>>         self.cmd = self.setpoint
            >>>         return
            >>>
            >>>     # otherwise, check that we have a custom controller
            >>>     if self.mode not in self.registered_controllers.keys():
            >>>         raise ValueError(
            >>>             f"Don't have other modes aside from 0, received {self.mode}."
            >>>         )
            >>>
            >>>     # run custom controllers if any
            >>>     self.cmd = self.instanced_controllers[self.mode].step(self.state, self.setpoint)
        """
        raise NotImplementedError

    @abstractmethod
    def update_physics(self):
        """Updates all physics on the vehicle at a rate specified by `physics_hz`.

        Example Implementation:
            >>> def update_physics(self):
            >>>     self.lifting_surfaces.physics_update(self.cmd[...])
            >>>     self.boosters.physics_update(self.cmd[...])
            >>>     ...
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(self):
        """Updates the vehicle's state values at a rate specified by `phyiscs_hz`.

        Example Implementation:
            >>> def update_state(self):
            >>>     # get all relevant information from PyBullet backend
            >>>     lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
            >>>     lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)
            >>>
            >>>     # express vels in local frame
            >>>     rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
            >>>     lin_vel = np.matmul(rotation, lin_vel)
            >>>     ang_vel = np.matmul(rotation, ang_vel)
            >>>
            >>>     # ang_pos in euler form
            >>>     ang_pos = self.p.getEulerFromQuaternion(ang_pos)
            >>>
            >>>     # create the state, this is accessed from the aviary
            >>>     self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)
            >>>
            >>>     # update all relevant components as required
            >>>     self.lifting_surfaces.state_update(rotation)
            >>>     ...
            >>>
            >>>     # update auxiliary information
            >>>     self.aux_state = np.concatenate(
            >>>         (self.lifting_surfaces.get_states(), self.boosters.get_states(), ...)
            >>>     )
        """
        raise NotImplementedError

    @abstractmethod
    def update_last(self):
        """Update that happens at the end of `Aviary.step()`, usually reserved for camera updates.

        Example Implementation:
            >>> def update_last(self):
            >>>     if self.use_camera:
            >>>         self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
        """
        raise NotImplementedError

    def set_mode(self, mode):
        """Default set_mode.

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
        controller_constructor: type[ControlClass],
        base_mode: int,
    ):
        """Default register_controller.

        Args:
            controller_id (int): ID to bind to this controller
            controller_constructor (type[ControlClass]): A class pointer to the controller implementation, must be subclass of `ControlClass`.
            base_mode (int): Whether this controller uses outputs of an underlying controller as setpoints.
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
        """Debugging function for displaying all joint IDs and names as defined in URDF."""
        # read out all infos
        infos = dict()
        for idx in range(self.p.getNumJoints(self.Id)):
            info = self.p.getJointInfo(self.Id, idx)
            infos[idx] = info[12]

        # add the base
        infos[-1] = "base"

        from pprint import pprint

        pprint(infos)

    def disable_artificial_damping(self):
        """Disable the artificial damping that pybullet has to enable more accurate aerodynamics simulation."""
        for idx in range(-1, self.p.getNumJoints(self.Id)):
            self.p.changeDynamics(self.Id, idx, linearDamping=0.0, angularDamping=0.0)
