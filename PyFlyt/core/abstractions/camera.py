"""A component to simulate a camera on a vehicle."""
from __future__ import annotations

import math
import warnings

import numpy as np
from pybullet_utils import bullet_client

from PyFlyt.core.utils.compile_helpers import check_numpy


class Camera:
    """A camera component.

    The `Camera` component simulates a camera attached to a link on the drone.
    The camera itself can be gimballed to achieve a horizon lock effect.
    In addition, the field-of-view (FOV), tilt angle, resolution, and offset distance from the main link can be adjusted.
    On image capture, the camera returns an RGBA image, a depth map, and a segmentation map with pixel values representing the IDs of objects in the environment.

    Args:
        p (bullet_client.BulletClient): PyBullet physics client ID.
        uav_id (int): ID of the drone.
        camera_id (int): integer representing the ID of the link that the camera is attached to.
        use_gimbal (bool): whether to lock the horizon of the camera.
        camera_FOV_degrees (float): the field-of-view of the camera in degrees.
        camera_angle_degrees (float): when gimballed, this is the angle of downtilt from horizon; when not gimballed, this is theh angle of uptile from horizon.
        camera_resolution (tuple[int, int]): the resolution of the camera in terms of [height, width].
        camera_position_offset (np.ndarray = np.array([0.0, 0.0, 0.0])): an (3,) array representing an offset of where the camera is from the center of the link in `camera_id`.
        is_tracking_camera (bool = False): if the camera is a tracking camera, the focus point of the camera is adjusted to focus on the center body of the aircraft instead of at infinity.
        cinematic (bool = False): it's not a bug, it's a feature.
    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        uav_id: int,
        camera_id: int,
        use_gimbal: bool,
        camera_FOV_degrees: float,
        camera_angle_degrees: float,
        camera_resolution: tuple[int, int],
        camera_position_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
        is_tracking_camera: bool = False,
        cinematic: bool = False,
    ):
        """Used for implementing camera modules.

        Args:
            p (bullet_client.BulletClient): PyBullet physics client ID.
            uav_id (int): ID of the drone.
            camera_id (int): integer representing the ID of the link that the camera is attached to.
            use_gimbal (bool): whether to lock the horizon of the camera.
            camera_FOV_degrees (float): the field-of-view of the camera in degrees.
            camera_angle_degrees (float): when gimballed, this is the angle of downtilt from horizon; when not gimballed, this is theh angle of uptile from horizon.
            camera_resolution (tuple[int, int]): the resolution of the camera in terms of [width, height].
            camera_position_offset (np.ndarray = np.array([0.0, 0.0, 0.0])): an (3,) array representing an offset of where the camera is from the center of the link in `camera_id`.
            is_tracking_camera (bool = False): if the camera is a tracking camera, the focus point of the camera is adjusted to focus on the center body of the aircraft instead of at infinity.
            cinematic (bool = False): it's not a bug, it's a feature.
        """
        check_numpy()
        if is_tracking_camera and use_gimbal:
            warnings.warn(
                "Use_gimbal and is_tracking_camera are both enabled. This will lead to funky behaviour."
            )

        # grab the pybullet client instance and relevant IDs
        self.p = p
        self.uav_id = uav_id
        self.camera_id = camera_id

        # camera parameters
        self.proj_mat = self.p.computeProjectionMatrixFOV(
            fov=camera_FOV_degrees,
            aspect=float(camera_resolution[1] / camera_resolution[0]),
            nearVal=0.1,
            farVal=255.0,
        )
        self.use_gimbal = use_gimbal
        self.camera_angle_degrees = camera_angle_degrees
        self.camera_FOV_degrees = camera_FOV_degrees
        self.camera_resolution = np.array(camera_resolution)

        # handle camera offset
        self.camera_position_offset = camera_position_offset
        self.is_tracking_camera = is_tracking_camera
        if np.sum(np.abs(self.camera_position_offset)) != 0.0:
            self.has_camera_offset = True
        else:
            self.has_camera_offset = False

        # it's not a bug, it's a feature
        self.cinematic = cinematic

    @property
    def view_mat(self) -> np.ndarray:
        """Generates the view matrix for the camera depending on the current orientation and implicit parameters.

        Returns:
            np.ndarray: view matrix.
        """
        # get the state of the camera on the robot
        camera_state = self.p.getLinkState(self.uav_id, self.camera_id)

        # pose and rot depends on offset if any
        position = np.array(camera_state[0])
        if self.has_camera_offset:
            offset_rotation = np.array(
                self.p.getMatrixFromQuaternion(camera_state[1])
            ).reshape(3, 3)
            offset_rotation = offset_rotation.T if self.cinematic else offset_rotation
            position += np.matmul(offset_rotation, self.camera_position_offset)

        # simulate gimballed camera if needed
        rotation = np.array(self.p.getEulerFromQuaternion(camera_state[1]))
        if self.use_gimbal:
            # camera tilted downward for gimballed mode
            rotation[0] = 0.0
            rotation[1] = -self.camera_angle_degrees / 180 * math.pi
        else:
            # camera tilted upward for FPV mode
            rotation[1] += self.camera_angle_degrees / 180 * math.pi
        rotation = np.array(self.p.getQuaternionFromEuler(rotation))
        rotation = np.array(self.p.getMatrixFromQuaternion(rotation)).reshape(3, 3)
        up_vector = np.matmul(rotation, np.array([0.0, 0.0, 1.0]))

        # where does the camera have the best focus
        if self.is_tracking_camera:
            target = camera_state[0]
        else:
            target = np.matmul(rotation, np.array([1000, 0, 0])) + position

        return self.p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=up_vector,
        )

    def get_states(self):
        """This does not need to be called for camera. Call `capture_image()` instead within `update_last()`."""
        warnings.warn(
            "This does not need to be called for camera. Call `capture_image()` instead within `update_last()`."
        )

    def state_update(self):
        """This does not need to be called for camera."""
        warnings.warn("`state_update` does not need to be called for camera.")

    def physics_update(self):
        """This does not need to be called for camera, call `capture_image` instead."""
        raise NameError(
            "`state_update` does not need to be called for camera, call `capture_image` instead."
        )

    def capture_image(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Captures the 3 relevant images from the camera.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: rgbaImg, depthImg, segImg
        """
        _, _, rgbaImg, depthImg, segImg = self.p.getCameraImage(
            height=self.camera_resolution[0],
            width=self.camera_resolution[1],
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
        )

        rgbaImg = np.asarray(rgbaImg).reshape(
            self.camera_resolution[0], self.camera_resolution[1], -1
        )
        depthImg = np.asarray(depthImg).reshape(
            self.camera_resolution[0], self.camera_resolution[1], -1
        )
        segImg = np.asarray(segImg).reshape(
            self.camera_resolution[0], self.camera_resolution[1], -1
        )

        return rgbaImg, depthImg, segImg
