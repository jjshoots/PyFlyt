"""Record a drone's onboard camera together with the exact camera pose.

`CameraRecorder` writes a posed image sequence from the `Camera` component of a drone:

    <output_dir>/
        images/000000.png, 000001.png, ...
        depth/000000.npy, ...       (optional, metric z-depth in metres, float32)
        poses.csv                   timestamp,tx,ty,tz,qx,qy,qz,qw
        intrinsics.txt              fx fy cx cy

Each row of `poses.csv` is the camera-to-world transform of the matching image,
with camera axes in the OpenCV convention (x right, y down, z forward) and the
PyBullet world frame (z up). Timestamps are simulation time in seconds.

The pose is obtained by inverting the same view matrix that is used to render
the image, so it follows every `Camera` option (gimbal, tilt angle, position
offset, tracking camera) and pose and pixels are consistent by construction.
Images are always rendered with `ER_TINY_RENDERER`, so the intrinsics are the
same with and without `render=True`.

This layout can be read by most visual odometry, SLAM, NeRF / Gaussian-splatting
and world-model training pipelines.
"""

from __future__ import annotations

import csv
import os
import struct
import zlib

import numpy as np
import pybullet as p

# OpenGL eye axes (x right, y up, z backward) from OpenCV camera axes
_GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def view_matrix_to_camera_pose(view_matrix) -> np.ndarray:
    """Converts a PyBullet view matrix into an OpenCV camera-to-world pose.

    Args:
        view_matrix: 16 floats, column-major, e.g. from `computeViewMatrix`.

    Returns:
        np.ndarray: (4, 4) camera-to-world transform with OpenCV camera axes.

    """
    view = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
    return np.linalg.inv(view) @ _GL_FROM_CV


def camera_pose_to_view_matrix(T_wc: np.ndarray) -> list[float]:
    """Converts an OpenCV camera-to-world pose into a PyBullet view matrix.

    Args:
        T_wc (np.ndarray): (4, 4) camera-to-world transform with OpenCV camera axes.

    Returns:
        list[float]: 16 floats, column-major.

    """
    view = np.linalg.inv(np.asarray(T_wc, dtype=np.float64) @ _GL_FROM_CV)
    return view.flatten(order="F").tolist()


def projection_matrix_to_intrinsics(
    projection_matrix, width: int, height: int, tiny_renderer: bool = True
) -> tuple[np.ndarray, float, float]:
    """Extracts pinhole intrinsics and clipping planes from a PyBullet projection matrix.

    PyBullet's TinyRenderer samples pixels half a pixel away from the ideal OpenGL
    convention: its principal point is at (W/2, H/2 - 1) instead of ((W-1)/2, (H-1)/2).

    Args:
        projection_matrix: 16 floats, column-major, e.g. from `computeProjectionMatrixFOV`.
        width (int): image width in pixels.
        height (int): image height in pixels.
        tiny_renderer (bool): whether images are rendered with `ER_TINY_RENDERER`.

    Returns:
        tuple[np.ndarray, float, float]: (3, 3) intrinsics matrix K (pixel centres at integer coordinates), near plane, far plane.

    """
    P = np.asarray(projection_matrix, dtype=np.float64).reshape(4, 4, order="F")
    fx = P[0, 0] * width / 2.0
    fy = P[1, 1] * height / 2.0
    cx = (1.0 - P[0, 2]) * width / 2.0 - 0.5
    cy = (1.0 + P[1, 2]) * height / 2.0 - 0.5
    if tiny_renderer:
        cx, cy = cx + 0.5, cy - 0.5
    near = P[2, 3] / (P[2, 2] - 1.0)
    far = P[2, 3] / (P[2, 2] + 1.0)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, float(near), float(far)


def depth_buffer_to_metric(
    depth_buffer: np.ndarray, near: float, far: float
) -> np.ndarray:
    """Converts a PyBullet depth buffer (values in [0, 1]) into metric z-depth.

    Args:
        depth_buffer (np.ndarray): depth buffer returned by `getCameraImage`.
        near (float): near clipping plane.
        far (float): far clipping plane.

    Returns:
        np.ndarray: distance along the camera's optical axis in metres.

    """
    depth_buffer = np.asarray(depth_buffer, dtype=np.float64)
    return far * near / (far - (far - near) * depth_buffer)


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix into a unit quaternion [x, y, z, w] with w >= 0.

    Args:
        R (np.ndarray): (3, 3) rotation matrix.

    Returns:
        np.ndarray: (4,) quaternion.

    """
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def write_png(path: str, image: np.ndarray) -> None:
    """Writes an 8-bit RGB or grayscale image as PNG without extra dependencies.

    Args:
        path (str): output file path.
        image (np.ndarray): (H, W, 3) or (H, W) uint8 array.

    """
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    color_type = 2 if image.ndim == 3 else 0
    rows = image.reshape(height, -1)
    raw = np.hstack([np.zeros((height, 1), dtype=np.uint8), rows]).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(
            chunk(
                b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
            )
        )
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


class CameraRecorder:
    """Saves posed images from the onboard `Camera` of a drone in an `Aviary`.

    The drone must be created with `use_camera=True`. Call `capture()` whenever a
    frame should be saved, typically every few `env.step()` calls.

    Example:
        ```python
        env = Aviary(..., drone_options=dict(use_camera=True))
        with CameraRecorder(env, "flight_0") as rec:
            for i in range(1000):
                env.step()
                if i % 12 == 0:
                    rec.capture()
        ```

    Args:
        aviary: the `Aviary` instance.
        output_dir (str): directory where the sequence is written (created if missing).
        drone_index (int): index of the drone in `aviary.drones`.
        save_depth (bool): whether to also save metric depth maps as `.npy` files.

    """

    def __init__(
        self,
        aviary,
        output_dir: str,
        drone_index: int = 0,
        save_depth: bool = False,
    ):
        """Initializes the recorder and writes `intrinsics.txt`.

        Args:
            aviary: the `Aviary` instance.
            output_dir (str): directory where the sequence is written (created if missing).
            drone_index (int): index of the drone in `aviary.drones`.
            save_depth (bool): whether to also save metric depth maps as `.npy` files.

        """
        self.aviary = aviary
        self.drone_index = drone_index
        drone = aviary.drones[drone_index]
        if not getattr(drone, "use_camera", False):
            raise ValueError(
                "CameraRecorder needs a drone created with `use_camera=True` in `drone_options`."
            )
        self.camera = drone.camera
        self.height, self.width = (int(v) for v in self.camera.camera_resolution)
        self.K, self.near, self.far = projection_matrix_to_intrinsics(
            self.camera.proj_mat, self.width, self.height
        )
        self.save_depth = save_depth

        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        if save_depth:
            os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
        np.savetxt(
            os.path.join(output_dir, "intrinsics.txt"),
            self.K[[0, 1, 0, 1], [0, 1, 2, 2]][None],
            fmt="%.6f",
            header="fx fy cx cy",
        )
        self._csv_file = open(os.path.join(output_dir, "poses.csv"), "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        self.frame_count = 0

    def camera_pose(self) -> np.ndarray:
        """Returns the current camera-to-world transform with OpenCV camera axes.

        Returns:
            np.ndarray: (4, 4) transform.

        """
        return view_matrix_to_camera_pose(self.camera.view_mat)

    def capture(self, timestamp: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Renders one frame, writes it to disk and appends its pose.

        Args:
            timestamp (float | None): time stamp in seconds, defaults to `aviary.elapsed_time`.

        Returns:
            tuple[np.ndarray, np.ndarray]: (H, W, 3) uint8 RGB image, (4, 4) camera-to-world transform.

        """
        view = self.camera.view_mat
        T_wc = view_matrix_to_camera_pose(view)
        if timestamp is None:
            timestamp = float(self.aviary.elapsed_time)

        _, _, rgba, depth_buffer, _ = self.aviary.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view,
            projectionMatrix=self.camera.proj_mat,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[
            ..., :3
        ]

        name = f"{self.frame_count:06d}"
        write_png(os.path.join(self.output_dir, "images", name + ".png"), rgb)
        if self.save_depth:
            depth = depth_buffer_to_metric(
                np.reshape(depth_buffer, (self.height, self.width)),
                self.near,
                self.far,
            )
            np.save(
                os.path.join(self.output_dir, "depth", name + ".npy"),
                depth.astype(np.float32),
            )

        quat = rotation_matrix_to_quat_xyzw(T_wc[:3, :3])
        self._csv.writerow(
            [f"{timestamp:.6f}"]
            + [f"{v:.9f}" for v in T_wc[:3, 3]]
            + [f"{v:.9f}" for v in quat]
        )
        self._csv_file.flush()
        self.frame_count += 1
        return rgb, T_wc

    def close(self) -> None:
        """Closes the pose file."""
        if not self._csv_file.closed:
            self._csv_file.close()

    def __enter__(self) -> CameraRecorder:
        """Enters a context that closes the recorder on exit."""
        return self

    def __exit__(self, *exc) -> None:
        """Closes the recorder."""
        self.close()
