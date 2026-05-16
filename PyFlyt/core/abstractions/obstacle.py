"""Obstacle class for spawning static URDF obstacles in the Aviary."""

from __future__ import annotations

import os

import numpy as np
from pybullet_utils import bullet_client

# directory containing the built-in obstacle URDFs
_BUILTIN_OBSTACLE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "models",
    "obstacles",
)

# mapping of friendly names to their built-in URDF files
_BUILTIN_OBSTACLES: dict[str, str] = {
    "cube": os.path.join(_BUILTIN_OBSTACLE_DIR, "cube.urdf"),
    "cylinder": os.path.join(_BUILTIN_OBSTACLE_DIR, "cylinder.urdf"),
    "sphere": os.path.join(_BUILTIN_OBSTACLE_DIR, "sphere.urdf"),
}


class Obstacle:
    """An `Obstacle` represents a static object in the Aviary that drones may collide with.

    Obstacles are loaded from URDF files. PyFlyt ships with a small number of built-in
    primitive obstacles (`"cube"`, `"cylinder"`, `"sphere"`), accessible by name.
    Arbitrary URDF files may also be loaded by providing a filesystem path.

    Obstacles are always spawned with `useFixedBase=True`, which means PyBullet will not
    update their positions during simulation regardless of the mass defined in the URDF.
    This is the intended behaviour for static scene geometry.

    Example:
        >>> from PyFlyt.core import Aviary
        >>> import numpy as np
        >>>
        >>> env = Aviary(
        ...     start_pos=np.array([[0.0, 0.0, 1.0]]),
        ...     start_orn=np.array([[0.0, 0.0, 0.0]]),
        ...     drone_type="quadx",
        ... )
        >>>
        >>> # spawn a built-in cube obstacle
        >>> env.add_obstacle("cube", position=np.array([2.0, 0.0, 0.5]))
        >>>
        >>> # spawn an obstacle from a custom URDF
        >>> env.add_obstacle("/path/to/my_obstacle.urdf", position=np.array([4.0, 0.0, 1.0]))

    Args:
        p (bullet_client.BulletClient): PyBullet physics client.
        urdf (str): either the name of a built-in obstacle (`"cube"`, `"cylinder"`, `"sphere"`),
            or an absolute path to a URDF file.
        position (np.ndarray): an `(3,)` array for the X, Y, Z spawn position.
        orientation (np.ndarray): an `(3,)` array for the spawn orientation as Euler angles
            (roll, pitch, yaw) in radians. Defaults to zero rotation.
        scale (float): a uniform scaling factor applied to the URDF on load. Defaults to 1.0.

    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        urdf: str,
        position: np.ndarray,
        orientation: np.ndarray | None = None,
        scale: float = 1.0,
    ):
        """Loads the obstacle URDF into the PyBullet client and stores its body ID.

        Args:
            p (bullet_client.BulletClient): PyBullet physics client.
            urdf (str): name of a built-in obstacle or path to a URDF file.
            position (np.ndarray): `(3,)` array for the X, Y, Z spawn position.
            orientation (np.ndarray): `(3,)` array of Euler angles in radians.
            scale (float): a uniform scaling factor applied to the URDF on load.

        """
        # resolve a built-in name to its packaged URDF path
        urdf_path = _BUILTIN_OBSTACLES.get(urdf, urdf)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"Could not find obstacle URDF `{urdf}`. "
                f"Expected either a built-in name from {list(_BUILTIN_OBSTACLES)} "
                f"or a path to an existing URDF file."
            )

        # validate the position shape
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(
                f"`position` must be shape (3,), got {position.shape}."
            )

        # default orientation is no rotation
        if orientation is None:
            orientation = np.zeros(3, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        if orientation.shape != (3,):
            raise ValueError(
                f"`orientation` must be shape (3,), got {orientation.shape}."
            )

        # store handles
        self.p = p
        self.urdf_path = urdf_path
        self.start_pos = position
        self.start_orn = orientation
        self.scale = float(scale)

        # convert euler to quaternion for pybullet
        quat = self.p.getQuaternionFromEuler(orientation.tolist())

        # spawn the obstacle as a fixed-base body
        self.Id: int = self.p.loadURDF(
            urdf_path,
            basePosition=position.tolist(),
            baseOrientation=quat,
            useFixedBase=True,
            globalScaling=self.scale,
        )

    @classmethod
    def builtin_obstacles(cls) -> list[str]:
        """Returns the list of built-in obstacle names recognised by `Obstacle`.

        Returns:
            list[str]: names usable as the `urdf` argument to `add_obstacle`.

        """
        return list(_BUILTIN_OBSTACLES)
