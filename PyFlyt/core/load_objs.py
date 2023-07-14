"""Convenience function to load an obj into the pybullet environment."""
from __future__ import annotations

import numpy as np
from pybullet_utils import bullet_client


def loadOBJ(
    env: bullet_client.BulletClient,
    fileName: str = "null",
    visualId: int = -1,
    collisionId: int = -1,
    baseMass: float = 0.0,
    meshScale: list[float] | np.ndarray = [1.0, 1.0, 1.0],
    basePosition: list[float] | np.ndarray = [0.0, 0.0, 0.0],
    baseOrientation: list[float] | np.ndarray = [0.0, 0.0, 0.0],
):
    """Loads an object into the environment.

    Args:
        env (Aviary): env
        fileName (str): fileName
        visualId (int): visualId
        collisionId (int): collisionId
        baseMass (float): baseMass
        meshScale (list[float] | np.ndarray): meshScale
        basePosition (list[float] | np.ndarray): basePosition
        baseOrientation (list[float] | np.ndarray): baseOrientation
    """
    if len(baseOrientation) == 3:
        baseOrientation = env.getQuaternionFromEuler(baseOrientation)

    if visualId == -1:
        visualId = obj_visual(env, fileName, meshScale)

    body_id = env.createMultiBody(
        baseMass=baseMass,
        baseVisualShapeIndex=int(visualId),
        baseCollisionShapeIndex=int(collisionId),
        basePosition=basePosition,
        baseOrientation=baseOrientation,
    )

    env.register_all_new_bodies()

    return body_id


def obj_visual(
    env: bullet_client.BulletClient,
    fileName: str,
    meshScale: list[float] | np.ndarray = [1.0, 1.0, 1.0],
):
    """Loads an object visual model.

    Args:
        env (Aviary): env
        fileName (str): fileName
        meshScale (list[float] | np.ndarray): meshScale
    """
    return env.createVisualShape(
        shapeType=env.GEOM_MESH,
        fileName=fileName,
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.0, 0.0, 0.0],
        meshScale=meshScale,
    )


def obj_collision(
    env: bullet_client.BulletClient,
    fileName: str,
    meshScale: list[float] | np.ndarray = [1.0, 1.0, 1.0],
    concave: bool = False,
):
    """Loads an object collision model.

    Args:
        env (Aviary): env
        fileName (str): fileName
        meshScale (list[float] | np.ndarray): meshScale
        concave (bool): Whether the object should use concave trimesh, do not use this for dynamic/moving objects
    """
    return env.createCollisionShape(
        shapeType=env.GEOM_MESH,
        fileName=fileName,
        meshScale=meshScale,
        flags=env.GEOM_FORCE_CONCAVE_TRIMESH if concave else None,
    )
