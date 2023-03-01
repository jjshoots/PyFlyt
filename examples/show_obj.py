"""Spawns an object into the environment from a urdf file."""
import time

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

if __name__ == "__main__":
    env = bullet_client.BulletClient(p.GUI)
    env.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    env.setGravity(0, 0, -9.81)

    """ CONSTRUCT THE WORLD """
    env.loadURDF("plane.urdf", useFixedBase=True)

    env.loadURDF(
        "examples/car.urdf",
        basePosition=[0.0, 0.0, 3.0],
        baseOrientation=env.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=False,
    )

    while True:
        time.sleep(1 / 240.0)
        env.stepSimulation()
