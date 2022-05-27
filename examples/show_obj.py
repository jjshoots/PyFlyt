import math
import time
from signal import SIGINT, signal

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from pybullet_swarming.core.load_objs import loadOBJ, obj_visual


def shutdown_handler(*_):
    print("ctrl-c invoked")
    exit(0)


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)

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
