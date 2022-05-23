import math
import time
from signal import SIGINT, signal

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from pybullet_swarming.utilities.load_objs import loadOBJ, obj_visual


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

    mesh_scale = np.array([1.0, 1.0, 1.0]) * 0.3
    grass_visual = obj_visual(env, "models/plants/grass_tile.obj", meshScale=mesh_scale)

    for _ in range(2):
        base_pos = [np.random.randn(1), np.random.randn(1), 0]
        base_orn = [math.pi / 2.0, 0.0, np.random.randn(1)]

        loadOBJ(
            env, visualId=grass_visual, basePosition=base_pos, baseOrientation=base_orn
        )

    env.loadURDF(
        "models/vehicles/primitive_car/car.urdf",
        basePosition=[0.0, 0.0, 3.0],
        baseOrientation=env.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=False,
    )

    while True:
        time.sleep(1 / 240.0)
        env.stepSimulation()
