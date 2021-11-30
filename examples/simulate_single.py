import os
import time
import numpy as np
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.env.twin import Twin

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == '__main__':
    check_venv()
    signal(SIGINT, shutdown_handler)

    # spawn a drone at 0, 0, 1
    start_pos = np.array([[0., 0., 1.]])
    start_orn = np.array([[0., 0., 0.]])
    env = Twin(start_pos=start_pos, start_orn=start_orn)

    # fly horizontally with x velocity 1 and rotate at 1 rad/s yaw
    env.set_setpoints(np.array([[1., 0., 0., 1.]]))
    env.sleep(5)
