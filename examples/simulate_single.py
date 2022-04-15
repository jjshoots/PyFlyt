import os
import numpy as np
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import *
from pybullet_swarming.environment.simulator import Simulator


def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == "__main__":
    check_venv()
    signal(SIGINT, shutdown_handler)

    # spawn a drone at 0, 0, 1
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])
    env = Simulator(start_pos=start_pos, start_orn=start_orn)
    env.set_pos_control(True)
    env.go([1])

    # fly in a box like pattern, allowing 5 seconds between each setpoint
    env.set_setpoints(np.array([[0.0, 0.0, 1.0, 1.0]]))
    env.sleep(5)
    env.set_setpoints(np.array([[1.0, 0.0, 1.0, 1.0]]))
    env.sleep(5)
    env.set_setpoints(np.array([[1.0, 1.0, 1.0, 1.0]]))
    env.sleep(5)
    env.set_setpoints(np.array([[0.0, 1.0, 1.0, 1.0]]))
    env.sleep(5)
    env.set_setpoints(np.array([[0.0, 0.0, 1.0, 1.0]]))
    env.sleep(5)
    env.go([0])
