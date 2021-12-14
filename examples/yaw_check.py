import os
import math
import numpy as np
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import *
from pybullet_swarming.flier.drone_controller import Drone_Controller
from pybullet_swarming.flier.swarm_controller import Swarm_Controller


def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == "__main__":
    check_venv()
    signal(SIGINT, shutdown_handler)

    URIs = []
    URIs.append("radio://0/10/2M/E7E7E7E7E2")

    # arm all drones
    UAV = Swarm_Controller(URIs)
    UAV.set_pos_control(True)

    for i in range(1000):
        print(UAV.UAVs[0].position_estimate[3])
        UAV.sleep(0.1)

    # end the drone control
    UAV.end()
