import os
import math
import numpy as np
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.flier.drone_controller import Drone_Controller
from pybullet_swarming.flier.swarm_controller import Swarm_Controller

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == '__main__':
    check_venv()
    signal(SIGINT, shutdown_handler)

    URIs = []
    URIs.append('radio://0/10/2M/E7E7E7E7E2')

    # arm all drones
    UAV = Swarm_Controller(URIs)
    UAV.set_pos_control(True)
    UAV.go([1])

    # initial hover
    UAV.set_setpoints(np.array([[0., 0., 1., 0.]]))
    UAV.sleep(10)

    # send to top pof circle
    UAV.set_setpoints(np.array([[1., 0., 1., 0.]]))
    UAV.sleep(10)

    for i in range(300):
        theta = float(i) / 10.
        c, s = math.cos(theta), math.sin(theta)

        UAV.set_setpoints(np.array([[c, s, 1., 0.]]))
        UAV.sleep(0.1)

    # send the drone back to origin hover
    UAV.set_setpoints(np.array([[0., 0., 1., 0]]))
    UAV.sleep(10)

    # send the drone back down
    UAV.set_setpoints(np.array([[0., 0., -1., 0]]))
    UAV.sleep(5)

    # stop the drone flight controller
    UAV.go([0])
    UAV.sleep(5)

    # end the drone control
    UAV.end()


