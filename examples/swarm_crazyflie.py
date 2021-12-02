import os
import numpy as np
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.flier.swarm_controller import *

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == '__main__':
    check_venv()
    signal(SIGINT, shutdown_handler)

    # here we attempt to control 2 drones in a swarm
    # each URI corresponds to one drone
    URIs = []
    URIs.append('radio://0/10/2M/E7E7E7E7E1')
    URIs.append('radio://0/10/2M/E7E7E7E7E2')
    URIs.append('radio://0/10/2M/E7E7E7E7E3')
    URIs.append('radio://0/10/2M/E7E7E7E7E4')
    URIs.append('radio://0/10/2M/E7E7E7E7E5')
    URIs.append('radio://0/10/2M/E7E7E7E7E6')

    # arm all drones
    swarm = Swarm_Controller(URIs)
    # swarm.set_pos_control(True)
    swarm.go([1] * 6)
    swarm.sleep(2)

    # initial position target (relative to local start positions)
    # disable pos control for velocity control
    setpoints = [np.array([0., 0., 1., 0.])] * swarm.num_drones
    setpoints = np.stack(setpoints, axis=0)
    swarm.set_setpoints(setpoints)
    swarm.sleep(5)

    # stop the swarm
    swarm.go([0] * swarm.num_drones)
    swarm.end()
