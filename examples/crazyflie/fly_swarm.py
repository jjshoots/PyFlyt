import os
from signal import SIGINT, signal

import numpy as np

from PyFlyt.crazyflie import SwarmController


def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)

    # here we attempt to control 2 drones in a swarm
    # each URI corresponds to one drone
    URIs = []
    URIs.append("radio://0/30/2M/E7E7E7E7E0")
    URIs.append("radio://0/30/2M/E7E7E7E7E1")
    URIs.append("radio://0/30/2M/E7E7E7E7E2")
    URIs.append("radio://0/30/2M/E7E7E7E7E3")
    URIs.append("radio://0/30/2M/E7E7E7E7E4")
    URIs.append("radio://0/30/2M/E7E7E7E7E5")
    URIs.append("radio://0/30/2M/E7E7E7E7E6")
    URIs.append("radio://0/30/2M/E7E7E7E7E7")

    # arm all drones
    swarm = SwarmController(URIs)
    # swarm.set_pos_control(True)
    swarm.go([1] * swarm.num_drones)

    # initial position target (relative to local start positions)
    # disable pos control for velocity control
    setpoints = [np.array([0.0, 0.0, 1.0, 0.0])] * swarm.num_drones
    setpoints = np.stack(setpoints, axis=0)
    swarm.set_setpoints(setpoints)
    swarm.sleep(5)

    # stop the swarm
    swarm.go([0] * swarm.num_drones)
    swarm.end()
