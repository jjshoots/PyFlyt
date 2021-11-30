import os
import time
import numpy as np
from signal import signal, SIGINT

from flier.swarm_controller import swarm_controller

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == '__main__':

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

    print('go')
    swarm = swarm_controller(URIs)
    swarm.set_pos_control(True)

    for i in range(swarm.num_drones):
        mask = [0 for _ in range(swarm.num_drones)]
        mask[i] = 1
        swarm.go(mask)
        time.sleep(2)

    swarm.go([0] * 6)
    time.sleep(2)
    swarm.go([1] * 6)
    time.sleep(2)

    # initial position target
    print('setpoint')
    setpoints = [np.array([0., 0., 1., 0.])] * swarm.num_drones
    setpoints = np.stack(setpoints, axis=0)
    swarm.set_setpoints(setpoints)
    time.sleep(5)

    # stop the swarm
    print('stopp')
    swarm.go([0] * swarm.num_drones)
    time.sleep(10)

    # end the swarm
    print('end')
    swarm.end()
