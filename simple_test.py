import os
import time
import numpy as np
from signal import signal, SIGINT

from drone_controller import *

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == '__main__':

    signal(SIGINT, shutdown_handler)

    # here we attempt to control 1 drone
    # the URI corresponds to the 1 drone
    UAV = drone_controller('radio://0/10/2M/E7E7E7E7E7')

    # start the flight controller and put into pos control
    UAV.set_pos_control(True)
    UAV.start()
    time.sleep(1)

    print('hover')
    # send the drone to hover 0.5 meters
    UAV.set_setpoint(np.array([0., 0., .5, 0.]))
    time.sleep(5)

    print('land')
    # send the drone back doen
    UAV.set_setpoint(np.array([0., 0., 0., 0.]))
    time.sleep(5)

    print('stop')
    # stop the drone flight controller
    UAV.stop()
    time.sleep(5)

    # end the drone control
    UAV.end()


