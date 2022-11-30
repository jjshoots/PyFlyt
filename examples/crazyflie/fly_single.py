import os
from signal import SIGINT, signal

import numpy as np

from PyFlyt.crazyflie import DroneController


def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)

    # here we attempt to control 1 drone
    # the URI corresponds to the 1 drone
    UAV = DroneController("radio://0/10/2M/E7E7E7E7E3")

    # start the flight controller and put into pos control
    UAV.set_pos_control(True)
    UAV.start()

    # send the drone to hover 0.5 meters
    UAV.set_setpoint(np.array([0.0, 0.0, 0.5, 0.0]))
    UAV.sleep(5)

    # send the drone back doen
    UAV.set_setpoint(np.array([0.0, 0.0, 0.0, 0.0]))
    UAV.sleep(5)

    # stop the drone flight controller
    UAV.stop()
    UAV.sleep(5)

    # end the drone control
    UAV.end()
