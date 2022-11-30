import argparse
import math
import os
from signal import SIGINT, signal

import numpy as np

from PyFlyt.crazyflie.simulator import Simulator
from PyFlyt.crazyflie.swarm_controller import Swarm_Controller


def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="Fly a single Crazyflie in a circle.")

    parser.add_argument(
        "--simulate",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Simulate the circle.",
    )

    parser.add_argument(
        "--hardware",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Run the circle on an actual drone.",
    )

    return parser.parse_args()


def fake_handler():
    start_pos = np.array([[0.0, 0.0, 0.05]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # spawn in a drone
    UAVs = Simulator(start_pos=start_pos, start_orn=start_orn)
    UAVs.set_pos_control(True)

    return UAVs


def real_handler():
    URIs = []
    URIs.append("radio://0/10/2M/E7E7E7E7E2")

    # connect to a drone
    UAVs = Swarm_Controller(URIs)
    UAVs.set_pos_control(True)

    return UAVs


if __name__ == "__main__":
    args = get_args()
    signal(SIGINT, shutdown_handler)

    UAVs = None
    if args.simulate:
        UAVs = fake_handler()
    elif args.hardware:
        UAVs = real_handler()
    else:
        print("Guess this is life now.")
        exit()

    # arm all drones
    UAVs.arm([1])

    # initial hover
    UAVs.set_setpoints(np.array([[0.0, 1.0, 1.0, 0.0]]))
    UAVs.sleep(10)

    # send to top of circle
    UAVs.set_setpoints(np.array([[1.0, 1.0, 1.0, 0.0]]))
    UAVs.sleep(10)

    for i in range(300):
        theta = float(i) / 10.0
        c, s = math.cos(theta), math.sin(theta)

        UAVs.set_setpoints(np.array([[c, s + 1, 1.0, 0.0]]))
        UAVs.sleep(0.1)

    # send the drone back to origin hover
    UAVs.set_setpoints(np.array([[0.0, 1.0, 1.0, 0]]))
    UAVs.sleep(10)

    # send the drone back down
    UAVs.set_setpoints(np.array([[0.0, 1.0, -1.0, 0]]))
    UAVs.sleep(5)

    # stop the drone flight controller
    UAVs.arm([0])
    UAVs.sleep(5)

    # end the drone control
    UAVs.end()
