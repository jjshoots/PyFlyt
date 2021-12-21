import os
import math
import numpy as np
import matplotlib.pyplot as plt
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import *
from pybullet_swarming.flier.swarm_controller import Swarm_Controller
from pybullet_swarming.environment.simulator import Simulator

global DIM_DRONES
DIM_DRONES = 2

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
    global DIM_DRONES
    # here we spawn drones in a circle
    theta = np.arange(0, 2 * math.pi, 2 * math.pi / (DIM_DRONES ** 3))
    distance = 2.0
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    z = np.ones_like(x) * 0.05
    start_pos = np.stack((x, y, z), axis=-1)
    start_orn = np.zeros_like(start_pos)

    # spawn in a drone
    UAVs = Simulator(start_pos=start_pos, start_orn=start_orn)
    UAVs.set_pos_control(True)

    return UAVs


def real_handler():
    URIs = []
    URIs.append("radio://0/10/2M/E7E7E7E7E7")
    URIs.append("radio://1/10/2M/E7E7E7E7E1")
    URIs.append("radio://1/10/2M/E7E7E7E7E6")
    URIs.append("radio://1/10/2M/E7E7E7E7E5")
    URIs.append("radio://0/30/2M/E7E7E7E7E0")
    URIs.append("radio://0/10/2M/E7E7E7E7E3")
    URIs.append("radio://0/10/2M/E7E7E7E7E2")
    URIs.append("radio://1/30/2M/E7E7E7E7E4")

    # connect to a drone
    UAVs = Swarm_Controller(URIs)
    UAVs.set_pos_control(True)

    return UAVs


def get_circle(radius, height):
    global DIM_DRONES

    theta = np.arange(0, 2 * math.pi, 2 * math.pi / (DIM_DRONES ** 3))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.ones_like(x) * height

    return np.stack((x, y, z), axis=-1)


def get_cube(radius):
    global DIM_DRONES

    lin_range = np.array([-radius, radius])
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=DIM_DRONES)
    height_range = np.array([-radius, radius])
    height_range = np.linspace(
        start=height_range[0], stop=height_range[1], num=DIM_DRONES
    )

    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    return np.stack([grid_x, grid_y, grid_z], axis=-1)



if __name__ == "__main__":
    check_venv()
    args = get_args()
    signal(SIGINT, shutdown_handler)

    # get the swarm handler
    UAVs = None
    if args.simulate:
        UAVs = fake_handler()
    elif args.hardware:
        UAVs = real_handler()
    else:
        print("Guess this is life now.")
        exit()

    # offsets for cube
    cube_offset = np.array([[0.0, 0.0, 1.]])
    # linear_offset = np.array([[0.3, 0.0, 0.15]])
    linear_offset = np.array([[0., 0., 0.]])

    # form the cube coordinates
    cube = get_cube(0.5)
    cube_orn = np.zeros_like(cube)

    # define cube rotation per timestep
    r = 1.0 / 1000.0
    c, s = math.cos(1.0 * r), math.sin(1.0 * r)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    c, s = math.cos(1.4 * r), math.sin(2.0 * r)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    c, s = math.cos(1.7 * r), math.sin(3.0 * r)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    R1 = Rx @ Ry @ Rz

    # define offset rotation per timestep
    r = 1.0 / 2000.0
    c, s = math.cos(1.0 * r), math.sin(1.0 * r)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    c, s = math.cos(1.4 * r), math.sin(2.0 * r)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    c, s = math.cos(1.7 * r), math.sin(3.0 * r)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    R2 = Rx @ Ry @ Rz

    # reshuffle drones according to cube pos, then arm from highest cost first
    cost = UAVs.reshuffle(
        cube + cube_offset + linear_offset, np.zeros((UAVs.num_drones, 3))
    )

    settings = np.zeros(UAVs.num_drones)
    for _ in range(UAVs.num_drones):
        i = np.argmax(cost)
        settings[i] = 1
        cost[i] = -100.0
        UAVs.go(settings)
        UAVs.sleep(1)
    UAVs.sleep(20)

    for i in range(1000):
        # at each timestep, update the target positions
        cube = (R1 @ cube.T).T
        linear_offset = (R2 @ linear_offset.T).T
        xyz = cube + cube_offset + linear_offset

        # append list of zeros to the end because setpoint has to be xyzr
        setpoints = np.concatenate((xyz, np.zeros((UAVs.num_drones, 1))), axis=-1)

        # send the setpoints and step
        UAVs.set_setpoints(setpoints)
        UAVs.step()

    # circle targets
    circle = get_circle(1., 1.)
    UAVs.reshuffle(circle, np.zeros_like(circle))
    UAVs.sleep(5)

    # circle targets
    circle = get_circle(1., -1.)
    UAVs.reshuffle(circle, np.zeros_like(circle))
    UAVs.sleep(5)

    UAVs.go([0] * UAVs.num_drones)
    UAVs.sleep(2)
    UAVs.end()
