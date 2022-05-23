import math
import os
from signal import SIGINT, signal

import matplotlib.pyplot as plt
import numpy as np

from pybullet_swarming.environment.simulator import Simulator


def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)

    # the cube is made up of 3x3x3 drones
    dim_drones = 3

    # use meshgrid to form the coordinates for drones to form the cube
    lin_range = np.array([-0.4, 0.4])
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=dim_drones)
    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, lin_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    # The cube is centered around offset
    cube = np.stack([grid_x, grid_y, grid_z], axis=-1)
    offset = np.array([[0.0, 0.0, 0.8]])

    # define cube rotation per timestep
    r = 1.0 / 1000.0
    c, s = math.cos(1.0 * r), math.sin(1.0 * r)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    c, s = math.cos(1.4 * r), math.sin(2.0 * r)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    c, s = math.cos(1.7 * r), math.sin(3.0 * r)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    R = Rx @ Ry @ Rz

    # spawn the drones and go!
    swarm = Simulator(start_pos=cube + offset, start_orn=np.zeros_like(cube))
    swarm.set_pos_control(True)
    swarm.go([1] * swarm.num_drones)

    for i in range(100000):
        # at each timestep, update the target positions
        cube = (R @ cube.T).T

        # append list of zeros to the end because setpoint has to be xyzr
        setpoints = np.concatenate(
            (cube + offset, np.zeros((swarm.num_drones, 1))), axis=-1
        )

        # send the setpoints and step
        swarm.set_setpoints(setpoints)
        swarm.step()
