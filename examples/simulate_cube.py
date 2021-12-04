import os
import math
import numpy as np
import matplotlib.pyplot as plt
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.environment.simulator import Simulator

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == '__main__':
    check_venv()
    signal(SIGINT, shutdown_handler)

    # here we spawn a set of 3x3x3 drones
    dim_drones = 3

    # The cube is centered around offset
    offset = np.array([[0., 0., 0.8]])

    lin_range = np.array([-.4, .4])
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=dim_drones)
    height_range = np.array([-.4, .4])
    height_range = np.linspace(start=height_range[0], stop=height_range[1], num=dim_drones)

    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    cube = np.stack([grid_x, grid_y, grid_z], axis=-1)
    cube_orn = np.zeros_like(cube)

    swarm = Simulator( \
                    start_pos=cube + offset, \
                    start_orn=np.zeros_like(cube) \
                    )
    swarm.set_pos_control(True)
    swarm.go([1] * swarm.num_drones)

    # define cube rotation per timestep
    r = 1. / 1000.
    c, s = math.cos(1. * r), math.sin(1. * r)
    Rx = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    c, s = math.cos(1.4 * r), math.sin(2. * r)
    Ry = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    c, s = math.cos(1.7 * r), math.sin(3. * r)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    R = Rx @ Ry @ Rz

    for i in range(100000):
        # at each timestep, update the target positions
        cube = (R @ cube.T).T

        # append list of zeros to the end because setpoint has to be xyzr
        setpoints = np.concatenate((cube + offset, np.zeros((swarm.num_drones, 1))), axis=-1)

        # send the setpoints and step
        swarm.set_setpoints(setpoints)
        swarm.step()
