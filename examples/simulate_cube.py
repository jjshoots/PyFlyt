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

    # here we spawn drones in a 3x3x3 grid
    drones_per_len = 3
    drones_per_height = 3
    offset = np.array([[0., 0., 2.]])
    linear_offset = np.array([[3., 0., 0.]])

    lin_range = np.array([-.5, .5])
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=drones_per_len)
    height_range = np.array([-.5, .5])
    height_range = np.linspace(start=height_range[0], stop=height_range[1], num=drones_per_height)

    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    start_pos = np.stack([grid_x, grid_y, grid_z], axis=-1)
    start_orn = np.zeros_like(start_pos)

    # spawn in the drones according to the positions plus offset and enable all of them
    swarm = Simulator( \
                      start_pos=start_pos + offset + linear_offset, \
                      start_orn=start_orn \
                      )
    swarm.set_pos_control(True)
    swarm.go([1] * swarm.num_drones)

    # cube positions is basically the spawn positions
    cube = start_pos
    ball = linear_offset

    # define cube rotation per timestep
    r = 1. / 1000.
    c, s = math.cos(1. * r), math.sin(1. * r)
    Rx = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    c, s = math.cos(2. * r), math.sin(2. * r)
    Ry = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    c, s = math.cos(3. * r), math.sin(3. * r)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    R1 = Rx @ Ry @ Rz

    # define ball rotation per timestep, only around z axis
    r = 1. / 1000.
    c, s = math.cos(r), math.sin(r)
    R2 = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    for i in range(10000):
        # at each timestep, update the target positions
        cube = (R1 @ cube.T).T
        ball = (R2 @ ball.T).T
        xyz = cube + ball + offset

        # append list of zeros to the end because setpoint has to be xyzr
        setpoint = np.concatenate((xyz, np.zeros((swarm.num_drones, 1))), axis=-1)

        # send the setpoints and step
        swarm.set_setpoints(setpoint)
        swarm.step()
