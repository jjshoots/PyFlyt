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
    drones_per_len = 4
    drones_per_height = 4
    height_offset = np.array([[0., 0., 2.]])

    lin_range = np.array([-.5, .5])
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=drones_per_len)
    height_range = np.array([-.5, .5])
    height_range = np.linspace(start=height_range[0], stop=height_range[1], num=drones_per_height)

    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    start_pos = np.stack([grid_x, grid_y, grid_z], axis=-1)
    start_orn = np.zeros_like(start_pos)

    # spawn in the drones according to the positions and enable all of them
    swarm = Simulator( \
                      start_pos=start_pos + height_offset, \
                      start_orn=start_orn \
                      )
    swarm.set_pos_control(True)
    swarm.go([1] * swarm.num_drones)

    # cube positions is basically the spawn positions minus height offset
    cube = start_pos
    ball = np.array([[0., 0., 0.]])

    # r is delta radian per 1/240 seconds
    r = 1. / 1000.
    c, s = math.cos(r), math.sin(r)
    Rx = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    r = 2. / 1000.
    c, s = math.cos(r), math.sin(r)
    Ry = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    r = 3. / 1000.
    c, s = math.cos(r), math.sin(r)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    r = 1. / 3000.
    c, s = math.cos(r), math.sin(r)
    Rb = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    for i in range(10000):
        cube = (Rx @ Ry @ Rz @ cube.T).T
        ball = (Rb @ ball.T).T
        xyz = cube + ball + height_offset

        setpoint = np.concatenate((xyz, np.zeros((swarm.num_drones, 1))), axis=-1)

        swarm.set_setpoints(setpoint)
        swarm.step()
