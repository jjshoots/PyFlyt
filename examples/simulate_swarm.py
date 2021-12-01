import os
import numpy as np
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.env.simulator import Simulator

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == '__main__':
    check_venv()
    signal(SIGINT, shutdown_handler)

    # here we spawn drones in a 2x2x1 grid
    drones_per_len = 2
    drones_per_height = 1

    lin_range = [-.2, .2]
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=drones_per_len)
    height_range = [1., 1.]
    height_range = np.linspace(start=height_range[0], stop=height_range[1], num=drones_per_height)

    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    start_pos = np.stack([grid_x, grid_y, grid_z], axis=-1)
    start_orn = np.zeros_like(start_pos)

    swarm = Twin(start_pos=start_pos, start_orn=start_orn)

    # make the drone fly in horizontal x with 1 rad/s yawrate
    setpoints = [1., 0., 0., 1.]
    setpoints = np.array([setpoints] * swarm.num_drones)

    # send the setpoint for 5 seconds
    swarm.set_setpoints(setpoints)
    swarm.sleep(4)

    # make the drone fly in horizontal x with 1 rad/s yawrate
    setpoints = [-1., 0., 0., 1.]
    setpoints = np.array([setpoints] * swarm.num_drones)

    # send the setpoint for 5 seconds
    swarm.set_setpoints(setpoints)
    swarm.sleep(4)
