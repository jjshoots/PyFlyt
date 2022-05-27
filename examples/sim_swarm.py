import os
from signal import SIGINT, signal

import numpy as np

from pybullet_swarming.hardware.simulator import Simulator


def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)

    # here we spawn drones in a 3x3x2 grid
    drones_per_len = 3
    drones_per_height = 2

    lin_range = [-1, 1]
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=drones_per_len)
    height_range = [1.0, 2.0]
    height_range = np.linspace(
        start=height_range[0], stop=height_range[1], num=drones_per_height
    )

    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    start_pos = np.stack([grid_x, grid_y, grid_z], axis=-1)
    start_orn = np.zeros_like(start_pos)

    # spawn in the drones according to the positions and enable all of them
    swarm = Simulator(start_pos=start_pos, start_orn=start_orn)
    swarm.set_pos_control(False)
    swarm.go([1] * swarm.num_drones)

    # make the drone fly in horizontal x with 1 rad/s yawrate
    setpoints = [1.0, 0.0, 0.0, 1.0]
    setpoints = np.array([setpoints] * swarm.num_drones)

    # send the setpoint for 5 seconds
    swarm.set_setpoints(setpoints)
    swarm.sleep(5)

    # make the drone fly in horizontal x with 1 rad/s yawrate
    setpoints = [-1.0, 0.0, 0.0, 1.0]
    setpoints = np.array([setpoints] * swarm.num_drones)

    # send the setpoint for 5 seconds
    swarm.set_setpoints(setpoints)
    swarm.sleep(5)

    # disarm all drones
    swarm.go([0] * swarm.num_drones)
    swarm.sleep(4)
