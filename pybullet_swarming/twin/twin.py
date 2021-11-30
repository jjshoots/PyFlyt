import math
import numpy as np
import matplotlib.pyplot as plt

from pybullet_swarming.env.environment import *
from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.flier.swarm_controller import *

class Twin():
    def __init__(self, URIs):
         # spawn drones
        drones_per_len = 4
        drones_per_height = 1

        lin_range = [-.2, .2]
        lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=drones_per_len)
        height_range = [.1, .1]
        height_range = np.linspace(start=height_range[0], stop=height_range[1], num=drones_per_height)

        grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
        grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

        start_pos = np.stack([grid_x, grid_y, grid_z], axis=-1)
        start_orn = np.zeros_like(start_pos)

        # instantiate the swarm
        swarm = Swarm_Controller(URIs=URIs)

        # get the swarm positions
        start_pos = swarm.states[:, :-1]
        start_orn = np.zeros_like(start_pos)

        # instantiate the digital twin
        env = Environment(start_pos=start_pos, start_orn=start_orn)


