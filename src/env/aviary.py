import time
import numpy as np
import multiprocessing as mp

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from env.drone import *

class Aviary(bullet_client.BulletClient):
    def __init__(self, drone_dir, render=True):
        super().__init__(p.GUI if render else p.DIRECT)

        self.drone_dir = drone_dir

        # default physics looprate is 240 Hz
        self.period = 1. / 240.
        self.now = time.time()

        self.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.render = render
        self.reset()


    def reset(self):
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.step_count = 0

        """ CONSTRUCT THE WORLD """
        self.planeId = self.loadURDF(
            "plane.urdf",
            useFixedBase=True,
            globalScaling=np.random.rand() * 20. + 1.
        )

        # spawn drones
        drones_per_len = 4
        drones_per_height = 1

        lin_range = [-.2, .2]
        lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=drones_per_len)
        height_range = [.1, .1]
        height_range = np.linspace(start=height_range[0], stop=height_range[1], num=drones_per_height)

        grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)

        self.drones = []
        for x_pos, y_pos, z_pos in zip(grid_x.flatten(), grid_y.flatten(), grid_z.flatten()):
            start_pos = np.array([x_pos, y_pos, z_pos])
            start_orn = np.array([0, 0, 0])
            self.drones.append(Drone(self, drone_dir=self.drone_dir, start_pos=start_pos, start_orn=start_orn))


    @property
    def num_drones(self):
        return len(self.drones)


    @property
    def states(self):
        """
        returns a list of states for each drone in the aviary
        """
        states = []
        for drone in self.drones:
            states.append(drone.state)

        states = np.stack(states, axis=0)

        return states


    def set_mode(self, flight_mode):
        """
        sets the flight mode for each drone
        """
        for drone in self.drones:
            drone.set_mode(flight_mode)


    def set_setpoints(self, setpoints):
        """
        commands each drone to go to a setpoint as specified in a list
        """
        for i, drone in enumerate(self.drones):
            drone.setpoint = setpoints[i]


    def step(self):
        """
        Steps the environment
        """
        if self.render:
            elapsed = time.time() - self.now
            time.sleep(max(self.period - elapsed, 0.))
            self.now = time.time()

            # print(f'RTF: {self.period / (elapsed + 1e-6)}')

        for drone in self.drones: drone.update()

        self.stepSimulation()
        self.step_count += 1
