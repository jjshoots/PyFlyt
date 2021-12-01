import time
import numpy as np
import multiprocessing as mp

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from pybullet_swarming.environment.drone import *

class Aviary(bullet_client.BulletClient):
    def __init__(self, start_pos: np.ndarray, start_orn: np.ndarray, render=False):
        super().__init__(p.GUI if render else p.DIRECT)

        # default physics looprate is 240 Hz
        self.period = 1. / 240.
        self.now = time.time()

        self.start_pos = start_pos
        self.start_orn = start_orn

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
            globalScaling=1.
        )

        # spawn drones
        self.drones = []
        for start_pos, start_orn in zip(self.start_pos, self.start_orn):
            self.drones.append(Drone(self, start_pos=start_pos, start_orn=start_orn))


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
