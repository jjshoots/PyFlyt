import time
import numpy as np

from typing import List

from pybullet_swarming.flier.drone_controller import *

class Swarm_Controller():
    def __init__(self, URIs: List[str]):
        self.UAVs = [Drone_Controller(URI, in_swarm=True) for URI in URIs]
        time.sleep(3)
        print(f'Swarm with {self.num_drones} drones ready to go...')


    @property
    def num_drones(self):
        """number of drones in swarm"""
        return len(self.UAVs)


    @property
    def position_estimate(self):
        """list of position estimate for each drone"""
        pos_est = []
        for UAV in self.UAVs:
            pos_est.append(UAV.position_estimate - UAV.position_offset)

        return np.stack(pos_est, axis=0)


    @property
    def states(self):
        return self.position_estimate


    def set_pos_control(self, setting):
        """sets entire swarm to fly using pos control"""
        for UAV in self.UAVs:
            UAV.set_pos_control(setting)


    def go(self, masks):
        """arms and starts control loop for drones with masks set to 1"""
        assert len(masks) == len(self.UAVs), 'masks length must be equal to number of drones'

        for mask, UAV in zip(masks, self.UAVs):
            if mask:
                UAV.start()
            else:
                UAV.stop()


    def end(self):
        """disarms each drone and closes all connections"""
        for UAV in self.UAVs:
            UAV.end()


    def set_setpoints(self, setpoints: np.ndarray):
        """sets setpoints for each drone, setpoints must be ndarray where len(setpoints) == len(UAVs)"""
        assert len(setpoints) == len(self.UAVs), 'number of setpoints must be equal to number of drones'

        for setpoint, UAV in zip(setpoints, self.UAVs):
            UAV.set_setpoint(setpoint)


    def sleep(self, seconds: float):
        time.sleep(seconds)
