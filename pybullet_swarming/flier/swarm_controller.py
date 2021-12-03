import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from typing import List

from pybullet_swarming.flier.drone_controller import *

class Swarm_Controller():
    """
    Class for controlling a swarm of Crazyflie UAVs.
    Realistically, only about 5 drones per dongle supported.
        Requied: URIs
        - correspond to list of URIs for the drones.
        Optional: start_pos and start_orn
        - correspond to desired initial positions of drones
            don't have to worry about drone numbering,
            assignment is solved using Hungarian algorithm.
            Only the last value of start_orn (yaw) is used.
            Position control automatically set if this is supplied.
    """
    def __init__(self, URIs: List[str], start_pos=None, start_orn=None):
        self.UAVs = [Drone_Controller(URI, in_swarm=True) for URI in URIs]
        time.sleep(1)

        # if start pos is given, reassign to get drones to their positions automatically
        if start_pos is not None and start_orn is not None:
            assert start_pos.shape == start_orn.shape, 'start_pos must have same shape as start_orn'
            assert len(start_pos) == self.num_drones, 'must have same number of drones as number of drones'
            assert start_pos[0].shape == 3, 'start pos must have only xyz, start orn must have only pqr'

            # compute cost matrix
            cost = abs(np.expand_dims(self.states, axis=0) - np.expand_dims(start_pos, axis=1))
            cost = np.sum(cost, axis=-1)

            # compute optimal assignment using Hungarian algo
            _, reassignment = linear_sum_assignment(cost)
            self.UAVs = [self.UAVs[i] for i in reassignment]

            # send setpoints
            setpoints = np.concatenate((start_pos, np.expand_dims(start_orn[:, -1], axis=-1)), axis=-1)
            self.set_setpoints(setpoints)
            self.set_pos_control(True)

        time.sleep(1)
        print(f'Swarm with {self.num_drones} drones ready to go...')
        time.sleep(1)



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
        time.sleep(1)


    def set_setpoints(self, setpoints: np.ndarray):
        """sets setpoints for each drone, setpoints must be ndarray where len(setpoints) == len(UAVs)"""
        assert len(setpoints) == len(self.UAVs), 'number of setpoints must be equal to number of drones'

        for setpoint, UAV in zip(setpoints, self.UAVs):
            UAV.set_setpoint(setpoint)


    def sleep(self, seconds: float):
        time.sleep(seconds)
