import os
import time
import numpy as np
from signal import signal, SIGINT

from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.environment.simulator import Simulator

def shutdown_handler(*_):
    print("ctrl-c invoked")
    os._exit(1)


class Cloud():
    """
    Wrapper for cloud control from swarm drones
    """
    def __init__(self):
        self.repellant_scale = .01
        self.repellant_distance = .3
        self.attraction_scale = 1000.
        self.peak_per_axis_velocity = .5


    def get_velocity_targets(self, uav_states, target_pos):
        uav_pos = uav_states[:, :-1]

        # compute repellant
        implicit = np.expand_dims(uav_pos, axis=1)
        explicit = np.expand_dims(uav_pos, axis=0)

        difference = (implicit - explicit)
        difference[difference > self.repellant_distance] = np.inf
        difference[difference == 0.] = np.inf
        repellant = np.sum(self.repellant_scale / (difference + 1e-6), axis=1)
        repellant = np.clip(repellant, -self.peak_per_axis_velocity, self.peak_per_axis_velocity)

        # compute attraction
        difference = self.attraction_scale * -(uav_pos - np.expand_dims(target_pos, axis=0))
        attraction = np.clip(difference, -self.peak_per_axis_velocity, self.peak_per_axis_velocity)

        influence = repellant + attraction

        target = np.zeros((influence.shape[0], 4), dtype=float)
        target[:, :-1] = influence

        return target


if __name__ == '__main__':
    check_venv()
    signal(SIGINT, shutdown_handler)

    # here we spawn drones in a 2x2x1 grid
    drones_per_len = 2
    drones_per_height = 1

    lin_range = [-.2, .2]
    lin_range = np.linspace(start=lin_range[0], stop=lin_range[1], num=drones_per_len)
    height_range = [.1, .1]
    height_range = np.linspace(start=height_range[0], stop=height_range[1], num=drones_per_height)

    grid_x, grid_y, grid_z = np.meshgrid(lin_range, lin_range, height_range)
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    start_pos = np.stack([grid_x, grid_y, grid_z], axis=-1)
    start_orn = np.zeros_like(start_pos)

    swarm = Simulator(start_pos=start_pos, start_orn=start_orn)

    cloud_control = Cloud()

    for i in range(10000):
        states = swarm.states

        setpoints = np.zeros((swarm.num_drones, 4))
        if i < 1000:
            setpoints = cloud_control.get_velocity_targets(states, np.array([0., 0., 2.]))
        elif i < 2000:
            setpoints = cloud_control.get_velocity_targets(states, np.array([2., 0., 2.]))
        elif i < 3000:
            setpoints = cloud_control.get_velocity_targets(states, np.array([2., 2., 2.]))
        elif i < 4000:
            setpoints = cloud_control.get_velocity_targets(states, np.array([0., 2., 2.]))
        elif i < 10000:
            setpoints = cloud_control.get_velocity_targets(states, np.array([0., 0., 2.]))

        swarm.set_setpoints(setpoints)
        swarm.step()
