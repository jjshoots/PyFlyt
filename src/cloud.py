import math
import numpy as np
import matplotlib.pyplot as plt

from env.environment import *

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
        uav_pos = uav_states[:, -1, :]

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
        target[:, :2] = influence[:, :2]
        target[:, -1] = influence[:, -1]

        return target


if __name__ == '__main__':
    env = Environment(
        drone_dir='models/vehicles/',
        num_envs=1,
        max_steps=1000,
        render=True
        )

    cloud_control = Cloud()

    for i in range(10000):
        states = env.get_state()

        if i < 1000:
            print('1')
            velocity_setpoints = cloud_control.get_velocity_targets(states, np.array([0., 0., 2.]))
        elif i < 2000:
            print('2')
            velocity_setpoints = cloud_control.get_velocity_targets(states, np.array([2., 0., 2.]))
        elif i < 3000:
            print('3')
            velocity_setpoints = cloud_control.get_velocity_targets(states, np.array([2., 2., 2.]))
        elif i < 4000:
            print('4')
            velocity_setpoints = cloud_control.get_velocity_targets(states, np.array([0., 2., 2.]))
        elif i < 10000:
            print('5')
            velocity_setpoints = cloud_control.get_velocity_targets(states, np.array([0., 0., 2.]))

        env.step(velocity_setpoints)
