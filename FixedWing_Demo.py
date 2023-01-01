"""Spawn a single fixed wing UAV on x=0, y=0, z=50, with 0 rpy."""
import numpy as np
import gymnasium
import PyFlyt.gym_envs
import time
from PyFlyt.core import Aviary, loadOBJ, obj_collision, obj_visual

from pyPS4Controller.controller import Controller
from threading import Thread, Event


class MyController(Controller):

    def __init__(self, **kwargs):
        Controller.__init__(self, **kwargs)

    def on_R3_down(self, value):
        global cmds

        value = value / 32767

        cmds[0] = value
        return value

    def on_R3_up(self, value):
        global cmds

        value = value / 32767

        cmds[0] = value
        return value

    def on_R3_left(self, value):
        global cmds

        value = value / 32767

        cmds[1] = value
        return value

    def on_R3_right(self, value):
        global cmds

        value = value / 32767

        cmds[1] = value
        return value

    def on_L3_left(self, value):
        global cmds

        value = value / 32767

        cmds[2] = value
        return value

    def on_L3_right(self, value):
        global cmds

        value = value / 32767

        cmds[2] = value
        return value

    def on_R2_press(self, value):
        global cmds

        value = value / 32767

        cmds[3] = value
        return value


def readDS4():
    controller = MyController(
        interface="/dev/input/js0", connecting_using_ds4drv=False)
    controller.listen()


t = Thread(target=readDS4, args=())
t.start()

cmds = [0, 0, 0, 0]


env = gymnasium.make("PyFlyt/SimpleWaypointEnv-v0", render_mode=None,
                     flight_dome_size=500, num_targets=1, goal_reach_distance=1)
env.reset()


# simulate for 1000 steps (1000/120 ~= 8 seconds)
while True:
    state, reward, termination, truncation, info = env.step(cmds)

    if termination or truncation:
        env.reset()

