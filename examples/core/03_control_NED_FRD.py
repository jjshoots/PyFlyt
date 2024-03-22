"""Spawn a single drone, then command it to go to two setpoints consecutively, and plots the xyz output."""
import matplotlib.pyplot as plt
import numpy as np

from PyFlyt.core import Aviary

# initialize the log
log = np.zeros((1000, 3), dtype=np.float32)

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, -1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    darw_local_axis=True,
    drone_type="quadx",
    orn_conv="NED_FRD",
)

# set to position control
env.set_mode(7)

# for the first 500 steps, go to x=1, y=0, z=-1
setpoint = np.array([1.0, 0.0, 0.0, -1.0])
env.set_setpoint(0, setpoint)

for i in range(500):
    env.step()

    # record the linear position state
    log[i] = env.state(0)[-1]

# for the next 500 steps, go to x=0, y=0, z=-2, rotate 45 degrees
setpoint = np.array([0.0, 0.0, np.pi / 4, -2.0])
env.set_setpoint(0, setpoint)

for i in range(500, 1000):
    env.step()

    # record the linear position state
    log[i] = env.state(0)[-1]

# plot stuff out
plt.plot(np.arange(1000), log)
plt.show()
