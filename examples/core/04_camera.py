"""Spawn a single drone and get its camera image."""
import matplotlib.pyplot as plt
import numpy as np

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type="quadx",
    drone_options=dict(use_camera=True),
)

# set to velocity control
env.set_mode(6)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(100):
    env.step()

# get the camera image and show it
RGBA_img = env.drones[0].rgbaImg
DEPTH_img = env.drones[0].depthImg
SEG_img = env.drones[0].segImg

plt.imshow(RGBA_img)
# plt.show()
plt.imshow(DEPTH_img)
# plt.show()
plt.imshow(SEG_img)
plt.show()
