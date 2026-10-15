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
    drone_options=dict(
        use_camera=True,
        camera_angle_degrees=30,
        camera_FOV_degrees=110,
        camera_fps=30,
        # by default, camera_fps=control_hz,
        # but this makes the simulation slow,
        # see https://github.com/jjshoots/PyFlyt/pull/43
    ),
)

# set to velocity control
env.set_mode(6)

# simulate for 100 steps (100/120 ~= 1 second)
for i in range(100):
    env.step()

# get the camera image and show it
RGBA_img = env.drones[0].rgbaImg
DEPTH_img = env.drones[0].depthImg
SEG_img = env.drones[0].segImg

# show the three images side by side
fig, axes = plt.subplots(1, 3)
for ax, img, title in zip(
    axes, [RGBA_img, DEPTH_img, SEG_img], ["RGBA", "Depth", "Segmentation"]
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
plt.show()
