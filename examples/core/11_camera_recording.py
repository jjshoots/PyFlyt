"""Fly a drone around a circle and record its camera with exact camera poses.

The output folder holds `images/`, `depth/`, `poses.csv` (camera-to-world,
OpenCV camera axes) and `intrinsics.txt`, which can be used for visual odometry,
SLAM, novel-view synthesis or world-model training.
"""

import numpy as np

from PyFlyt.core import Aviary, CameraRecorder

# the starting position and orientations
start_pos = np.array([[1.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, np.pi / 2]])

# environment setup, a forward-facing camera tilted 20 degrees towards the ground
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=False,
    drone_type="quadx",
    drone_options=dict(
        use_camera=True,
        camera_angle_degrees=20,
        camera_FOV_degrees=90,
        camera_resolution=(240, 320),
        camera_fps=10,
    ),
)

# add some things to look at
for x, y in [(2.5, 0.0), (0.0, 2.5), (-2.5, 0.0), (0.0, -2.5)]:
    env.loadURDF("duck_vhacd.urdf", basePosition=[x, y, 0.0], globalScaling=10.0)

# call this to register all new bodies for collision
env.register_all_new_bodies()

# set to position control
env.set_mode(7)

# fly one climbing circle in 20 seconds while facing the direction of travel,
# saving a frame at 10 Hz (the default control rate is 120 Hz)
num_steps = 20 * 120
with CameraRecorder(env, "camera_recording", save_depth=True) as recorder:
    for i in range(num_steps):
        theta = 2 * np.pi * i / num_steps
        height = 1.0 + 0.5 * np.sin(theta)

        # the yaw setpoint is not wrapped by the controller,
        # so command the heading relative to the current yaw
        yaw = env.state(0)[1, 2]
        yaw_error = (theta + np.pi / 2 - yaw + np.pi) % (2 * np.pi) - np.pi
        env.set_setpoint(
            0, np.array([np.cos(theta), np.sin(theta), yaw + yaw_error, height])
        )
        env.step()

        if i % 12 == 0:
            recorder.capture()

print(f"Saved {recorder.frame_count} posed frames to camera_recording/")
env.disconnect()
