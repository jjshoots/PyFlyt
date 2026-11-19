# Camera Recorder

## Description

`CameraRecorder` saves the onboard camera of a drone as a posed image sequence: every image comes with the exact camera-to-world pose it was rendered from, plus the pinhole intrinsics and, optionally, metric depth.
This is the data layout expected by most visual odometry, SLAM, novel-view synthesis (NeRF, Gaussian splatting) and world-model training pipelines.

```
<output_dir>/
    images/000000.png, 000001.png, ...
    depth/000000.npy, ...       (optional, metric z-depth in metres, float32)
    poses.csv                   timestamp,tx,ty,tz,qx,qy,qz,qw
    intrinsics.txt              fx fy cx cy
```

- Poses are camera-to-world transforms with OpenCV camera axes (x right, y down, z forward) in the PyBullet world frame (z up); timestamps are simulation time in seconds.
- The pose is recovered by inverting the view matrix used to render the frame, so it respects every `Camera` option (gimbal, tilt angle, position offset, tracking camera).
- Frames are always rendered with PyBullet's TinyRenderer, so the intrinsics do not change between `render=True` and `render=False`.

## Usage

The drone must be spawned with `use_camera=True`.

```python
import numpy as np
from PyFlyt.core import Aviary, CameraRecorder

env = Aviary(
    start_pos=np.array([[0.0, 0.0, 1.0]]),
    start_orn=np.zeros((1, 3)),
    render=False,
    drone_type="quadx",
    drone_options=dict(use_camera=True, camera_resolution=(240, 320), camera_fps=10),
)
env.set_mode(7)
env.set_setpoint(0, np.array([2.0, 1.0, 0.0, 1.5]))

with CameraRecorder(env, "flight_0", save_depth=True) as recorder:
    for i in range(1200):
        env.step()
        if i % 12 == 0:  # 10 Hz at the default 120 Hz control rate
            recorder.capture()
```

A complete example is in `examples/core/11_camera_recording.py`.

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.utils.camera_recorder.CameraRecorder
    :members:
```
