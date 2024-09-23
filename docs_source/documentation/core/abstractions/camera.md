# Camera

## Description

The `Camera` component simulates a camera attached to a link on the drone.
The camera itself can be gimballed to achieve a horizon lock effect.
In addition, the field-of-view (FOV), tilt angle, resolution, and offset distance from the main link can be adjusted.
On image capture, the camera returns an RGBA image, a depth map, and a segmentation map with pixel values representing the IDs of objects in the environment.

### Camera FPS

Camera capture consumes a significant amount of overhead, see [this github issue](https://github.com/jjshoots/PyFlyt/pull/43).
It is not recommended to perform camera capture on every physics step.

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.Camera
    :members:
```
