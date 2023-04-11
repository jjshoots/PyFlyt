# Base Drone Class

## Description

The `DroneClass` is an abstract class that all drones should inherit from.
It provides the basic functionalities for interfacing with the [`aviary`](../../core/aviary).

## Key Usage Information

### Disable Artificial Damping

Call `.disable_artificial_damping()` within `reset()` to disable the artificial damping that PyBullet imposes.
This allows for more accurate drag simulation.

### Get Joint Info for Debugging

Call `.get_joint_info()` to print out all joint infos and their relevant IDs.
This is useful when constructing and debugging your custom drone.

## Class Descriptions
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.DroneClass
    :members:
```
