# Rocket

## Model Description

The rocket describes a 1:10th scale SpaceX Falcon 9 v1.2 first stage and interstage.
Mass and geometry properties were extracted from [Space Launch Report datasheet](https://web.archive.org/web/20170825204357/spacelaunchreport.com/falcon9ft.html#f9stglog).

## Control Mode

Only one control mode is provided.
Setpoints correspond to:

- finlet x deflection
- finlet y deflection
- finlet yaw
- ignition
- throttle
- booster gimbal axis 1
- booster gimbal axis 2

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.drones.Rocket
    :members:
```
