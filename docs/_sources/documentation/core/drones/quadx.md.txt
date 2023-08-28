# QuadX

## Model Description

The QuadX UAV describes a multirotor UAV in the Quad-X configuration as described by [ArduPilot](https://ardupilot.org/copter/docs/connect-escs-and-motors.html) and [PX4](https://docs.px4.io/main/en/airframes/airframe_reference.html#quadrotor-x).
It consists of four motors with implementations for cascaded PID controllers.

## Variants

Two different models of the QuadX drone are provided.
This option can be selected through the `drone_model` parameter.
The available models are:

- `cf2x` (default)
- `primitive_drone`

## Control Modes

The control mode of the QuadX can be changed using the [`Aviary.set_mode`](https://jjshoots.github.io/PyFlyt/documentation/core/aviary.html#PyFlyt.core.Aviary.set_mode) or specifically calling [`set_mode`](https://jjshoots.github.io/PyFlyt/documentation/core/drones/quadx.html#PyFlyt.core.drones.QuadX.set_mode) on the drone instance itself.
Inspired by [pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones).
The various modes available are documented [below](https://jjshoots.github.io/PyFlyt/documentation/core/drones/quadx.html#PyFlyt.core.drones.QuadX.set_mode).

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.drones.QuadX
    :members:
```
