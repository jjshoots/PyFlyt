# `PyFlyt/QuadX-Waypoints-v0`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/quadx_waypoint.gif
    :width: 50%
```

## Task Description

The goal of this environment is to fly a quadrotor aircraft towards a set of waypoints as fast as possible.

## Usage

```python
import PyFlyt.gym_envs
env = gymnasium.make("PyFlyt/QuadX-Waypoints-v0")
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env.QuadXWaypointsEnv
```
