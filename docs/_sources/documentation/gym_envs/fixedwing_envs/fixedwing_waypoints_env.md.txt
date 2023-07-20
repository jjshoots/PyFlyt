# `PyFlyt/Fixedwing-Waypoints-v0`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/fixedwing_waypoint.gif
    :width: 50%
```

## Task Description

The goal of this environment is to fly a fixedwing aircraft towards a set of waypoints as fast as possible.

## Usage

```python
import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env.FixedwingWaypointsEnv
```
