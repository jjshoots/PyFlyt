# `PyFlyt/Fixedwing-Waypoints-v3`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/fixedwing_waypoint.gif
    :width: 50%
```

## Task Description

The goal of this environment is to fly a fixedwing aircraft towards a set of waypoints as fast as possible.

## Usage

```python
import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/Fixedwing-Waypoints-v3", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Flattening the Environment

This environment uses the [`Dict`](https://gymnasium.farama.org/api/spaces/composite/#dict) and [`Sequence`](https://gymnasium.farama.org/api/spaces/composite/#sequence) spaces from `Gymnasium`, which are spaces with non-constant sizes.
This allows them to have complete observability without observation padding while making (human-)readability easier.
However, this results in them not being compatible with most popular reinforcement learning libraries like [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) without custom wrappers.
If you would like to use this environment with those libraries, you can flatten the environment using the `FlattenWaypointEnv` wrapper, where the argument `context_length` specifies how many immediate targets are included in the observation.

```python
import gymnasium
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

env = gymnasium.make("PyFlyt/Fixedwing-Waypoints-v3", render_mode="human")
env = FlattenWaypointEnv(env, context_length=2)

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env.FixedwingWaypointsEnv
```
