# `PyFlyt/Rocket-Landing-v0`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/rocket_landing.gif
    :width: 50%
```

## Task Description

The goal of this environment is to land a rocket falling at terminal velocity on a landing pad, with only 1% of fuel remaining.

## Usage

```python
import PyFlyt.gym_envs
env = gymnasium.make("PyFlyt/Rocket-Landing-v0")
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.gym_envs.rocket_envs.rocket_landing_env.RocketLandingEnv
```
