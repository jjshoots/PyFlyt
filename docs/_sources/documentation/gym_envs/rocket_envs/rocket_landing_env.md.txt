# `PyFlyt/Rocket-Landing-v0`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/rocket_landing.gif
    :width: 50%
```

## Task Description

The goal of this environment is to land a rocket falling at terminal velocity on a landing pad, with only 1% of fuel remaining.

## Usage

```python
import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/Rocket-Landing-v0", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.gym_envs.rocket_envs.rocket_landing_env.RocketLandingEnv
```
