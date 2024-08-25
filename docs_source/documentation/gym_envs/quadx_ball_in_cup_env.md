# `PyFlyt/QuadX-Ball-In-Cup-v2`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/quadx_pole_balance.gif
    :width: 50%
```

## Task Description

The goal is to swing up a suspended ball onto the drone, and then bring it to the starting position.

This environment was introduced and contributed by GitHub user [defrag-bambino](https://github.com/jjshoots/PyFlyt/pull/58).

## Usage

```python
import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/QuadX-Ball-In-Cup-v2", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.gym_envs.quadx_envs.quadx_pole_balance_env.QuadXBallInCupEnv
```
