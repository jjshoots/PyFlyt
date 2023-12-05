# `MAQuadXHoverEnv`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/quadx_hover.gif
    :width: 50%
```

## Task Description

The goal of this environment is for all agents to hover at their starting positions for as long as possible.

## Usage

```python
from PyFlyt.pz_envs import MAQuadXHoverEnv

env = MAQuadXHoverEnv(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env.MAQuadXHoverEnv
```
