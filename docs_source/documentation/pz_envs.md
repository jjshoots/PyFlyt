# PettingZoo Environments

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/fixedwing_dogfight.gif
    :width: 65%
```

While PettingZoo may not provide the fastest API for multi agent reinforcement learning environments, it is the most widely supported and well maintained API.
For that reason, PyFlyt provides various default [PettingZoo](https://pettingzoo.farama.org/) environments for testing reinforcement learning algorithms.
All environments are by default [ParallelEnv](https://pettingzoo.farama.org/api/parallel/)s.
Usage is no different to how PettingZoo environments are initialized:

```python
from PyFlyt.pz_envs import MAQuadXHoverEnv
env = MAQuadXHoverEnv(render_mode="human")
observations, infos = env.reset(seed=42)

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

```{toctree}
:hidden:
pz_envs/ma_quadx_hover_env
pz_envs/ma_fixedwing_dogfight_env
```
