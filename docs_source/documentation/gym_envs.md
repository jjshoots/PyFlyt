# Gymnasium Environments

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/quadx_waypoint.gif
    :width: 65%
```

Natively, PyFlyt provides various default [Gymnasium](https://gymnasium.farama.org/) environments for testing reinforcement learning algorithms.
Usage is no different to how Gymnasium environments are initialized:

```python
import gymnasium
import PyFlyt.gym_envs # noqa

env = gymnasium.make("PyFlyt/QuadX-Hover-v0", render_mode="human")
obs = env.reset()

termination = False
truncation = False

while not termination or truncation:
    observation, reward, termination, truncation, info = env.step(env.action_space.sample())
```

```{toctree}
:hidden:
gym_envs/fixedwing_envs
gym_envs/rocket_envs
gym_envs/quadx_envs
```
