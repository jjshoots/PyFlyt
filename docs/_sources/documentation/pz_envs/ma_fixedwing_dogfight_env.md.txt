# `MAFixedwingDogfightEnv`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/fixedwing_dogfight.gif
    :width: 50%
```

## Task Description

This is a reinforcement learning environment for training AI agents to perform aerial dogfighting.

## Usage

```python
from PyFlyt.pz_envs import MAFixedwingDogfightEnv

env = MAFixedwingDogfightEnv(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

## Environment Rules

- This is a cannons only environment. Meaning there are no missiles. An agent has to point its nose directly at the enemy for it to be considered a hit.
- The gun is only effective within `lethal range`. Outside of this range, the gun deals no damage.
- The gun automatically fires when it can, there is no action for the agent to fire the weapon. This is similar to many [fire control systems](https://en.wikipedia.org/wiki/Fire-control_system) on modern aircraft.
- An agent loses if it:
  a) Hits anything
  b) Flies out of bounds
  c) Loses all its health

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.pz_envs.fixedwing_envs.ma_fixedwing_dogfight_env.MAFixedwingDogfightEnv
```
