# Obstacle

## Description

The `Obstacle` component represents a static URDF object spawned in the `Aviary`.
Obstacles are intended for building scene geometry (walls, boxes, cylinders, etc.) that drones can perceive and collide with, without altering the drone abstractions themselves.

PyFlyt ships with a small set of primitive obstacles --- `"cube"`, `"cylinder"`, and `"sphere"` --- which can be loaded by name.
Arbitrary URDFs may also be loaded by passing a path.

Obstacles are always spawned with `useFixedBase=True`, so the URDF's mass is ignored and the body will not move under gravity or contact.

## Usage

Obstacles are most commonly added through the [`Aviary.add_obstacle`](aviary) helper, which constructs the `Obstacle`, tracks it for collision bookkeeping, and respawns it after every `reset()`:

```python
import numpy as np
from PyFlyt.core import Aviary

env = Aviary(
    start_pos=np.array([[0.0, 0.0, 1.0]]),
    start_orn=np.array([[0.0, 0.0, 0.0]]),
    drone_type="quadx",
)

# built-in primitive obstacle
env.add_obstacle("cube", position=np.array([2.0, 0.0, 0.5]))

# custom URDF, with rotation and scaling
env.add_obstacle(
    "/path/to/wall.urdf",
    position=np.array([4.0, 0.0, 1.0]),
    orientation=np.array([0.0, 0.0, np.pi / 2]),
    scale=1.5,
)
```

Collisions between drones and obstacles are reflected in `env.contact_array` just like any other contact.

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.Obstacle
    :members:
```
