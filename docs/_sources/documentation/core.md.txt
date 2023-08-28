# Core

```{toctree}
:hidden:
core/aviary
core/wind
core/drones
core/abstractions
```

## Core API

The core API of PyFlyt looks something like the following:

```python
"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""
# Step 1: import things
import numpy as np
from PyFlyt.core import Aviary

# Step 2: define starting positions and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# Step 3: instantiate aviary
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx")

# Step 4: (Optional) define control mode to use for drone
env.set_mode(7)

# Step 5: (Optional) define a setpoint for the first drone (at index 0) in the aviary
setpoint = np.array([1.0, 0.0, 0.0, 1.0])
env.set_setpoint(0, setpoint)

# Step 6: step the physics
for i in range(1000):
    env.step()

# Gracefully close
env.close()
```

1. At the start, we import `numpy` and the `Aviary`.
2. Then, we define the starting positions and orientations as an `[n, 3]` array each, for `n` number of drones.
3. The `Aviary` is instantiated by passing the starting positions and orientations, as well as a string representing the drone type to use.
4. It is possible to define base flight modes for the drones, this is elaborated in the [drone](core/abstractions/base_drone.md) section.
5. We set a setpoint for the drone to reach, in this case it is `(x, y, yaw, z) = (1 meter, 0, 1 radian, 0)`.
6. Finally, we step through the physics.

Drone setpoints are __persistent__ attributes - you don't have to repeatedly set them at every step if there is no desire to change setpoints.

The `Aviary` itself is a highly flexible multi-drone handling environment.
The extent of its capabilities are elaborated in [its relevant section](core/aviary.md).

## General Architecture

Loosely put, PyFlyt has the following architecture:

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/pyflyt_architecture.png
    :width: 70%
```

At the core lies the [`Aviary`](core/aviary), serving as a domain for all [`drones`](core/drones).
Each drone is defined by its own Python class, allowing for user-defined drones to be built.
Drones can be built from the ground up, or can be an amalgamation of various component [`abstractions`](core/abstractions).

## Axis Conventions

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/duck_frame.png
    :width: 70%
```

Frame conventions are essential for describing the orientation and movement of objects in space.
PyFlyt uses two reference frames, the ground frame (subscript `G`) and the body frame (subscript `B`), shown above.
The ground frame defines three axes relative to the local horizontal plane and the direction of gravity.
The body frame defines the axes relative to the body of the drone.

We utilize the [ENU](https://en.wikipedia.org/wiki/Geodetic_datum#Local_east.2C_north.2C_up_.28ENU.29_coordinates) frame convention, where for the ground frame,
the X-axis points East, Y-axis points North, and Z-axis points Up (East and North are just cardinal references).
On the body frame, this convention defines the X, Y, and Z axes to point out the front, left, and upward direction from the drone.
