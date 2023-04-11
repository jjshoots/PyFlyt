# Aviary

## Description

The `aviary` is a handler for physics stepping, setpoint handling, collisions tracking, and much more.
It provides a common endpoint from where users may control drones or define tasks.
This page briefly goes through the key usage traits,
although more experienced users may wish to consult the whole class description at the bottom of this page,
or even (oh shit!) consult [the source code](https://github.com/jjshoots/PyFlyt/blob/master/PyFlyt/core/aviary.py).

## Setup Options

The `aviary` accepts these basic arguments:

- `start_pos` `(np.ndarray)`: an `[n, 3]` array for the starting X, Y, Z positions for each drone.
- `start_orn` `(np.ndarray)`: an `[n, 3]` array for the starting orientations for each drone, in terms of Euler angles.
- `drone_type` `(str)`: a _lowercase_ string representing what type of drone to spawn.
- `drone_type_mappings` `(dict(str: DroneClass))`: a dictionary mapping of `{str: DroneClass}` for spawning custom drones.
- `drone_options` `(dict(str: Any))`: dictionary mapping of custom parameters for each drone.
- `render` `(bool)`: a boolean whether to render the simulation.
- `physics_hz` `(int)`: physics looprate (not recommended to be changed).
- `worldScale` `(float)`: how big to spawn the floor.
- `seed` `(None | int)`: optional int for seeding the simulation RNG.

Some of these are elaborated below.

### Drone Type

The drone type is a _lowercase_ string that defines the type of drone(s) to spawn in the `aviary`.
By default, PyFlyt ships with three different drones, listed under the [drones](../core/drones) section.
These are:

1. `Fixedwing`
2. `Rocket`
3. `QuadX`

So, for example, to spawn the `Fixedwing` drone instead of the `QuadX`, simply do:

```python
...
env = Aviary(..., drone_type="fixedwing")
...
```

### Drone Type Mappings

For custom drones not shipped in PyFlyt (such as the [`RocketBrick`](https://github.com/jjshoots/PyFlyt/tree/master/examples/core/custom_uavs)), you can call them into the `aviary` via:

```python
...
# import the drone class so it's callable
from ... import MyCustomDrone

# define a new drone so the aviary can call it
# note we do not instantiate the object, this is only a class pointer
drone_type_mappings = dict()
drone_type_mappings["mycustomdrone"] = MyCustomDrone

# pass the relevant arguments to the `aviary`
env = Aviary(..., drone_type="mycustomdrone", drone_type_mappings=drone_type_mappings)
...
```

### Drone Options

Various drones can have different instantiation parameters, such as the `Fixedwing` drone having a configurable starting velocity which the `QuadX` drone does not have.

To define these custom parameters, use the `drone_options` argument as so:

```python
...
# define the parameters for the underlying drone
drone_options = dict()
drone_options["starting_velocity"] = np.array([3.0, 0.0, 3.0])

# pass the relevant arguments to the `aviary`
env = Aviary(..., drone_type="fixedwing", drone_options=drone_options)
...
```

## Multi Drone Setup

To spawn multiple drones with different types and parameters for each, the lists of `drone_type` and `drone_options` can be used instead to give each drone a unique set of parameters.
For example:

```python
...
# the starting position and orientations
start_pos = np.array([[0.0, 5.0, 5.0], [3.0, 3.0, 1.0], [5.0, 0.0, 1.0]])
start_orn = np.zeros_like(start_pos)

# spawn different types of drones
drone_type = ["rocket", "quadx", "fixedwing"]

# individual spawn options for each drone
rocket_options = dict()
quadx_options = dict(use_camera=True, drone_model="primitive_drone")
fixedwing_options = dict(starting_velocity=np.array([0.0, 0.0, 0.0]))
drone_options = [rocket_options, quadx_options, fixedwing_options]

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type=drone_type,
    drone_options=drone_options,
)

# set quadx to position control, rocket and fixedwing as nothing
env.set_mode([0, 7, 0])

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()
```

Here, we spawn three drones, a `Rocket`, a `QuadX` and a `Fixedwing`, at three different positions.
Each of the drones has different options.

## Accessing Individual Drones

All drones are stored within a `drones` attribute (very creatively named).
This allows raw access for any `drone` from outside the `aviary`.

```python
...
# instantiate the aviary
env = Aviary(...)

# assuming there are 3 drones and the last drone has a camera,
# we can get the camera image like so
rgbImg = env.drones[-1].rgbImg
```

## Looprates

By default, PyFlyt runs the simulation at 240 Hz - the default for a PyBullet environment.
Although not recommended, this can be changed via the `physics_hz` argument.

The various drones within the environment can also be configured to have a control looprate different to the physics looprate.
This is configured through the `drone_options` argument, like so:

```python
...
# define a control looprate
drone_options = dict()
drone_options["control_hz"] = 120

# pass the relevant arguments to the `aviary`
env = Aviary(..., drone_type="quadx", drone_options=drone_options)
...
```

__All control looprates must be a common denominator of the physics looprate.__
For instance, for a physics looprate of 240 Hz, a control looprate of 60 Hz is valid, but 50 is not since `240 % 50 != 0`.

### Single Drone Physics Stepping

Every call to `step` of the `aviary` steps the simulation enough times for one control loop to elapse.
For example, if the physics looprate is 240 Hz and the control looprate is 120 Hz, each call to `step` steps the physics in the environment 2 times.

### Multi Drone Physics Stepping

In a multi drone setting, it is possible for various drones to have various looprates.
The caveat here is that, when control looprates are arranged in ascending order, the `i+1`th looprate must be a round multiple of the `i`th looprate.
For instance, this is a valid set of looprates:

```python
...
drone_options = []
drone_options.append(dict(control_hz=60))
drone_options.append(dict(control_hz=30))
drone_options.append(dict(control_hz=120))
...
```

... but this is not:

```python
...
drone_options = []
drone_options.append(dict(control_hz=40))
drone_options.append(dict(control_hz=30))
drone_options.append(dict(control_hz=120))
...
```

In the first sample, when arranged in ascending order, we get a list of `looprates = [30, 60, 120]`.
When we do `looprate[1:] / looprate[:-1]`, we get `[2, 2]`, which is all integers.
This is valid.

In the second sample, when arranged in ascending order, this list is `looprates = [30, 40, 120]`.
Similarly, when we do `looprate[1:] / looprate[:-1]`, we get `[1.25, 3]`, which is __not__ all integers.
This is __invalid__.

## Class Descriptions
```{eval-rst}
.. autoclass:: PyFlyt.core.Aviary
    :members:
```
