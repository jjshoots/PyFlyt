# PyFlyt - UAV Flight Simulator Gymnasium Environments for Reinforcement Learning Research

This is a library for testing reinforcement learning algorithms on UAVs.
This repo is still under development.

PyFlyt currently supports two separate UAV platforms:
- QuadX UAV
  - Inspired by [the original pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones)
  - Quadrotor UAV in the X configuration
  - Actual full cascaded PID flight controller implementations for each drone.
  - Actual motor RPM simulation using first order differential equation.
  - Modular control structure
  - For developers - 8 implemented flight modes that use tuned cascaded PID flight controllers, available in `PyFlyt/core/drones/quadx.py`.

- Fixedwing UAV
  -

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](#environments)
  - [`PyFlyt/QuadX-Hover-v0`](#pyflytquadx-hover-v0)
  - [`PyFlyt/QuadX-Waypoints-v0`](#pyflytquadx-waypoints-v0)
  - [`PyFlyt/Fixedwing-Waypoints-v0`](#pyflytfixedwing-waypoints-v0)
- [Non-Gymnasium examples](#non-gymnasium-examples)
  - [Simulation Only](#simulation-only)
    - [`sim_single.py`](#sim_singlepy)
    - [`sim_swarm.py`](#sim_swarmpy)
    - [`sim_cube.py`](#sim_cubepy)
  - [Hardware Only](#hardware-only)
    - [`fly_single.py`](#fly_singlepy)
    - [`fly_swarm.py`](#fly_swarmpy)
  - [Simulation or Hardware](#simulation-or-hardware)
    - [`sim_n_fly_single.py`](#sim_n_fly_singlepy)
    - [`sim_n_fly_multiple.py`](#sim_n_fly_multiplepy)
    - [`sim_n_fly_cube_from_scratch.py`](#sim_n_fly_cube_from_scratchpy)


## Installation

```
pip3 install pyflyt
```

## Usage

Usage is similar to any other Gymnasium and (soon) PettingZoo environment:

```py
import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/QuadX-Hover-v0")

# ommit the below line to remove renderring and let
# the simulation go as fast as possible
env.render()
obs = env.reset()

done = False
while not done:
    observation, reward, termination, truncation, info = env.step(env.observation_space.sample())
```

## Environments

### `PyFlyt/QuadX-Hover-v0`

A simple environment where an agent can learn to hover.
The environment ends when either the quadcopter collides with the ground or exits the permitted flight dome.

```py
env = gymnasium.make(
  "PyFlyt/QuadX-Hover-v0",
  flight_dome_size: float = 3.0,
  max_duration_seconds: float = 10.0,
  angle_representation: str = "quaternion",
  agent_hz: int = 40,
  render_mode: None | str = None,
)
```

> `angle_representation` can be either `"quaternion"` or `"euler"`.
>
> `render_mode` can be either `"human"` or `None`.

### `PyFlyt/QuadX-Waypoints-v0`

A simple environment where the goal is to fly the quadrotor to a collection of random waypoints in space within the permitted flight dome.
The environment ends when either the quadrotor collides with the ground or exits the permitted flight dome.

```py
env = gymnasium.make(
  "PyFlyt/QuadX-Waypoints-v0",
  sparse_reward: bool = False,
  num_targets: int = 4,
  use_yaw_targets: bool = False,
  goal_reach_distance: float = 0.2,
  goal_reach_angle: float = 0.1,
  flight_dome_size: float = 5.0,
  max_duration_seconds: float = 10.0,
  angle_representation: str = "quaternion",
  agent_hz: int = 30,
  render_mode: None | str = None,
)
```

> `angle_representation` can be either `"quaternion"` or `"euler"`.
>
> `render_mode` can be either `"human"` or `None`.

<p align="center">
    <img src="/readme_assets/quadx_waypoint.gif" width="500px"/>
</p>

### `PyFlyt/Fixedwing-Waypoints-v0`

A simple environment where the goal is to fly a fixedwing aircraft towards set of random waypionts in space within the permitted flight dome.
The environment ends when either the aircraft collides with the ground or exits the permitted flight dome.

```py
env = gymnasium.make(
  "PyFlyt/Fixedwing-Waypoints-v0",
  sparse_reward: bool = False,
  num_targets: int = 4,
  goal_reach_distance: float = 2.0,
  flight_dome_size: float = 100.0,
  max_duration_seconds: float = 120.0,
  angle_representation: str = "quaternion",
  agent_hz: int = 30,
  render_mode: None | str = None,
)
```

> `angle_representation` can be either `"quaternion"` or `"euler"`.
>
> `render_mode` can be either `"human"` or `None`.

<p align="center">
    <img src="/readme_assets/fixedwing_waypoint.gif" width="500px"/>
</p>

****
****

## Non-Gymnasium examples

If you're not interested in RL but want to use the library for your own research, we provide a bunch of example code in `examples/` that you can run with `python3 examples/***.py` in _macOS_ and _Linux_.

PyFlyt also has naive support for flying real Crazyflie drones.
These examples are provided under `examples/crazyflie/***.py`.
The library is built using CrazyFlie drones, check out the [documentation](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/).
These scripts are built with as little dependencies as possible, but enable interfacing with real (using the CrazyPA module) or virtual drones easy.

### Simulation Only

#### `sim_single.py`
Simulates a single drone in the pybullet env with position control.
<p align="center">
    <img src="/readme_assets/simulate_single.gif" width="500px"/>
</p>

#### `sim_swarm.py`
Simulates a swarm of drones in the pybullet env with velocity control.
<p align="center">
    <img src="/readme_assets/simulate_swarm.gif" width="500px"/>
</p>

#### `sim_cube.py`
Simulates a swarm of drones in a spinning cube.
<p align="center">
    <img src="/readme_assets/simulate_cube.gif" width="500px"/>
</p>

### Hardware Only

#### `fly_single.py`
Flies a real Crazyflie, check out the [documentation](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/) and [how to connect](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/#config-client) to get your URI(s) and modify them in line 18.

#### `fly_swarm.py`
Flies a real Crazyflie swarm, same as the previous example, but now takes in a list of URIs.

### Simulation or Hardware

#### `sim_n_fly_single.py`
Simple script that can be used to fly a single crazyflie in sim or with a real drone using either the `--hardware` or `--simulate` args.

#### `sim_n_fly_multiple.py`
Simple script that can be used to fly a swarm of crazyflies in sim or with real drones using either the `--hardware` or `--simulate` args.

#### `sim_n_fly_cube_from_scratch.py`
Simple script that can be used to fly a swarm of crazyflies in sim or with real drones using either the `--hardware` or `--simulate` args, and forms the same spinning cube from takeoff as in `sim_cube.py`.

