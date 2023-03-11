# PyFlyt - UAV Flight Simulator Gymnasium Environments for Reinforcement Learning Research

<p align="center">
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/pyflyt_cover_photo.png?raw=true" width="800px"/>
</p>

This is a library for testing reinforcement learning algorithms on UAVs.
This repo is still under development.
We are also actively looking for users and developers, if this sounds like you, don't hesitate to get in touch!

PyFlyt currently supports two separate UAV platforms:
- QuadX UAV
  - Inspired by [the original pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones)
  - Quadrotor UAV in the X configuration
  - Actual full cascaded PID flight controller implementations for each drone.
  - Actual motor RPM simulation using first order differential equation.
  - Modular control structure
  - For developers - 8 implemented flight modes that use tuned cascaded PID flight controllers, available in `PyFlyt/core/drones/quadx.py`.

- Fixedwing UAV
  - Flight model designed for a small fixed wing UAV (< 10 Kg)
  - Assumes a conventional tube and wing design
  - Single puller propeller with thrust line passing through CG
  - Aerofoil characteristics derived from the paper: [*Real-time modeling of agile fixed-wing UAV aerodynamics*](https://ieeexplore.ieee.org/document/7152411)

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](#environments)
  - [`PyFlyt/QuadX-Hover-v0`](#pyflytquadx-hover-v0)
  - [`PyFlyt/QuadX-Waypoints-v0`](#pyflytquadx-waypoints-v0)
  - [`PyFlyt/Fixedwing-Waypoints-v0`](#pyflytfixedwing-waypoints-v0)


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

# omit the below line to remove rendering and let
# the simulation go as fast as possible
env.render()
obs = env.reset()

done = False
while not done:
    observation, reward, termination, truncation, info = env.step(env.action_space.sample())
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
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/quadx_waypoint.gif?raw=true" width="500px"/>
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
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/fixedwing_waypoint.gif?raw=true" width="500px"/>
</p>
