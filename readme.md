![GitHub CI](https://github.com/jjshoots/PyFlyt/actions/workflows/linux-test.yml/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![hits](https://hits.dwyl.com/jjshoots/PyFlyt.svg)](https://hits.dwyl.com/jjshoots/PyFlyt)
[![total downloads](https://static.pepy.tech/personalized-badge/pyflyt?period=total&units=international_system&left_color=grey&right_color=green&left_text=total%20downloads)](https://pepy.tech/project/pyflyt)
[![weekly downloads](https://static.pepy.tech/personalized-badge/pyflyt?period=week&units=international_system&left_color=grey&right_color=green&left_text=weekly%20downloads)](https://pepy.tech/project/pyflyt)

# PyFlyt - UAV Flight Simulator Gymnasium Environments for Reinforcement Learning Research

[View the documentation here!](https://jjshoots.github.io/PyFlyt/)

<p align="center">
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/pyflyt_cover_photo.png?raw=true" width="650px"/>
</p>

This is a library for testing reinforcement learning algorithms on UAVs.
This repo is still under development.
We are also actively looking for users and developers, if this sounds like you, don't hesitate to get in touch!

PyFlyt currently supports two separate UAV platforms:
- QuadX
  - Quadrotor UAV in the X configuration
  - Inspired by [the original pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones)
  - Actual full cascaded PID flight controller implementations for each drone.
  - Actual motor RPM simulation using first order differential equation.
  - For developers - 8 implemented flight modes that use tuned cascaded PID flight controllers, available in `PyFlyt/core/drones/quadx.py`.

- Fixedwing
  - Small tube-and-wing fixed wing UAV with a single motor (< 3 Kg)
  - Aerofoil characteristics derived from the paper: [*Real-time modeling of agile fixed-wing UAV aerodynamics*](https://ieeexplore.ieee.org/document/7152411)

- Rocket
  - 1:10th scale SpaceX Falcon 9 v1.2 first stage and interstage
  - Mass and geometry properties extracted from [Space Launch Report datasheet](https://web.archive.org/web/20170825204357/spacelaunchreport.com/falcon9ft.html#f9stglog)

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](#environments)
  - [`PyFlyt/QuadX-Hover-v0`](#pyflytquadx-hover-v0)
  - [`PyFlyt/QuadX-Waypoints-v0`](#pyflytquadx-waypoints-v0)
  - [`PyFlyt/Fixedwing-Waypoints-v0`](#pyflytfixedwing-waypoints-v0)
  - [`PyFlyt/Rocket-Landing-v0`](#pyflytrocket-landing-v0)


## Installation

```
pip3 install pyflyt
```

## Usage

Usage is similar to any other Gymnasium and (soon) PettingZoo environment:

```py
import gymnasium
import PyFlyt.gym_envs # noqa

env = gymnasium.make("PyFlyt/QuadX-Hover-v0", render_mode="human")
obs = env.reset()

termination = False
truncation = False

while not termination or truncation:
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
> `render_mode` can be either `"human"` or `rgb_array` or `None`.


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
> `render_mode` can be either `"human"` or `rgb_array` or `None`.


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
> `render_mode` can be either `"human"` or `rgb_array` or `None`.


<p align="center">
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/fixedwing_waypoint.gif?raw=true" width="500px"/>
</p>

### `PyFlyt/Rocket-Landing-v0`

An environment where the goal is to land a rocket on a landing pad at a speed of less than 1 m/s and comes to a halt successfully.
The 4 m tall rocket starts off with only 1% of fuel and is dropped from a height of 450 meters with a random linear and rotational velocity.
The environment ends when the rocket lands outside of the landing pad, or hits the landing pad at more than 1 m/s.

```py
env = gymnasium.make(
  "PyFlyt/Rocket-Landing-v0",
  sparse_reward: bool = False,
  ceiling: float = 500.0,
  max_displacement: float = 200.0,
  max_duration_seconds: float = 10.0,
  angle_representation: str = "quaternion",
  agent_hz: int = 40,
  render_mode: None | str = None,
  render_resolution: tuple[int, int] = (480, 480),
)
```

> `angle_representation` can be either `"quaternion"` or `"euler"`.
>
> `render_mode` can be either `"human"` or `rgb_array` or `None`.

<p align="center">
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/fixedwing_waypoint.gif?raw=true" width="500px"/>
</p>

## Citation

If you use our work in your research and would like to cite it, please use the following bibtex entry:

```
@software{pyflyt2023github,
  author = {Jun Jet Tai and Jim Wong},
  title = {PyFlyt - UAV Flight Simulator Gymnasium Environments for Reinforcement Learning Research},
  url = {http://github.com/jjshoots/PyFlyt},
  version = {1.0.0},
  year = {2023},
}
```
