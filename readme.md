# PyFlyt - Freestyle Quadcopter Flight in Pybullet with Gymnasium and (soon) PettingZoo APIs

This is a library for running reinforcement learning algorithms on both real crazyflies and simulated ones using the Gymnasium and (soon) PettingZoo APIs.
This repo's `master` branch is still under development.

Inspired by [the original pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones) with several key differences:

- Actual full cascaded PID flight controller implementations for each drone.
- Actual motor RPM simulation using first order differential equation.
- More modular control structure
- For developers - 8 implemented flight modes that use tuned cascaded PID flight controllers, available in `PyFlyt/core/drone.py`.
- For developers - easily build your own multiagent environments using the `PyFlyt.core.aviary.Aviary` class.
- More environments with increasing difficulties, targetted at enabling hiearchical learning for as true-to-realistic freestyle quadcopter flight.

## Table of Contents

- [Installation](#installation)
    - [On _macOS_ and _Ubuntu_](#on-macos-and-ubuntu)
- [Usage](#usage)
  - [Venv Activation](#venv-activation)
    - [On _macOS_ and _Ubuntu_](#on-macos-and-ubuntu-1)
  - [Usage](#usage-1)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
- [Environments](#environments)
  - [`PyFlyt/SimpleHoverEnv-v0`](#pyflytsimplehoverenv-v0)
  - [`PyFlyt/SimpleWaypointEnv-v0`](#pyflytsimplewaypointenv-v0)
- [Non-Gymnasium examples](#non-Gymnasium-examples)
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
pip3 install wheel
pip3 install pyflyt
```

## Usage

Usage is similar to any other Gymnasium and (soon) PettingZoo environment:

```py
import gymnasium
import PyFlyt

env = gymnasium.make("PyFlyt/SimpleHoverEnv-v0")

# ommit the below line to remove renderring and let
# the simulation go as fast as possible
env.render()
obs = env.reset()

done = False
while not done:
    observation, reward, done, _ = env.step(env.observation_space.sample())
```

### Observation Space

All observation spaces use `gymnasium.spaces.Box`.
For `Simple` environments, the observation spaces are simple 1D vectors of length less than 25, all values are not normalized.
For `Advanced` environments, the observation spaces use `gymnasium.spaces.Dict` with an `image` and `state` component.

### Action Space

All environments use the same action space, this is to allow hiearchical learning to take place - an RL agent can learn to hover before learning to move around.

By default, all environments have an action space that corresponds to FPV controls - pitch angular rate, roll angular rate, yaw_angular rate, thrust.
The limits for angular rate are +-3 rad/s.
The limits for thrust commands are -1 to 1.

The angular rates are intentionally not normalized to allow for a) better interpretation and b) more realistic inputs.

## Environments

### `PyFlyt/SimpleHoverEnv-v0`

A simple environment where an agent can learn to hover.
The environment ends when either the Quadcopter collides with the ground or exits the permitted flight dome.

### `PyFlyt/SimpleWaypointEnv-v0`

A simple environment where the goal is to position the Quadcopter at random setpoints in space within the permitted flight dome.
The environment ends when either the Quadcopter collides with the ground or exits the permitted flight dome.
<p align="center">
    <img src="/readme_assets/simple_waypoint.gif" width="500px"/>
</p>

**MORE ARE ON THE WAY**

## Non-Gymnasium examples

If you're not interested in RL but want to use the library for your own research, we provide a bunch of template scripts in `examples/` that you can run with `python3 examples/***.py` in _macOS_ and _Linux_.
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
