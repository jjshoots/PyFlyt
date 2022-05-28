# PyFlyt - Freestyle Quadcopter Flight in Pybullet with Gym and (soon) PettingZoo APIs

This is a library for running reinforcement learning algorithms on both real crazyflies and simulated ones using the Gym and (soon) PettingZoo APIs.
This repo's `master` branch is still under development.

Inspired by [the original pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones) with several key differences:

- Actual full cascaded PID flight controller implementations for each drone.
- Actual motor RPM simulation using first order differential equation.
- More modular control structure
- For developers - 8 implemented flight modes that use tuned cascaded PID flight controllers, available in `PyFlyt/core/drone.py`.
- For developers - easily build your own multiagent environments using the `PyFlyt.core.aviary.Aviary` class.
- More environments with increasing difficulties, targetted at enabling hiearchical learning for as true-to-realistic freestyle quadcopter flight.

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

## Installation
Code is written and tested using _Python 3.8_ with `venv` and tested on Ubuntu 20.04 LTS.

#### On _macOS_ and _Ubuntu_
This is the recommended way to install things, using a venv.
```
git clone https://github.com/jjshoots/PyFlyt.git
cd PyFlyt
python3 -m venv venv
source venv/bin/activate
pip3 install -e .
```

## Usage

### Venv Activation

The installation above installs the packages required in a [python virtual environment](https://docs.python.org/3/library/venv.html).
So, to activate the environment, you need to do:

#### On _macOS_ and _Ubuntu_
```
source venv/bin/activate
```

Deactivation is the same except replace `activate` with `deactivate`.

### Usage

Usage is similar to any other Gym and (soon) PettingZoo environment:

```py
import PyFlyt

env = gym.make("PyFlyt/SimpleHoverEnv-v0")


env.render()
obs = env.reset()

done = False
while not done:
    observation, reward, done, _ = env.step(env.observation_space.sample())
```

### Observation Space

All observation spaces use `gym.spaces.Box`.
For `Simple` environments, the observation spaces are simple 1D vectors of length less than 25, all values are not normalized.
For `Advanced` environments, the observation spaces are usually images, and sometimes 1D vectors.

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
