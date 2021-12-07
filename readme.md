# PyBullet Swarming

This is a library for running swarming algorithms on both real crazyflies and simulated ones.
This repo's `master` branch is still under development.

Inspired by [the original pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones). But more modular, modifiable, and incorporates a companion module for interfacing with a real Crazyflie drone swarm using the Crazyradio PA module. Built for AVAILab, Coventry University.

## Installation
Code is written and tested using _Python 3.8_ with `venv` and tested on Windows 10 and Ubuntu 20.04 LTS.

#### On _macOS_ and _Ubuntu_
```
git clone https://github.com/jjshoots/pybullet_swarming.git
cd pybullet_swarming
python3 -m venv venv
source venv/bin/activate
pip3 install -e .
```

#### On _Windows_
Follow [this](https://deepakjogi.medium.com/how-to-install-pybullet-physics-simulation-in-windows-e1f16baa26f6) guide, then:
```
git clone https://github.com/jjshoots/pybullet_swarming.git
cd pybullet_swarming
python -m venv venv
./venv/bin/activate.bash
pip install -e .
```

## Usage
The installation above installs the packages required in a [python virtual environment](https://docs.python.org/3/library/venv.html). So, to activate the environment, you need to do:

#### On _macOS_ and _Ubuntu_
```
source venv/bin/activate
```

#### On _macOS_ and _Ubuntu_
```
./venv/bin/activate.bash
```

Deactivation is the same except replace `activate` with `deactivate`.


## Examples
There are multiple template scripts available in `examples/` that you can run with `python3 examples/***.py`.

#### `simulate_single.py`
Simulates a single drone in the pybullet env with position control.
![simulate a single drone](/resource/simulate_single.gif)

#### `simulate_swarm.py`
Simulates a swarm of drones in the pybullet env with velocity control.
![simulate a swarm of drones](/resource/simulate_swarm.gif)

#### `simulate_cube.py`
Simulates a swarm of drones in a spinning cube.
![You spin me round right round](/resource/simulate_cube.gif)

#### `single_crazyflie.py`
Flies a real Crazyflie, check out the [documentation](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/) and [how to connect](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/#config-client) to get your URI(s) and modify them in line 18.

#### `swarm_crazyflie.py`
Flies a real Crazyflie swarm, same as the previous example, but now takes in a list of URIs.
