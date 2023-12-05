# Documentation

```{toctree}
:hidden:
documentation/core
documentation/gym_envs
documentation/pz_envs
```

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/pyflyt_cover_photo.png
    :width: 80%
```

## Installation

We recommend installations using Python [virtual environments](https://docs.python.org/3/library/venv.html).
It is possible to install PyFlyt using [`conda`](https://docs.conda.io/en/latest/), but YMMV.

### Linux and MacOS

Installation on _Linux_ and _MacOS_ is simple:
```sh
pip3 install wheel numpy
pip3 install pyflyt
```
> `numpy` and `wheel` must be installed prior to `pyflyt` such that `pybullet` is built with `numpy` support.

### Windows

1. First, install Microsoft Visual Studio Build Tools.
    - Go [here](https://visualstudio.microsoft.com/downloads/), scroll down to **All Downloads**, expand **Tools for Visual Studio**, download **Build Tools for Visual Studio 20XX**, where XX is just the latest year available.
    - Run the installer.
    - Select **Desktop development with C++**, then click **Install while downloading** or the alternate option if you wish.
2. Now, you can install `PyFlyt` as usual:
    ```sh
    pip3 install wheel numpy
    pip3 install pyflyt
    ```

## Gymnasium Environments

If all you want are the Gymnasium environments, you can skip everything and go straight to the [Gymnasium Environments](documentation/gym_envs) section.


## PettingZoo Environments

Similarly, if all you want are the PettingZoo environments, you can skip everything and go straight to the [PettingZoo Environments](documentation/pz_envs) section.

## I want to do more!

A set of helpful examples are provided in [the source repository](https://github.com/jjshoots/PyFlyt/tree/master/examples/core).
While this documentation provides a detailed overview of PyFlyt, the examples can bring you 80% of the way to a fully fledged user in a couple hours.
New users are highly encouraged to go through them first.
