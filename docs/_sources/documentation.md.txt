# Documentation

```{toctree}
:hidden:
documentation/core
documentation/gym_envs
```

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/pyflyt_cover_photo.png
    :width: 80%
```

## Installation

Installation on _Linux_ and _MacOS_ is simple:
```sh
pip3 install wheel numpy
pip3 install pyflyt
```
> `numpy` and `wheel` must be installed prior to `pyflyt` such that `pybullet` is built with `numpy` support.

On _Windows_, additional installation may be required for various Microsoft C++ components.

## Gymnasium Environments

If all you want are the Gymnasium environments, you can skip everything and go straight to the [Gymnasium Environments](documentation/gym_envs) section.

## I want to do more!

A set of helpful examples are provided in [the source repository](https://github.com/jjshoots/PyFlyt/tree/master/examples/core).
While this documentation provides a detailed overview of PyFlyt, the examples can bring you 80% of the way to a fully fledged user in a couple hours.
New users are highly encouraged to go through them first.
