# Core

```{toctree}
:hidden:
core/aviary
core/abstractions
core/drones
```

Loosely put, PyFlyt has the following architecture:

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/pyflyt_architecture.png
    :width: 50%
```

At the core lies the [`aviary`](core/aviary), serving as a domain for all [`drones`](core/drones).
Each drone is defined by its own Python class, allowing for user-defined drones to be built.
The aviary also handles collision/contact tracking between entities and performs step scheduling according to the various looprates.
It also includes convenience functions for setting setpoints, arming status, and flight modes for all drones.


## Custom Drones

If you're interested in constructing your own drone, the best place to get started would be by going through the [examples section](https://github.com/jjshoots/PyFlyt/tree/master/examples/core).
First, read the [aviary documentation](./core/aviary.md), then the [custom drone example](https://github.com/jjshoots/PyFlyt/blob/master/examples/core/07_custom_uav.py), which demonstrates how to construct a custom drone using various pre-implemented components, listed in the [abstractions](core/abstractions) section.

Should you encounter any difficulty in doing so, don't hesitate to [get in touch](https://github.com/jjshoots/PyFlyt/issues)!

## Gymnasium Environments

If all you want are the gymnasium environments, you can skip everything and go straight to the [Gymnasium Environments](gym_envs) section.
