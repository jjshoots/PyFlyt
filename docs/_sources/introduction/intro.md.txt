# Introduction

Loosely put, PyFlyt has the following architecture:

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/pyflyt_architecture.png
    :width: 50%
    :name: pyflyt_architecture
```

At the core lies the `aviary`, serving as a domain for all `drones`.
It handles collision/contact tracking between entities and performs step scheduling according to the various looprates.
It also includes convenience functions for setting setpoints, arming status, and flight modes for all drones.

The aviary can accommodate any number of `drones`.
Each drone is defined by its own Python class, allowing for user-defined drones to be built.


## Custom Drones

The best place to get started would be by going through the [examples section](https://github.com/jjshoots/PyFlyt/tree/master/examples/core).
If you're interested in building custom drones, read the [aviary documentation](../core/aviary.md) first, then the [custom drone example](https://github.com/jjshoots/PyFlyt/blob/master/examples/core/07_custom_uav.py).
PyFlyt natively has several components pre-implemented, visit the Abstractions section.

## Gymnasium Environments

If all you want are the gymnasium environments, you can skip everything and go straight to the Gymnasium Environments section.
