# Core

Loosely put, PyFlyt has the following architecture:

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/pyflyt_architecture.png
    :width: 50%
```

At the core lies the [`aviary`](core/aviary), serving as a domain for all [`drones`](core/drones).
Each drone is defined by its own Python class, allowing for user-defined drones to be built.
Drones can be built from the ground up, or can be an amalgamation of various component [`abstractions`](core/abstractions).

```{toctree}
:hidden:
core/aviary
core/abstractions
core/drones
```
