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
Each drone is defined by its own Python class, which inherits from a common
