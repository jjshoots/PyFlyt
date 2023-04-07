---
hide-toc: true
firstpage:
lastpage:
---

# PyFlyt - UAV Flight Simulator Gymnasium Environments for Reinforcement Learning Research

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/pyflyt_cover_photo.png
    :width: 80%
    :name: pyflyt_cover_photo
```

> This repo is still under development. We are also actively looking for users and developers, if this sounds like you, don't hesitate to get in touch!

## Installation

```sh
pip3 install PyFlyt
```

```{toctree}
:hidden:
:caption: Introduction

introduction/intro
```

```{toctree}
:hidden:
:caption: Core

core/aviary
```

```{toctree}
:hidden:
:caption: Core.Drones

core/drones/fixedwing.md
core/drones/rocket.md
core/drones/quadx.md
```

```{toctree}
:hidden:
:caption: Core.Abstractions

core/abstractions/base_drone.md
core/abstractions/boring_bodies.md
core/abstractions/boosters.md
core/abstractions/camera.md
core/abstractions/gimbals.md
core/abstractions/lifting_surfaces.md
core/abstractions/motors.md
```

```{toctree}
:hidden:
:caption: Gymnasium Environments

gym_envs/quadx
gym_envs/fixedwing
gym_envs/rocket
```
