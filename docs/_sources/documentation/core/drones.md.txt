# Drones

```{toctree}
:hidden:
drones/fixedwing
drones/rocket
drones/quadx
```

## Overview

Several default UAV platforms are offered in PyFlyt.
These are:

- `quadx`
  - Quadrotor UAV in the X configuration.
  - Inspired by [the original pybullet drones by University of Toronto's Dynamic Systems Lab](https://github.com/utiasDSL/gym-pybullet-drones).
  - 8 implemented flight modes that use tuned cascaded PID flight controllers.
  - Two variants provided, use `drone_options=dict(drone_model="cf2x")` or `drone_options=dict(drone_model="primitive_drone")`.

- Fixedwing
  - Small tube-and-wing fixed wing UAV with a single motor (< 3 Kg).

- Rocket
  - 1:10th scale SpaceX Falcon 9 v1.2 first stage and interstage.
  - Mass and geometry properties extracted from [Space Launch Report datasheet](https://web.archive.org/web/20170825204357/spacelaunchreport.com/falcon9ft.html#f9stglog).