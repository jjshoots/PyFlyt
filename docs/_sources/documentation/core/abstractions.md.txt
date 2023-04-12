# Abstractions

```{toctree}
:hidden:
abstractions/base_drone
abstractions/boosters
abstractions/boring_bodies
abstractions/camera
abstractions/gimbals
abstractions/lifting_surfaces
abstractions/motors
```

## Description

The `abstractions` submodule is, as the name suggests, a set of abstractions that allow easier construction of drones.
It implements various basic components, such as [motors](abstractions/motors) or [lifting surfaces](abstractions/lifting_surfaces) that allow construction of complicated configurations of drones without requiring reimplementation of all relevant mathematical models.

> **Custom Drone Example**
>
> While this section of the documentation serves as the Rosetta stone for all abstracted components, a very comprehensive tutorial on constructing your own drone can be found in the [tutorials section](../../tutorials).

## Usage

For all abstracted components, these methods are implemented and must be called within their designated methods for each drone class.
Some components may have special functions that need to be called elsewhere, refer to their relevant documentation for more information.

- `reset`: called on `drone.reset()`.
- `state_update`: called on `drone.update_state()`.
- (Optional) `get_states`: called on `drone.state_update()`, this returns the state of the component as a flattened `np.ndarray`.
- `physics_update`: called on `drone.update_physics()`.
