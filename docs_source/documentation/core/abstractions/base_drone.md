# Base Drone Class

> **Custom Drone Example**
>
> While this section of the documentation serves as the Rosetta stone for all abstracted components, a very comprehensive tutorial on constructing your own drone can be found in the [tutorials section](../../../tutorials).

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.DroneClass
```

### Default Attributes
```{eval-rst}
.. property:: PyFlyt.core.abstractions.DroneClass.np_random

  **dtype** - `np.random.RandomState`

.. property:: PyFlyt.core.abstractions.DroneClass.physics_period

  **dtype** - `float`

.. property:: PyFlyt.core.abstractions.DroneClass.control_period

  **dtype** - `float`

.. property:: PyFlyt.core.abstractions.DroneClass.drone_path

  Path to the drone's URDF file.

  **dtype** - `str`

.. property:: PyFlyt.core.abstractions.DroneClass.param_path

  Path to the drone's YAML parameters file.

  **dtype** - `str`

.. property:: PyFlyt.core.abstractions.DroneClass.Id

  PyBullet ID of the drone itself.

  **dtype** - `int`
```

### Required Attributes
```{eval-rst}
.. property:: PyFlyt.core.abstractions.DroneClass.state

  **dtype** - `np.ndarray`

.. property:: PyFlyt.core.abstractions.DroneClass.aux_state

  **dtype** - `np.ndarray`

.. property:: PyFlyt.core.abstractions.DroneClass.setpoint

  **dtype** - `np.ndarray`
```

### Optional Attributes
```{eval-rst}
.. property:: PyFlyt.core.abstractions.DroneClass.rgbaImg

  **dtype** - `np.ndarray`

.. property:: PyFlyt.core.abstractions.DroneClass.depthImg

  **dtype** - `np.ndarray`

.. property:: PyFlyt.core.abstractions.DroneClass.segImg

  **dtype** - `np.ndarray`

.. property:: PyFlyt.core.abstractions.DroneClass.registered_controllers

  **dtype** - `dict[int, type[ControlClass]]`

.. property:: PyFlyt.core.abstractions.DroneClass.instanced_controllers

  **dtype** - `dict[int, ControlClass]`

.. property:: PyFlyt.core.abstractions.DroneClass.registered_base_controllers

  **dtype** - `dict[int, int]`
```

### Default Methods
```{eval-rst}
.. autofunction:: PyFlyt.core.abstractions.DroneClass.set_mode
.. autofunction:: PyFlyt.core.abstractions.DroneClass.get_joint_info
.. autofunction:: PyFlyt.core.abstractions.DroneClass.disable_artificial_damping
```

### Required Methods
```{eval-rst}
.. autofunction:: PyFlyt.core.abstractions.DroneClass.reset
.. autofunction:: PyFlyt.core.abstractions.DroneClass.update_control
.. autofunction:: PyFlyt.core.abstractions.DroneClass.update_physics
.. autofunction:: PyFlyt.core.abstractions.DroneClass.update_state
.. autofunction:: PyFlyt.core.abstractions.DroneClass.update_last
```

### Optional Methods
```{eval-rst}
.. autofunction:: PyFlyt.core.abstractions.DroneClass.register_controller
```
