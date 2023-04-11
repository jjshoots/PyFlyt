# Boosters

## Description

The `Boosters` component is used to represent an array of fueled boosters, each at arbitrary locations with different parameters.

Fueled boosters are propulsion units that produce no meaningful torque around their thrust axis and have limited throttleability.
More crucially, they depend on a fuel source that depletes with usage, changing the mass and inertia properties of the drone.
Additionally, some boosters, typically of the solid fuel variety, cannot be extinguished and reignited, a property we call reignitability.

## Usage

The `Boosters` component is parameterized with the following:

- `p` `(bullet_client.BulletClient)`: PyBullet physics client ID.
- `physics_period` `(float)`: physics period of the simulation.
- `np_random` `(np.random.RandomState)`: random state generator of the simulation.
- `uav_id` `(int)`: ID of the drone.
- `booster_ids` `(np.ndarray | list[int])`: list of integers representing the link index that each booster should be attached to.
- `fueltank_ids` `(np.ndarray | list[None | int])`: list of integers representing the link index for the fuel tank that each booster is attached to.
- `tau` `(np.ndarray)`: list of floats representing the ramp up time constant of each booster.
- `total_fuel_mass` `(np.ndarray)`: list of floats representing the fuel mass of each fuel tank at maximum fuel.
- `max_fuel_rate` `(np.ndarray)`: list of floats representing the maximum fuel burn rate for each booster.
- `max_inertia` `(np.ndarray)`: an `[n, 3]` array representing the diagonal elements of the moment of inertia matrix for each fuel tank at full fuel load.
- `min_thrust` `(np.ndarray)`: list of floats representing the thrust output for each booster at a minimum duty cycle that is NOT OFF.
- `max_thrust` `(np.ndarray)`: list of floats representing the maximum thrust output for each booster.
- `thrust_unit` `(np.ndarray)`: an `[n, 3]` array representing the unit vector pointing in the direction of force for each booster, relative to the booster link's body frame.
- `reignitable` `(np.ndarray | list[bool])`: a list of booleans representing whether the booster can be extinguished and then reignited.
- `noise_ratio` `(np.ndarray)`: a list of floats representing the percent amount of fluctuation present in each booster.

## Class Descriptions
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.Boosters
    :members:
```
