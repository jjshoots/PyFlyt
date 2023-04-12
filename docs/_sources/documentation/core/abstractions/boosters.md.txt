# Boosters

## Description

The `Boosters` component is used to simulate a list of fueled boosters.
Fueled boosters are propulsion units that rely on the presence of available fuel from a fuel tank in order to generate thrust.
They generate no torque around the thrust axis.

The booster is parameterized with `booster_ids` and `fueltank_ids`, where the thrust acts on the links indicated by `booster_ids` and the mass properties of the fuel tanks are modified according to the indices specified in `fueltank_ids`.
Additionally, the boosters have a `reignitable` option, specifying whether the booster can be extinguished and then reignited or not --- a trait unique to liquid fuel boosters.

It takes several steps for the `ignition` and `pwm` command to be transformed into a thrust value:

1. If there is no fuel remaining, the `target_duty_cycle` is always 0.
Likewise, if the booster is not ignited, `target_duty_cycle` is also 0.
If the booster is lit and there is available fuel, `target_duty_cycle = min_thrust + (min_thrust / max_thrust) * pwm`.

2. The actual duty cycle of the rocket depends on the ramp up time constant, `actual_duty_cycle += 1 / physics_hz / tau * (target_duty_cycle - actual_duty_cycle)`.

3. The thrust of the rocket then depends on the actual duty cycle plus some noise, `thrust = (actual_duty_cycle * max_thrust) * (1 + noise * noise_ratio)`.
`noise` is sampled from a standard Normal.

Fuel burn is calculated proportional to the amount of thrust relative to maximum thrust, `fuel_burn = thrust / max_thrust * max_fuel_burn`.
The mass and inertia properties of the fuel tank are then proportional to the amount of fuel remaining.

The booster additionally accepts a `rotation` matrix argument in `physics_update`.
This allows the thrust of the booster to be redirected.
Conveniently, the `Gimbals` component outputs this exact rotation matrix.

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.Boosters
    :members:
```
