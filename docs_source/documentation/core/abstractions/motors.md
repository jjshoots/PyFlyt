# Motors

## Description

The `Motors` component is used to simulate a series of brushless electric motor driven propellers.
The do not have a fuel requirement, but generate a net torque around their thrust axis.

More concisely, the `Motors` component is similar to the `Boosters` component, except with more linear dynamics and the ability to have reversed thrust.
Similar to `Boosters`, the thrust generated depends on a ramp up time constant plus some noise:

1. The actual duty cycle depends on the `pwm` command and the ramp up time constant, `actual_duty_cycle += 1 / physics_hz / tau * (target_duty_cycle - actual_duty_cycle)`.

2. The RPM of the motor then depends on the actual duty cycle plus some noise, `rpm = (actual_duty_cycle * max_rpm) * (1 + noise * noise_ratio)`.
`noise` is sampled from a standard Normal.

3. Thrust and torque then depend on the RPM value and the propeller coefficients, `thrust = thrust_coef * rpm` and `torque = torque_coef * rpm`.

The motor additionally accepts a `rotation` matrix argument in `physics_update`.
This allows the thrust of the motor to be redirected.
Conveniently, the `Gimbals` component outputs this exact rotation matrix.

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.motors.Motors
    :members:
```
