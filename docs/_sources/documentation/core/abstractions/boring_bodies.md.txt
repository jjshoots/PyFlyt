# Boring Bodies

## Description

The `BoringBodies` component represents a list of bodies falling through air, experiencing drag in all three axes as a primary result.

As aerodynamics of arbitrary bodies is difficult to simulate accurately, `BoringBodies` simulates them by assuming that the velocities experienced any body can be resolved into the three principled axes (X, Y, Z), which then cause individual drag forces on the body in the same three axes (X, Y, Z).
The following figure illustrates this idea in 2D.

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/boring_bodies.png
    :width: 70%
```

More concisely, the figure shows two reference frames --- a ground frame {math}`XZ_G` and a body frame {math}`XZ_B`.
The velocity {math}`\mathbf{v}` is resolved into body frame components {math}`v_x`, {math}`v_y` (not shown in figure) and {math}`v_z`.
The drag on the body itself is then computed using the familiar drag equation:
```{math}
F_i = - \text{sign}(v_i) \cdot \frac{1}{2} \cdot \rho \cdot A \cdot C_d \cdot v_i^2
```
where:
- {math}`i \in \{x, y, z\}`
- {math}`\rho` is the density of air, uniformly assumed to be 1.225 kg/m^3
- {math}`A` is the frontal area in the direction of {math}`i`
- {math}`C_d` is the drag coefficient in the direction of {math}`i`

## Class Description
```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.BoringBodies
    :members:
```
