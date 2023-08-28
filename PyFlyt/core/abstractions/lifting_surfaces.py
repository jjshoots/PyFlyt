"""A component to simulate lifting surfaces vehicle."""
from __future__ import annotations

import warnings

import numpy as np
from pybullet_utils import bullet_client

from PyFlyt.core.utils.compile_helpers import jitter


class LiftingSurfaces:
    """Handler for multiple lifting surfaces.

    This is a convenience class for handling multiple lifting surfaces as a single object.
    Simply pass it a list of `LiftingSurface` objects.

    Args:
        lifting_surfaces (list[LiftingSurface]): a list of `LiftingSurface` objects.
    """

    def __init__(self, lifting_surfaces: list[LiftingSurface]):
        """__init__.

        Args:
            lifting_surfaces (list[LiftingSurface]): a list of `LiftingSurface` objects.
        """
        # assert all is lifting surfaces
        assert all(
            [isinstance(surface, LiftingSurface) for surface in lifting_surfaces]
        )

        # store some stuff
        self.p = lifting_surfaces[0].p
        self.uav_id = lifting_surfaces[0].uav_id
        self.surfaces: list[LiftingSurface] = lifting_surfaces
        self.surface_ids = np.array([s.surface_id for s in self.surfaces])

    def reset(self):
        """Resets all lifting surfaces."""
        [surface.reset() for surface in self.surfaces]

    def get_states(self) -> np.ndarray:
        """Gets the current state of the components.

        Returns:
            np.ndarray: a (num_surfaces, ) array representing the actuation state for each surface
        """
        return np.array([surface.actuation for surface in self.surfaces])

    def physics_update(self, cmd: np.ndarray):
        """Converts actuation commands into forces on the lifting surfaces.

        Args:
            cmd (np.ndarray): the full command array, command mapping is handled through `command_id` and `command_sign` on each surface, normalized in [-1, 1].
        """
        assert len(cmd.shape) == 1, f"`{cmd=}` must be 1D array."
        assert cmd.shape[0] == len(
            self.surfaces
        ), f"`{cmd=}` must have same number of elements as surfaces ({len(self.surfaces)})."
        assert np.all(cmd >= -1.0) and np.all(
            cmd <= 1.0
        ), f"`{cmd=} has values out of bounds of -1.0 and 1.0.`"

        for surface, actuation in zip(self.surfaces, cmd):
            surface.physics_update(actuation)

    def state_update(self, rotation_matrix: np.ndarray):
        """Updates all local surface velocities of the lifting surface, place under `update_state`.

        Args:
            rotation_matrix (np.ndarray): (3, 3) OR (num_surfaces, 3, 3) array rotation_matrix
        """
        # get all the states for all the surfaces
        link_states = self.p.getLinkStates(
            self.uav_id, self.surface_ids, computeLinkVelocity=True
        )

        # get all the velocities
        surface_velocities = np.array([item[-2] for item in link_states])

        # query for wind if available and add to surface velocities
        if self.p.wind_field is not None:
            surface_positions = np.array([item[0] for item in link_states])
            surface_velocities -= self.p.wind_field(
                self.p.elapsed_time, surface_positions
            )

        # convert all to local velocities, depending on rotation matrix style
        if rotation_matrix.shape == (len(self.surfaces), 3, 3):
            surface_velocities = np.matmul(
                rotation_matrix, np.expand_dims(surface_velocities, -1)
            ).squeeze(-1)
        elif rotation_matrix.shape == (3, 3):
            surface_velocities = np.matmul(rotation_matrix, surface_velocities.T).T
        else:
            raise ValueError(
                f"Only accept (num_surfaces, 3, 3) or (3, 3) array for `rotation_matrix`, got {rotation_matrix.shape}."
            )

        # update the velocities of all surfaces
        surface_velocities = np.ascontiguousarray(surface_velocities)
        for surface, velocity in zip(self.surfaces, surface_velocities):
            surface.state_update(velocity)


class LiftingSurface:
    """Used to represent a single lifting surface.

    The `Lifting Surface` component is used to simulate a single lifting surface based on "Real-time modeling of agile fixed-wing uav aerodynamics, Khan et. al.".

    Args:
        p (bullet_client.BulletClient): PyBullet physics client ID.
        physics_period (float): physics period of the simulation.
        np_random (np.random.RandomState): random number generator of the simulation.
        uav_id (int): ID of the drone.
        surface_id (int): an integer for the link ID for this lifting surface.
        lifting_unit (np.ndarray): (3,) unit vector representing the direction of lift.
        forward_unit (np.ndarray): (3,) unit vector representing the direction of travel.
        Cl_alpha_2D (float): lift coefficient slope under a no-stall condition.
        chord (float): chord of the lifting surface.
        span (float): span of the lifting surface.
        flap_to_chord (float): ratio of the wing that is an actuated flap, can be in [0, 1].
        eta (float): correction factor for viscosity effects, usually 0.65.
        alpha_0_base (float): zero lift angle-of-attack.
        alpha_stall_P_base (float): positive stall angle in degrees.
        alpha_stall_N_base (float): negative stall angle in degrees.
        Cd_0 (float): drag coefficient at zero angle-of-attack.
        deflection_limit (float): maximum deflection limit of the actuated flap in degrees.
        tau (float): actuation ramp time constant.
    """

    def __init__(
        self,
        p: bullet_client.BulletClient,
        physics_period: float,
        np_random: np.random.RandomState,
        uav_id: int,
        surface_id: int,
        lifting_unit: np.ndarray,
        forward_unit: np.ndarray,
        Cl_alpha_2D: float,
        chord: float,
        span: float,
        flap_to_chord: float,
        eta: float,
        alpha_0_base: float,
        alpha_stall_P_base: float,
        alpha_stall_N_base: float,
        Cd_0: float,
        deflection_limit: float,
        tau: float,
    ):
        """Used for simulating a single lifting surface.

        Args:
            p (bullet_client.BulletClient): PyBullet physics client ID.
            physics_period (float): physics period of the simulation.
            np_random (np.random.RandomState): random number generator of the simulation.
            uav_id (int): ID of the drone.
            surface_id (int): an integer for the link ID for this lifting surface.
            lifting_unit (np.ndarray): (3,) unit vector representing the direction of lift.
            forward_unit (np.ndarray): (3,) unit vector representing the direction of travel.
            Cl_alpha_2D (float): lift coefficient slope under a no-stall condition.
            chord (float): chord of the lifting surface.
            span (float): span of the lifting surface.
            flap_to_chord (float): ratio of the wing that is an actuated flap, can be in [0, 1].
            eta (float): correction factor for viscosity effects, usually 0.65.
            alpha_0_base (float): zero lift angle-of-attack.
            alpha_stall_P_base (float): positive stall angle in degrees.
            alpha_stall_N_base (float): negative stall angle in degrees.
            Cd_0 (float): drag coefficient at zero angle-of-attack.
            deflection_limit (float): maximum deflection limit of the actuated flap in degrees.
            tau (float): actuation ramp time constant.
        """
        self.p = p
        self.physics_period = physics_period
        self.np_random = np_random

        # store IDs
        self.uav_id = uav_id
        self.surface_id = surface_id

        # some checks for the lifting and norm vectors
        assert lifting_unit.shape == (3,)
        assert forward_unit.shape == (3,)
        assert (
            tau >= 0.0 / physics_period
        ), f"Setting `tau = 1 / physics_period` is equivalent to 0, 0 is not a valid option, got {tau}."
        if np.linalg.norm(lifting_unit) != 1.0:
            warnings.warn(f"Norm of `{lifting_unit=}` is not 1.0, normalizing...")
            lifting_unit /= np.linalg.norm(lifting_unit)
        if np.linalg.norm(forward_unit) != 1.0:
            warnings.warn(f"Norm of `{forward_unit=}` is not 1.0, normalizing...")
            forward_unit /= np.linalg.norm(forward_unit)
        if np.dot(lifting_unit, forward_unit) != 0.0:
            warnings.warn(
                f"`{forward_unit}` and `{lifting_unit}` are not orthogonal, you have been warned..."
            )

        # get the lift, drag, and torque units
        self.lift_unit = lifting_unit
        self.drag_unit = forward_unit
        self.torque_unit = np.cross(lifting_unit, forward_unit)

        # wing parameters
        self.Cl_alpha_2D = Cl_alpha_2D
        self.chord = chord
        self.span = span
        self.flap_to_chord = flap_to_chord
        self.eta = eta
        self.alpha_0_base = alpha_0_base
        self.alpha_stall_P_base = alpha_stall_P_base
        self.alpha_stall_N_base = alpha_stall_N_base
        self.Cd_0 = Cd_0
        self.deflection_limit = deflection_limit
        self.cmd_tau = tau

        # precompute some constants
        self.half_rho = 0.5 * 1.225
        self.area = self.chord * self.span
        self.aspect = self.span / self.chord
        self.alpha_stall_P_base = np.deg2rad(self.alpha_stall_P_base)
        self.alpha_stall_N_base = np.deg2rad(self.alpha_stall_N_base)
        self.alpha_0_base = np.deg2rad(self.alpha_0_base)
        self.Cl_alpha_3D = self.Cl_alpha_2D * (
            self.aspect
            / (self.aspect + ((2.0 * (self.aspect + 4.0)) / (self.aspect + 2.0)))
        )
        self.theta_f = np.arccos(2.0 * self.flap_to_chord - 1.0)
        self.aero_tau = 1 - ((self.theta_f - np.sin(self.theta_f)) / np.pi)

        # runtime parameters
        self.local_surface_velocity = np.array([0.0, 0.0, 0.0])

    def reset(self):
        """Reset the lifting surfaces."""
        self.actuation = 0.0

    def get_states(self) -> float:
        """Gets the current state of the components.

        Returns:
            float: the level of deflection of the surface.
        """
        return self.actuation

    def state_update(self, surface_velocity: np.ndarray):
        """Updates the local surface velocity of the lifting surface.

        Args:
            surface_velocity (np.ndarray): surface_velocity.
        """
        self.local_surface_velocity = surface_velocity

    def physics_update(self, cmd: float):
        """Converts a commanded actuation state into forces on the lifting surface.

        Args:
            cmd (float): normalized actuation in [-1, 1].

        Returns:
            tuple[np.ndarray, np.ndarray]: vec3 force, vec3 torque
        """
        # model the deflection using first order ODE, y' = T/tau * (setpoint - y)
        self.actuation += (self.physics_period / self.cmd_tau) * (cmd - self.actuation)

        # compute angle of attack and freestream velocity
        (alpha, freestream_speed) = self._compute_aoa_freestream(
            self.local_surface_velocity, self.lift_unit, self.drag_unit
        )

        # compute aerofoil parameters
        (Cl, Cd, CM) = self._jitted_compute_aero_data(
            alpha,
            self.aspect,
            self.flap_to_chord,
            self.aero_tau,
            self.actuation,
            self.deflection_limit,
            self.eta,
            self.Cl_alpha_3D,
            self.alpha_stall_P_base,
            self.alpha_0_base,
            self.alpha_stall_N_base,
            self.Cd_0,
        )

        # compute the forces and torques
        (force, torque) = self._jitted_compute_force_torque(
            alpha,
            freestream_speed,
            Cl,
            Cd,
            CM,
            self.half_rho,
            self.area,
            self.chord,
            self.lift_unit,
            self.drag_unit,
            self.torque_unit,
        )

        self.p.applyExternalForce(
            self.uav_id,
            self.surface_id,
            force,
            [0.0, 0.0, 0.0],
            self.p.LINK_FRAME,
        )
        self.p.applyExternalTorque(
            self.uav_id, self.surface_id, torque, self.p.LINK_FRAME
        )

    @staticmethod
    @jitter
    def _compute_aoa_freestream(
        local_surface_velocity: np.ndarray, lift_unit: np.ndarray, drag_unit: np.ndarray
    ) -> tuple[float, float]:
        """Computes the angle of attack (alpha) as well as the freestream speed.

        Args:
            local_surface_velocity (np.ndarray): local_surface_velocity from self
            lift_unit (np.ndarray): lift_unit from self
            drag_unit (np.ndarray): drag_unit from self

        Returns:
            tuple[float, float]:
        """
        freestream_speed = np.linalg.norm(local_surface_velocity).item()
        lifting_airspeed = np.dot(local_surface_velocity, lift_unit)
        forward_airspeed = np.dot(local_surface_velocity, drag_unit)
        alpha = np.arctan2(-lifting_airspeed, forward_airspeed)

        return alpha, freestream_speed

    @staticmethod
    @jitter
    def _jitted_compute_aero_data(
        alpha: float,
        aspect: float,
        flap_to_chord: float,
        aero_tau: float,
        actuation: float,
        deflection_limit: float,
        eta: float,
        Cl_alpha_3D: float,
        alpha_stall_P_base: float,
        alpha_0_base: float,
        alpha_stall_N_base: float,
        Cd_0: float,
    ) -> tuple[float, float, float]:
        """Computes the relevant aerodynamic data depending on the current state of the lifting surface.

        Args:
            alpha (float): alpha
            aspect (float): aspect from self
            flap_to_chord (float): flap_to_chord from self
            aero_tau (float): aero_tau from self
            actuation (float): actuation from self
            deflection_limit (float): deflection_limit from self
            eta (float): eta from self
            Cl_alpha_3D (float): Cl_alpha_3D from self
            alpha_stall_P_base (float): alpha_stall_P_base from self
            alpha_0_base (float): alpha_0_base from self
            alpha_stall_N_base (float): alpha_stall_N_base from self
            Cd_0 (float): Cd_0 from self

        Returns:
            tuple[float, float, float]:
        """
        # deflection must be in degrees because engineering uses degrees
        deflection_radians = np.deg2rad(actuation * deflection_limit)

        delta_Cl = Cl_alpha_3D * aero_tau * eta * deflection_radians
        delta_Cl_max = flap_to_chord * delta_Cl
        Cl_max_P = Cl_alpha_3D * (alpha_stall_P_base - alpha_0_base) + delta_Cl_max
        Cl_max_N = Cl_alpha_3D * (alpha_stall_N_base - alpha_0_base) + delta_Cl_max
        alpha_0 = alpha_0_base - (delta_Cl / Cl_alpha_3D)
        alpha_stall_P = alpha_0 + (Cl_max_P / Cl_alpha_3D)
        alpha_stall_N = alpha_0 + (Cl_max_N / Cl_alpha_3D)

        # no stall condition
        if alpha_stall_N < alpha and alpha < alpha_stall_P:
            Cl = Cl_alpha_3D * (alpha - alpha_0)
            alpha_i = Cl / (np.pi * aspect)
            alpha_eff = alpha - alpha_0 - alpha_i
            CT = Cd_0 * np.cos(alpha_eff)
            CN = (Cl + (CT * np.sin(alpha_eff))) / np.cos(alpha_eff)
            Cd = (CN * np.sin(alpha_eff)) + (CT * np.cos(alpha_eff))
            CM = -CN * (0.25 - (0.175 * (1.0 - ((2.0 * alpha_eff) / np.pi))))

            return Cl, Cd, CM

        # positive stall
        if alpha > 0.0:
            # Stall calculations to find alpha_i at stall
            Cl_stall = Cl_alpha_3D * (alpha_stall_P - alpha_0)
            alpha_i_at_stall = Cl_stall / (np.pi * aspect)
            # alpha_i post-stall Pos
            alpha_i = np.interp(
                alpha, [alpha_stall_P, np.pi / 2.0], [alpha_i_at_stall, 0.0]
            )
        # negative stall
        else:
            # Stall calculations to find alpha_i at stall
            Cl_stall = Cl_alpha_3D * (alpha_stall_N - alpha_0)
            alpha_i_at_stall = Cl_stall / (np.pi * aspect)
            # alpha_i post-stall Neg
            alpha_i = np.interp(
                alpha, [-np.pi / 2.0, alpha_stall_N], [0.0, alpha_i_at_stall]
            )

        alpha_eff = alpha - alpha_0 - alpha_i

        # Drag coefficient at 90 deg dependent on deflection angle
        Cd_90 = (
            ((-4.26 * (10**-2)) * (deflection_radians**2))
            + ((2.1 * (10**-1)) * deflection_radians)
            + 1.98
        )
        CN = (
            Cd_90
            * np.sin(alpha_eff)
            * (
                1.0 / (0.56 + 0.44 * abs(np.sin(alpha_eff)))
                - 0.41 * (1.0 - np.exp(-17.0 / aspect))
            )
        )
        CT = 0.5 * Cd_0 * np.cos(alpha_eff)
        Cl = (CN * np.cos(alpha_eff)) - (CT * np.sin(alpha_eff))
        Cd = (CN * np.sin(alpha_eff)) + (CT * np.cos(alpha_eff))
        CM = -CN * (0.25 - (0.175 * (1.0 - ((2.0 * abs(alpha_eff)) / np.pi))))

        return Cl, Cd, CM

    @staticmethod
    @jitter
    def _jitted_compute_force_torque(
        alpha: float,
        freestream_speed: float,
        Cl: float,
        Cd: float,
        CM: float,
        half_rho: float,
        area: float,
        chord: float,
        lift_unit: np.ndarray,
        drag_unit: np.ndarray,
        torque_unit: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the force and torque vectors on the surface.

        Args:
            alpha (float): alpha
            freestream_speed (float): freestream_speed
            Cl (float): Cl
            Cd (float): Cd
            CM (float): CM
            half_rho (float): half_rho from self
            area (float): area from self
            chord (float): chord from self
            lift_unit (np.ndarray): lift_unit from self
            drag_unit (np.ndarray): drag_unit from self
            torque_unit (np.ndarray): torque_unit from self

        Returns:
            tuple[np.ndarray, np.ndarray]:
        """
        # compute dynamic pressure
        Q = half_rho * np.square(freestream_speed)
        Q_area = Q * area

        # compute lift and drag
        lift = Cl * Q_area
        drag = Cd * Q_area
        force_normal = (lift * np.cos(alpha)) + (drag * np.sin(alpha))
        force_parallel = (lift * np.sin(alpha)) - (drag * np.cos(alpha))

        # compute forces and torques
        force = lift_unit * force_normal + drag_unit * force_parallel
        torque = Q_area * CM * chord * torque_unit

        return force, torque
