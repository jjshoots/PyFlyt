from __future__ import annotations

import numpy as np


class LiftingSurface:
    def __init__(
        self,
        id: int,
        command_id: None | int,
        command_sign: float,
        z_axis_lift: bool,
        aerofoil_params: dict,
    ):
        self.id = id
        self.command_id = command_id
        self.command_sign = command_sign
        self.z_axis_lift = z_axis_lift

        if z_axis_lift:
            self.lift_axis = np.array([0.0, 0.0, 1.0])
            self.drag_axis = np.array([0.0, 1.0, 0.0])
            self.torque_axis = np.array([1.0, 0.0, 0.0])
        else:
            self.lift_axis = np.array([1.0, 0.0, 0.0])
            self.drag_axis = np.array([0.0, 1.0, 0.0])
            self.torque_axis = np.array([0.0, 0.0, 1.0])

        # wing parameters
        self.Cl_alpha_2D = aerofoil_params["Cl_alpha_2D"]
        self.chord = aerofoil_params["chord"]
        self.span = aerofoil_params["span"]
        self.flap_to_chord = aerofoil_params["flap_to_chord"]
        self.eta = aerofoil_params["eta"]
        self.alpha_0_base = aerofoil_params["alpha_0_base"]
        self.alpha_stall_P_base = aerofoil_params["alpha_stall_P_base"]
        self.alpha_stall_N_base = aerofoil_params["alpha_stall_N_base"]
        self.Cd_0 = aerofoil_params["Cd_0"]
        self.deflection_limit = aerofoil_params["deflection_limit"]

        # precompute some constants
        self.half_rho = 0.5 * 1.225
        self.area = self.chord * self.span
        self.aspect = self.span / self.chord
        self.alpha_stall_P_base = np.deg2rad(self.alpha_stall_P_base)
        self.alpha_stall_N_base = np.deg2rad(self.alpha_stall_N_base)
        self.alpha_0_base = np.deg2rad(self.alpha_0_base)
        self.Cl_alpha_3D = self.Cl_alpha_2D * (
            self.aspect / (self.aspect + ((2 * (self.aspect + 4)) / (self.aspect + 2)))
        )
        self.theta_f = np.arccos(2 * self.flap_to_chord - 1)
        self.tau = 1 - ((self.theta_f - np.sin(self.theta_f)) / np.pi)

        # runtime parameters
        self.local_surface_velocity = np.array([0.0, 0.0, 0.0])

    def update_local_surface_velocity(
        self, rotation_matrix: np.ndarray, surface_velocity: np.ndarray
    ):
        self.local_surface_velocity = np.matmul(rotation_matrix, surface_velocity)

    def compute_force_torque(
        self,
        actuation: float,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.z_axis_lift:
            alpha = np.arctan2(
                -self.local_surface_velocity[2], self.local_surface_velocity[1]
            )
            freestream_speed = np.linalg.norm(
                [self.local_surface_velocity[1], self.local_surface_velocity[2]]
            )
        else:
            alpha = np.arctan2(
                -self.local_surface_velocity[0], self.local_surface_velocity[1]
            )
            freestream_speed = np.linalg.norm(
                [self.local_surface_velocity[0], self.local_surface_velocity[1]]
            )

        deflection = self.deflection_limit * actuation
        [Cl, Cd, CM] = self._compute_aero_data(deflection, alpha)

        Q = self.half_rho * np.square(freestream_speed)  # Dynamic pressure
        Q_area = Q * self.area

        lift = Cl * Q_area
        drag = Cd * Q_area
        force_normal = (lift * np.cos(alpha)) + (drag * np.sin(alpha))
        force_parallel = (lift * np.sin(alpha)) - (drag * np.cos(alpha))

        force = self.lift_axis * force_normal + self.drag_axis * force_parallel
        torque = Q_area * CM * self.chord * self.torque_axis

        return force, torque

    def _compute_aero_data(
        self, deflection: float, alpha: float
    ) -> tuple[float, float, float]:
        """Returns Cl, Cd, and CM for a given aerofoil, control surface deflection, and alpha"""

        # deflection must be in degrees because engineering uses degrees
        deflection = np.deg2rad(deflection)

        delta_Cl = self.Cl_alpha_3D * self.tau * self.eta * deflection
        delta_Cl_max = self.flap_to_chord * delta_Cl
        Cl_max_P = (
            self.Cl_alpha_3D * (self.alpha_stall_P_base - self.alpha_0_base)
            + delta_Cl_max
        )
        Cl_max_N = (
            self.Cl_alpha_3D * (self.alpha_stall_N_base - self.alpha_0_base)
            + delta_Cl_max
        )
        alpha_0 = self.alpha_0_base - (delta_Cl / self.Cl_alpha_3D)
        alpha_stall_P = alpha_0 + (Cl_max_P / self.Cl_alpha_3D)
        alpha_stall_N = alpha_0 + (Cl_max_N / self.Cl_alpha_3D)

        # Check if stalled
        if (alpha >= alpha_stall_P) or (alpha <= alpha_stall_N):
            if alpha >= alpha_stall_P:
                # Stall calculations to find alpha_i at stall
                Cl_stall = self.Cl_alpha_3D * (alpha_stall_P - alpha_0)
                alpha_i_at_stall = Cl_stall / (np.pi * self.aspect)
                # alpha_i post-stall Pos
                alpha_i = np.interp(
                    alpha, [alpha_stall_P, np.pi / 2], [alpha_i_at_stall, 0]
                )

            elif alpha <= alpha_stall_N:
                # Stall calculations to find alpha_i at stall
                Cl_stall = self.Cl_alpha_3D * (alpha_stall_N - alpha_0)
                alpha_i_at_stall = Cl_stall / (np.pi * self.aspect)
                # alpha_i post-stall Neg
                alpha_i = np.interp(
                    alpha, [-np.pi / 2, alpha_stall_N], [0, alpha_i_at_stall]
                )

            alpha_eff = alpha - alpha_0 - alpha_i
            # Drag coefficient at 90 deg dependent on deflection angle
            Cd_90 = (
                ((-4.26 * (10**-2)) * (deflection**2))
                + ((2.1 * (10**-1)) * deflection)
                + 1.98
            )
            CN = (
                Cd_90
                * np.sin(alpha_eff)
                * (
                    1 / (0.56 + 0.44 * abs(np.sin(alpha_eff)))
                    - 0.41 * (1 - np.exp(-17 / self.aspect))
                )
            )
            CT = 0.5 * self.Cd_0 * np.cos(alpha_eff)
            Cl = (CN * np.cos(alpha_eff)) - (CT * np.sin(alpha_eff))
            Cd = (CN * np.sin(alpha_eff)) + (CT * np.cos(alpha_eff))
            CM = -CN * (0.25 - (0.175 * (1 - ((2 * abs(alpha_eff)) / np.pi))))

        else:  # No stall
            Cl = self.Cl_alpha_3D * (alpha - alpha_0)
            alpha_i = Cl / (np.pi * self.aspect)
            alpha_eff = alpha - alpha_0 - alpha_i
            CT = self.Cd_0 * np.cos(alpha_eff)
            CN = (Cl + (CT * np.sin(alpha_eff))) / np.cos(alpha_eff)
            Cd = (CN * np.sin(alpha_eff)) + (CT * np.cos(alpha_eff))
            CM = -CN * (0.25 - (0.175 * (1 - ((2 * alpha_eff) / np.pi))))

        return Cl, Cd, CM
