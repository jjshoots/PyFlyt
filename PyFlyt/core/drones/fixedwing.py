from __future__ import annotations

import math

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions import CtrlClass, DroneClass
from ..pid import PID


class FixedWing(DroneClass):
    """FixedWing instance that handles everything about a FixedWing."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model: str = "fixedwing",
        model_dir: None | str = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 0,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        np_random: None | np.random.RandomState = None,
    ):
        """Creates a fixed wing UAV and handles all relevant control and physics.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            ctrl_hz (int): ctrl_hz
            physics_hz (int): physics_hz
            model_dir (None | str): model_dir
            drone_model (str): drone_model
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
            np_random (None | np.random.RandomState): np_random
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            ctrl_hz=ctrl_hz,
            physics_hz=physics_hz,
            model_dir=model_dir,
            drone_model=drone_model,
            np_random=np_random,
        )

        """Reads fixedwing.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)

            # Main wing
            main_wing_params = all_params["main_wing_params"]

            # Left flapped wing
            left_wing_flapped_params = all_params["left_wing_flapped_params"]

            # Right flapped wing
            right_wing_flapped_params = all_params["right_wing_flapped_params"]

            # Horizontal tail
            horizontal_tail_params = all_params["horizontal_tail_params"]

            # Vertical tail
            vertical_tail_params = all_params["vertical_tail_params"]

            # Motor
            motor_params = all_params["motor_params"]
            self.t2w = motor_params["thrust_to_weight"]

            self.kf = motor_params["thrust_const"]
            self.km = motor_params["torque_const"]

            self.thr_coeff = np.array([0.0, 1.0, 0.0]) * self.kf
            self.tor_coeff = np.array([0.0, 1.0, 0.0]) * self.km
            self.tor_dir = np.array([[1.0]])
            self.noise_ratio = motor_params["motor_noise_ratio"]

            self.max_rpm = np.sqrt((self.t2w * 9.81) / (self.kf))
            self.motor_tau = 0.01

            # Combine [ail_left, ail_right, hori_tail, main_wing, vert_tail]
            self.aerofoil_params = [
                left_wing_flapped_params,
                right_wing_flapped_params,
                horizontal_tail_params,
                main_wing_params,
                vertical_tail_params,
            ]

        """ CAMERA """
        self.use_camera = use_camera
        if self.use_camera:
            self.proj_mat = self.p.computeProjectionMatrixFOV(
                fov=camera_FOV_degrees, aspect=1.0, nearVal=0.1, farVal=255.0
            )
            self.use_gimbal = use_gimbal
            self.camera_angle_degrees = camera_angle_degrees
            self.camera_FOV_degrees = camera_FOV_degrees
            self.camera_resolution = np.array(camera_resolution)

        """ CUSTOM CONTROLLERS """
        # dictionary mapping of controller_id to controller objects
        self.registered_controllers = dict()
        self.instanced_controllers = dict()
        self.registered_base_modes = dict()

        self.reset()

    def rpm2forces(self, rpm):
        """maps rpm to individual motor force and torque"""
        rpm = np.expand_dims(rpm, axis=1)
        thrust = (rpm**2) * self.thr_coeff
        torque = (rpm**2) * self.tor_coeff * self.tor_dir

        # add some random noise to the motor output
        thrust += self.np_random.randn(*thrust.shape) * self.noise_ratio * thrust
        torque += self.np_random.randn(*torque.shape) * self.noise_ratio * torque

        self.p.applyExternalForce(
            self.Id, 0, thrust[0], [0.0, 0.0, 0.0], self.p.LINK_FRAME
        )
        self.p.applyExternalTorque(self.Id, 0, torque[0], self.p.LINK_FRAME)

    def pwm2rpm(self, pwm):
        """model the motor using first order ODE, y' = T/tau * (setpoint - y)"""
        self.rpm += (self.physics_hz / self.motor_tau) * (self.max_rpm * pwm - self.rpm)

        return self.rpm

    def cmd2pwm(self, cmd):
        """maps angular torque commands to motor rpms"""
        pwm = cmd
        # deal with motor saturations
        if (high := np.max(pwm)) > 1.0:
            pwm /= high
        if (low := np.min(pwm)) < 0.05:
            pwm += (1.0 - pwm) / (1.0 - low) * (0.05 - low)

        pwm = cmd

        return pwm

    def update_forces(self):
        """Calculates and applies forces acting on UAV"""
        # Motor thrust
        self.pwm = self.cmd2pwm(self.cmd[3])  # Extract Throttle cmd
        self.rpm2forces(self.pwm2rpm(self.pwm))

        # Left aileron Forces
        self.left_aileron_forces(0)

        # Right aileron Forces
        self.right_aileron_forces(1)

        # Horizontal tail Forces
        self.horizontal_tail_forces(2)

        # Main wing Forces
        self.main_wing_forces(3)

        # Vertical tail Forces
        self.vertical_tail_forces(4)

    def left_aileron_forces(self, i):

        vel = self.surface_vels[i]
        local_surface_vel = np.matmul((vel), self.rotation)

        alpha = np.arctan2(-local_surface_vel[2], local_surface_vel[1])
        alpha_deg = np.rad2deg(alpha)

        defl = self.aerofoil_params[i]["defl_lim"] * self.cmd[1]
        [Cl, Cd, CM] = self.get_aero_data(self.aerofoil_params[i], defl, alpha_deg)

        freestream_speed = np.linalg.norm(
            [local_surface_vel[1], local_surface_vel[2]]
        )  # Only in Y and Z directions
        Q = 0.5 * 1.225 * np.square(freestream_speed)  # Dynamic pressure
        area = self.aerofoil_params[i]["chord"] * self.aerofoil_params[i]["span"]

        lift = Q * area * Cl
        drag = Q * area * Cd  # Negative coz front is positive
        pitching_moment = Q * area * CM * self.aerofoil_params[i]["chord"]

        up = (lift * np.cos(alpha)) + (drag * np.sin(alpha))
        front = (lift * np.sin(alpha)) - (drag * np.cos(alpha))

        self.p.applyExternalForce(
            self.Id,
            self.surface_ids[i],
            [0, front, up],
            [0.0, 0.0, 0.0],
            self.p.LINK_FRAME,
        )
        self.p.applyExternalTorque(
            self.Id, self.surface_ids[i], [pitching_moment, 0, 0], self.p.LINK_FRAME
        )

    def right_aileron_forces(self, i):

        vel = self.surface_vels[i]
        local_surface_vel = np.matmul((vel), self.rotation)

        alpha = np.arctan2(-local_surface_vel[2], local_surface_vel[1])
        alpha_deg = np.rad2deg(alpha)

        defl = self.aerofoil_params[i]["defl_lim"] * -self.cmd[1]
        [Cl, Cd, CM] = self.get_aero_data(self.aerofoil_params[i], defl, alpha_deg)

        freestream_speed = np.linalg.norm(
            [local_surface_vel[1], local_surface_vel[2]]
        )  # Only in Y and Z directions
        Q = 0.5 * 1.225 * np.square(freestream_speed)  # Dynamic pressure
        area = self.aerofoil_params[i]["chord"] * self.aerofoil_params[i]["span"]

        lift = Q * area * Cl
        drag = Q * area * Cd
        pitching_moment = Q * area * CM * self.aerofoil_params[i]["chord"]

        up = (lift * np.cos(alpha)) + (drag * np.sin(alpha))
        front = (lift * np.sin(alpha)) - (drag * np.cos(alpha))

        self.p.applyExternalForce(
            self.Id,
            self.surface_ids[i],
            [0, front, up],
            [0.0, 0.0, 0.0],
            self.p.LINK_FRAME,
        )
        self.p.applyExternalTorque(
            self.Id, self.surface_ids[i], [pitching_moment, 0, 0], self.p.LINK_FRAME
        )

    def horizontal_tail_forces(self, i):

        vel = self.surface_vels[i]
        local_surface_vel = np.matmul((vel), self.rotation)

        alpha = np.arctan2(-local_surface_vel[2], local_surface_vel[1])
        alpha_deg = np.rad2deg(alpha)

        defl = self.aerofoil_params[i]["defl_lim"] * -self.cmd[0]
        [Cl, Cd, CM] = self.get_aero_data(self.aerofoil_params[i], defl, alpha_deg)

        freestream_speed = np.linalg.norm(
            [local_surface_vel[1], local_surface_vel[2]]
        )  # Only in Y and Z directions
        Q = 0.5 * 1.225 * np.square(freestream_speed)  # Dynamic pressure
        area = self.aerofoil_params[i]["chord"] * self.aerofoil_params[i]["span"]

        lift = Q * area * Cl
        drag = Q * area * Cd
        pitching_moment = Q * area * CM * self.aerofoil_params[i]["chord"]

        up = (lift * np.cos(alpha)) + (drag * np.sin(alpha))
        front = (lift * np.sin(alpha)) - (drag * np.cos(alpha))

        self.p.applyExternalForce(
            self.Id,
            self.surface_ids[i],
            [0, front, up],
            [0.0, 0.0, 0.0],
            self.p.LINK_FRAME,
        )
        self.p.applyExternalTorque(
            self.Id, self.surface_ids[i], [pitching_moment, 0, 0], self.p.LINK_FRAME
        )

    def main_wing_forces(self, i):

        vel = self.surface_vels[i]
        local_surface_vel = np.matmul((vel), self.rotation)

        alpha = np.arctan2(-local_surface_vel[2], local_surface_vel[1])
        alpha_deg = np.rad2deg(alpha)

        [Cl, Cd, CM] = self.get_aero_data(self.aerofoil_params[i], 0, alpha_deg)

        freestream_speed = np.linalg.norm(
            [local_surface_vel[1], local_surface_vel[2]]
        )  # Only in Y and Z directions
        Q = 0.5 * 1.225 * np.square(freestream_speed)  # Dynamic pressure
        area = self.aerofoil_params[i]["chord"] * self.aerofoil_params[i]["span"]

        lift = Q * area * Cl
        drag = Q * area * Cd
        pitching_moment = Q * area * CM * self.aerofoil_params[i]["chord"]

        up = (lift * np.cos(alpha)) + (drag * np.sin(alpha))
        front = (lift * np.sin(alpha)) - (drag * np.cos(alpha))

        self.p.applyExternalForce(
            self.Id, self.surface_ids[i], [0, front, up], [0, 0, 0], self.p.LINK_FRAME
        )
        self.p.applyExternalTorque(
            self.Id, self.surface_ids[i], [pitching_moment, 0, 0], self.p.LINK_FRAME
        )

    def vertical_tail_forces(self, i):

        vel = self.surface_vels[i]
        local_surface_vel = np.matmul((vel), self.rotation)

        alpha = np.arctan2(-local_surface_vel[0], local_surface_vel[1])
        alpha_deg = np.rad2deg(alpha)

        defl = self.aerofoil_params[i]["defl_lim"] * -self.cmd[2]
        [Cl, Cd, CM] = self.get_aero_data(self.aerofoil_params[i], defl, alpha_deg)

        freestream_speed = np.linalg.norm(
            [local_surface_vel[0], local_surface_vel[1]]
        )  # Only in X and Y directions
        Q = 0.5 * 1.225 * np.square(freestream_speed)  # Dynamic pressure
        area = self.aerofoil_params[i]["chord"] * self.aerofoil_params[i]["span"]

        lift = Q * area * Cl
        drag = Q * area * Cd
        pitching_moment = Q * area * CM * self.aerofoil_params[i]["chord"]

        right = (lift * np.cos(alpha)) + (drag * np.sin(alpha))
        front = (lift * np.sin(alpha)) - (drag * np.cos(alpha))

        self.p.applyExternalForce(
            self.Id,
            self.surface_ids[i],
            [right, front, 0],
            [0.0, 0.0, 0.0],
            self.p.LINK_FRAME,
        )
        self.p.applyExternalTorque(
            self.Id, self.surface_ids[i], [0, 0, pitching_moment], self.p.LINK_FRAME
        )

    def get_aero_data(self, params, defl, alpha):
        """Returns Cl, Cd, and CM for a given aerofoil, control surface deflection, and alpha"""

        AR = params["span"] / params["chord"]
        defl = np.deg2rad(defl)
        alpha = np.deg2rad(alpha)

        Cl_alpha_3D = params["Cl_alpha_2D"] * (AR / (AR + ((2 * (AR + 4)) / (AR + 2))))

        theta_f = np.arccos((2 * params["flap_to_chord"]) - 1)
        tau = 1 - ((theta_f - np.sin(theta_f)) / np.pi)
        delta_Cl = Cl_alpha_3D * tau * params["eta"] * defl
        delta_Cl_max = params["flap_to_chord"] * delta_Cl

        alpha_stall_P_base = np.deg2rad(params["alpha_stall_P_base"])
        alpha_stall_N_base = np.deg2rad(params["alpha_stall_N_base"])

        alpha_0_base = np.deg2rad(params["alpha_0_base"])

        Cl_max_P = Cl_alpha_3D * (alpha_stall_P_base - alpha_0_base) + delta_Cl_max
        Cl_max_N = Cl_alpha_3D * (alpha_stall_N_base - alpha_0_base) + delta_Cl_max

        alpha_0 = alpha_0_base - (delta_Cl / Cl_alpha_3D)
        alpha_stall_P = alpha_0 + ((Cl_max_P) / Cl_alpha_3D)
        alpha_stall_N = alpha_0 + ((Cl_max_N) / Cl_alpha_3D)

        # Check if stalled
        if (alpha >= alpha_stall_P) or (alpha <= alpha_stall_N):
            if alpha >= alpha_stall_P:
                # Stall calculations to find alpha_i at stall
                Cl_stall = Cl_alpha_3D * (alpha_stall_P - alpha_0)
                alpha_i_at_stall = Cl_stall / (np.pi * AR)
                # alpha_i post-stall Pos
                alpha_i = np.interp(
                    alpha, [alpha_stall_P, np.pi / 2], [alpha_i_at_stall, 0]
                )

            elif alpha <= alpha_stall_N:
                # Stall calculations to find alpha_i at stall
                Cl_stall = Cl_alpha_3D * (alpha_stall_N - alpha_0)
                alpha_i_at_stall = Cl_stall / (np.pi * AR)
                # alpha_i post-stall Neg
                alpha_i = np.interp(
                    alpha, [-np.pi / 2, alpha_stall_N], [0, alpha_i_at_stall]
                )

            alpha_eff = alpha - alpha_0 - alpha_i
            # Drag coefficient at 90 deg dependent on deflection angle
            Cd_90 = (
                ((-4.26 * (10**-2)) * (defl**2))
                + ((2.1 * (10**-1)) * defl)
                + 1.98
            )
            CN = (
                Cd_90
                * np.sin(alpha_eff)
                * (
                    1 / (0.56 + 0.44 * abs(np.sin(alpha_eff)))
                    - 0.41 * (1 - np.exp(-17 / AR))
                )
            )
            CT = 0.5 * params["Cd_0"] * np.cos(alpha_eff)
            Cl = (CN * np.cos(alpha_eff)) - (CT * np.sin(alpha_eff))
            Cd = (CN * np.sin(alpha_eff)) + (CT * np.cos(alpha_eff))
            CM = -CN * (0.25 - (0.175 * (1 - ((2 * abs(alpha_eff)) / np.pi))))

        else:  # No stall
            Cl = Cl_alpha_3D * (alpha - alpha_0)
            alpha_i = Cl / (np.pi * AR)
            alpha_eff = alpha - alpha_0 - alpha_i
            CT = params["Cd_0"] * np.cos(alpha_eff)
            CN = (Cl + (CT * np.sin(alpha_eff))) / np.cos(alpha_eff)
            Cd = (CN * np.sin(alpha_eff)) + (CT * np.cos(alpha_eff))
            CM = -CN * (0.25 - (0.175 * (1 - ((2 * alpha_eff) / np.pi))))

        return Cl, Cd, CM

    def update_state(self):
        """ang_vel, ang_pos, lin_vel, lin_pos"""
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)
        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

        # get the surface velocities and angles
        for i, id in enumerate(self.surface_ids):
            state = self.p.getLinkState(self.Id, id, computeLinkVelocity=True)
            self.surface_vels[i] = state[-2]

        self.orn = state[-3]
        self.rotation = np.array(self.p.getMatrixFromQuaternion(self.orn)).reshape(3, 3)
        # print("Vel: {}, Height:{}".format(lin_vel, lin_pos[2]))

    def update_control(self):
        """runs through controllers"""
        mode = self.mode

        # custom controllers run first if any
        if self.mode in self.registered_controllers.keys():
            custom_output = self.instanced_controllers[self.mode].step(
                self.state, self.setpoint
            )
            assert custom_output.shape == (
                4,
            ), f"custom controller outputting wrong shape, expected (4, ) but got {custom_output.shape}."

            mode = self.registered_base_modes[self.mode]

        # Final cmd, [Roll, Pitch, Yaw, Throttle] from [-1, 1]
        self.cmd = self.setpoint
        # self.cmd = cmds

    @property
    def view_mat(self):
        """view_mat."""
        # get the state of the camera on the robot
        camera_state = self.p.getLinkState(self.Id, 0)

        # UAV orientation
        orn = camera_state[1]
        rotation = np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_offset = [0, -3, 1]
        cam_offset_world_frame = np.matmul(cam_offset, rotation.transpose())

        # pose and rot
        position = np.array(camera_state[0]) + cam_offset_world_frame

        # simulate gimballed camera if needed
        up_vector = None
        if self.use_gimbal:
            # camera tilted downward for gimballed mode
            rot = np.array(self.p.getEulerFromQuaternion(camera_state[1]))
            rot[0] = 0.0
            rot[1] = self.camera_angle_degrees / 180 * math.pi
            rot = np.array(self.p.getQuaternionFromEuler(rot))
            rot = np.array(self.p.getMatrixFromQuaternion(rot)).reshape(3, 3)

            up_vector = np.matmul(rot, np.array([0.0, 0.0, 1.0]))
        else:
            # camera rotated upward for FPV mode
            rot = np.array(self.p.getEulerFromQuaternion(camera_state[1]))
            rot[0] += -self.camera_angle_degrees / 180 * math.pi
            rot = np.array(self.p.getQuaternionFromEuler(rot))
            rot = np.array(self.p.getMatrixFromQuaternion(rot)).reshape(3, 3)

            up_vector = np.matmul(rot, np.array([0, 0, 1]))

        # target position is 1000 units ahead of camera relative to the current camera pos
        target = camera_state[0]

        return self.p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=up_vector,
        )

    def capture_image(self):
        """capture_image."""
        _, _, self.rgbaImg, self.depthImg, self.segImg = self.p.getCameraImage(
            width=self.camera_resolution[1],
            height=self.camera_resolution[0],
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
        )
        self.rgbaImg = np.array(self.rgbaImg).reshape(*self.camera_resolution, -1)
        self.depthImg = np.array(self.depthImg).reshape(*self.camera_resolution, -1)
        self.segImg = np.array(self.segImg).reshape(*self.camera_resolution, -1)

        UAV_pos = self.state[-1]
        UAV_yaw = np.rad2deg(np.arctan2(-UAV_pos[0], UAV_pos[1]))
        UAV_pitch = np.rad2deg(
            np.abs(np.arctan(UAV_pos[2] / np.linalg.norm([UAV_pos[0], UAV_pos[1]])))
        )

        self.p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=UAV_yaw,
            cameraPitch=UAV_pitch,
            cameraTargetPosition=[0, 0, 5],
        )

        return np.array(self.rgbaImg).reshape(
            self.camera_resolution[1], self.camera_resolution[0], -1
        )

    def reset(self):
        self.set_mode(1)
        self.setpoint = np.zeros((4))
        self.rpm = np.zeros((1))
        self.pwm = np.zeros((1))
        self.cmd = np.zeros((4))
        self.surface_orns = [0] * 5
        self.surface_vels = [0] * 5

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)

        self.p.resetBaseVelocity(self.Id, [0, 20, 0], [0, 0, 0])

        # [ail_left, ail_right, hori_tail, main_wing, vert_tail]
        # Maps .urdf idx to surface_ids
        self.surface_ids = [3, 4, 1, 5, 2]
        self.update_state()

        if self.use_camera:
            self.capture_image()
        pass

    def set_mode(self, mode):
        """
        Mode 1 - [Roll, Pitch, Yaw, Throttle]
        """
        # WIP, copied and pasted from quadx
        if (mode < -1 or mode > 7) and mode not in self.registered_controllers.keys():
            raise ValueError(
                f"`mode` must be between -1 and 7 or be registered in {self.registered_controllers.keys()=}, got {mode}."
            )

        self.mode = mode

        # for custom modes
        if mode in self.registered_controllers.keys():
            self.instanced_controllers[mode] = self.registered_controllers[mode]()
            mode = self.registered_base_modes[mode]

    def update_physics(self):
        self.update_state()
        self.update_forces()

    def update_avionics(self):
        """
        updates state and control
        """
        self.update_control()

        if self.use_camera:
            self.capture_image()
