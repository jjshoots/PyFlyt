import math
import os
import xml.etree.ElementTree as etxml

import numpy as np
from pybullet_utils import bullet_client

from PyFlyt.core.PID import PID


class Drone:
    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model="cf2x",
        use_camera=False,
        use_gimbal=False,
        camera_FOV=90,
        camera_frame_size=(128, 128),
    ):

        if physics_hz != 240.0:
            raise UserWarning(
                f"physics_hz is currently {physics_hz}, not the 240.0 that is recommended by pybullet. There may be physics errors."
            )

        self.p = p
        self.physics_hz = 1.0 / physics_hz
        self.ctrl_period = 1.0 / ctrl_hz
        file_dir = os.path.dirname(os.path.realpath(__file__))
        drone_dir = os.path.join(file_dir, f"../models/vehicles/{drone_model}.urdf")

        """ SPAWN """
        self.start_pos = start_pos
        self.start_orn = self.p.getQuaternionFromEuler(start_orn)
        self.Id = self.p.loadURDF(
            drone_dir,
            basePosition=self.start_pos,
            baseOrientation=self.start_orn,
            useFixedBase=False,
        )

        """
        DRONE CONTROL
            motor ids correspond to quadrotor X in PX4, using the ENU convention
            control commands are in the form of pitch-roll-yaw-thrust
        """

        # All the params for the drone
        URDF_TREE = etxml.parse(drone_dir).getroot()
        # self.mass = float(URDF_TREE[1][0][1].attrib['value'])
        # self.ixx = float(URDF_TREE[1][0][2].attrib['ixx'])
        # self.iyy = float(URDF_TREE[1][0][2].attrib['iyy'])
        # self.izz = float(URDF_TREE[1][0][2].attrib['izz'])
        # self.arm = float(URDF_TREE[0].attrib['arm'])
        self.thrust2weight = float(URDF_TREE[0].attrib["thrust2weight"])
        self.kf = float(URDF_TREE[0].attrib["kf"])
        self.km = float(URDF_TREE[0].attrib["km"])
        # self.max_speed_kmh = float(URDF_TREE[0].attrib['max_speed_kmh'])
        # self.gnd_eff_coeff = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        # self.prop_radius = float(URDF_TREE[0].attrib['prop_radius'])
        # self.drag_coeff_xy = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        # self.drag_coeff_z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        # self.dw_coeff_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        # self.dw_coeff_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        # self.dw_coeff_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        # self.length = float(URDF_TREE[1][2][1][0].attrib['length'])
        # self.radius = float(URDF_TREE[1][2][1][0].attrib['radius'])
        # self.collision_z_offset = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')][2]

        # the joint IDs corresponding to motorID 1234
        self.thr_coeff = np.array([[0.0, 0.0, 1.0]]) * self.kf
        self.tor_coeff = np.array([[0.0, 0.0, 1.0]]) * self.km
        self.tor_dir = np.array([[1.0], [1.0], [-1.0], [-1.0]])
        self.noise_ratio = 0.02

        # pseudo drag coef
        self.drag_const_xyz = 1e-3
        self.drag_const_pqr = 1e-4

        # maximum motor RPM
        self.max_rpm = np.sqrt((self.thrust2weight * 9.81) / (4 * self.kf))
        # motor modelled with first order ode, below is time const
        self.motor_tau = 0.01
        # motor mapping from command to individual motors
        self.motor_map = np.array(
            [
                [+1.0, -1.0, +1.0, +1.0],
                [-1.0, +1.0, +1.0, +1.0],
                [+1.0, +1.0, -1.0, +1.0],
                [-1.0, -1.0, -1.0, +1.0],
            ]
        )

        # input: angular velocity command
        # output: normalized angular torque command
        self.Kp_ang_vel = np.array([8e-3, 8e-3, 1e-2])
        self.Ki_ang_vel = np.array([2.5e-7, 2.5e-7, 1.3e-4])
        self.Kd_ang_vel = np.array([10e-5, 10e-5, 0.0])
        self.lim_ang_vel = np.array([1.0, 1.0, 1.0])

        # input: angular position command
        # output: angular velocity
        self.Kp_ang_pos = np.array([2.0, 2.0, 2.0])
        self.Ki_ang_pos = np.array([0.0, 0.0, 0.0])
        self.Kd_ang_pos = np.array([0.0, 0.0, 0.0])
        self.lim_ang_pos = np.array([3.0, 3.0, 3.0])

        # input: linear velocity command
        # output: angular position
        self.Kp_lin_vel = np.array([0.8, 0.8])
        self.Ki_lin_vel = np.array([0.1, 0.1])
        self.Kd_lin_vel = np.array([0.5, 0.5])
        self.lim_lin_vel = np.array([0.4, 0.4])

        # input: linear position command
        # outputs: linear velocity
        self.Kp_lin_pos = np.array([1.0, 1.0])
        self.Ki_lin_pos = np.array([0.0, 0.0])
        self.Kd_lin_pos = np.array([0.0, 0.0])
        self.lim_lin_pos = np.array([0.5, 0.5])

        # height controllers
        z_pos_PID = PID(1.0, 0.0, 0.0, 1.0, self.ctrl_period)
        z_vel_PID = PID(0.15, 1.0, 0.015, 0.3, self.ctrl_period)
        self.z_PIDs = [z_vel_PID, z_pos_PID]
        self.PIDs = []

        """ CAMERA """
        self.use_camera = use_camera
        self.use_gimbal = use_gimbal
        if self.use_camera:
            self.proj_mat = self.p.computeProjectionMatrixFOV(
                fov=camera_FOV, aspect=1.0, nearVal=0.1, farVal=255.0
            )
            self.camera_FOV = camera_FOV
            self.camera_frame_size = np.array(camera_frame_size)

        self.reset()

    def reset(self):
        self.set_mode(0)
        self.state = np.zeros((4, 3))
        self.setpoint = np.zeros((4))
        self.rpm = np.zeros((4))
        self.pwm = np.zeros((4))

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.update_state()

        if self.use_camera:
            self.capture_image()

    def update_drag(self):
        """adds drag to the model, this is not physically correct but only approximation"""
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)
        drag_xyz = -self.drag_const_xyz * (np.array(lin_vel) ** 2)
        drag_pqr = -self.drag_const_pqr * (np.array(ang_vel) ** 2)

        # warning, the physics is funky for bounces
        if len(self.p.getContactPoints()) == 0:
            self.p.applyExternalForce(
                self.Id, -1, drag_xyz, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.Id, -1, drag_pqr, self.p.LINK_FRAME)

    def rpm2forces(self, rpm):
        """maps rpm to individual motor forces and torques"""
        rpm = np.expand_dims(rpm, axis=1)
        thrust = (rpm**2) * self.thr_coeff
        torque = (rpm**2) * self.tor_coeff * self.tor_dir

        # add some random noise to the motor outputs
        thrust += np.random.randn(*thrust.shape) * self.noise_ratio * thrust
        torque += np.random.randn(*torque.shape) * self.noise_ratio * torque

        for idx, (thr, tor) in enumerate(zip(thrust, torque)):
            self.p.applyExternalForce(
                self.Id, idx, thr, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.Id, idx, tor, self.p.LINK_FRAME)

    def pwm2rpm(self, pwm):
        """model the motor using first order ODE, y' = T/tau * (setpoint - y)"""
        self.rpm += (self.physics_hz / self.motor_tau) * (self.max_rpm * pwm - self.rpm)

        return self.rpm

    def cmd2pwm(self, cmd):
        """maps angular torque commands to motor rpms"""
        pwm = np.matmul(self.motor_map, cmd)

        # deal with motor saturations
        if (high := np.max(pwm)) > 1.0:
            pwm /= high
        if (low := np.min(pwm)) < 0.05:
            pwm += (1.0 - pwm) / (1.0 - low) * (0.05 - low)

        return pwm

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

    def set_mode(self, mode):
        """
        vp, vq, vr = angular velocities
        p, q, r = angular positions
        u, v, w = linear velocities
        x, y, z = linear positions
        vx, vy, vz = linear velocities
        T = thrust

        sets the flight mode:
            0 - vp, vq, vr, T
            1 - p, q, r, vz
            2 - vp, vq, vr, z
            3 - p, q, r, z
            4 - u, v, vr, z
            5 - u, v, vr, vz
            6 - vx, vy, vr, vz
            7 - x, y, r, z
        """

        # preset setpoints on mode change
        # self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)
        if mode == 0:
            # racing mode, thrust to 0
            self.setpoint = np.array([0.0, 0.0, 0.0, -1.0])
        elif mode in [1, 5, 6]:
            # anything with a vz component, set to 0 vz
            self.setpoint = np.array([0.0, 0.0, 0.0, 0.0])
        elif mode == 7:
            # position mode just hold position
            self.setpoint = np.array([*self.state[-1, :2], self.state[1, -1]])
        else:
            # everything else set to 0 except z component maintain
            self.setpoint = np.array([0.0, 0.0, 0.0, 0.0])
            self.setpoint[-1] = self.state[-1, -1]

        # instantiate PIDs
        self.mode = mode
        if mode == 0 or mode == 2:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID]
        elif mode == 1 or mode == 3:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos,
                self.Ki_ang_pos,
                self.Kd_ang_pos,
                self.lim_ang_pos,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID]
        elif mode == 4 or mode == 5 or mode == 6:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos[:2],
                self.Ki_ang_pos[:2],
                self.Kd_ang_pos[:2],
                self.lim_ang_pos[:2],
                self.ctrl_period,
            )
            lin_vel_PID = PID(
                self.Kp_lin_vel,
                self.Ki_lin_vel,
                self.Kd_lin_vel,
                self.lim_lin_vel,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID, lin_vel_PID]
        elif mode == 7:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos,
                self.Ki_ang_pos,
                self.Kd_ang_pos,
                self.lim_ang_pos,
                self.ctrl_period,
            )
            lin_vel_PID = PID(
                self.Kp_lin_vel,
                self.Ki_lin_vel,
                self.Kd_lin_vel,
                self.lim_lin_vel,
                self.ctrl_period,
            )
            lin_pos_PID = PID(
                self.Kp_lin_pos,
                self.Ki_lin_pos,
                self.Kd_lin_pos,
                self.lim_lin_pos,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID, lin_vel_PID, lin_pos_PID]

        for controller in self.PIDs:
            controller.reset()

    def update_control(self):
        """runs through PID controllers"""
        output = None
        # angle controllers
        if self.mode == 0 or self.mode == 2:
            output = self.PIDs[0].step(self.state[0], self.setpoint[:3])
        elif self.mode == 1 or self.mode == 3:
            output = self.PIDs[1].step(self.state[1], self.setpoint[:3])
            output = self.PIDs[0].step(self.state[0], output)
        elif self.mode == 4 or self.mode == 5:
            output = self.PIDs[2].step(self.state[2][:2], self.setpoint[:2])
            output = np.array([-output[1], output[0]])
            output = self.PIDs[1].step(self.state[1][:2], output)
            output = self.PIDs[0].step(
                self.state[0], np.array([*output, self.setpoint[2]])
            )
        elif self.mode == 6:
            c = math.cos(self.state[1, -1])
            s = math.sin(self.state[1, -1])
            rot_mat = np.array([[c, -s], [s, c]]).T
            output = np.matmul(rot_mat, self.setpoint[:2])

            output = self.PIDs[2].step(self.state[2][:2], output)
            output = np.array([-output[1], output[0]])
            output = self.PIDs[1].step(self.state[1][:2], output)
            output = self.PIDs[0].step(
                self.state[0], np.array([*output, self.setpoint[2]])
            )
        elif self.mode == 7:
            output = self.PIDs[3].step(self.state[3][:2], self.setpoint[:2])

            c = math.cos(self.state[1, -1])
            s = math.sin(self.state[1, -1])
            rot_mat = np.array([[c, -s], [s, c]]).T
            output = np.matmul(rot_mat, output)

            output = self.PIDs[2].step(self.state[2][:2], output)
            output = np.array([-output[1], output[0], self.setpoint[2]])
            output = self.PIDs[1].step(self.state[1], output)
            output = self.PIDs[0].step(self.state[0], output)

        z_output = None
        # height controllers
        if self.mode == 0:
            z_output = np.clip(self.setpoint[-1], 0.0, 1.0)
        elif self.mode == 1 or self.mode == 5 or self.mode == 6:
            z_output = self.z_PIDs[0].step(self.state[2][-1], self.setpoint[-1])
            z_output = np.clip(z_output, 0, 1)
        elif self.mode == 2 or self.mode == 3 or self.mode == 4 or self.mode == 7:
            z_output = self.z_PIDs[1].step(self.state[3][-1], self.setpoint[-1])
            z_output = self.z_PIDs[0].step(self.state[2][-1], z_output)
            z_output = np.clip(z_output, 0, 1)

        # mix the commands
        self.pwm = self.cmd2pwm(np.array([*output, z_output]))

    def update_forces(self):
        self.rpm2forces(self.pwm2rpm(self.pwm))
        self.update_drag()

    @property
    def view_mat(self):
        # get the state of the camera on the robot
        camera_state = self.p.getLinkState(self.Id, 0)

        # pose and rot
        position = camera_state[0]

        # simulate gimballed camera if needed
        up_vector = None
        if self.use_gimbal:
            rot = np.array(self.p.getEulerFromQuaternion(camera_state[1]))
            rot[0] = 0.0
            rot[1] = 45 / 180 * math.pi
            rot = np.array(self.p.getQuaternionFromEuler(rot))
            rot = np.array(self.p.getMatrixFromQuaternion(rot)).reshape(3, 3)

            up_vector = np.matmul(rot, np.array([0.0, 0.0, 1.0]))
        else:
            # camera rotated upward 20 degrees
            rot = np.array(self.p.getEulerFromQuaternion(camera_state[1]))
            rot[1] += -20 / 180 * math.pi
            rot = np.array(self.p.getQuaternionFromEuler(rot))
            rot = np.array(self.p.getMatrixFromQuaternion(rot)).reshape(3, 3)

            # camera rotated upward 20 degrees
            up_vector = np.matmul(rot, np.array([0, 0, 1]))

        # target position is 1000 units ahead of camera relative to the current camera pos
        target = np.dot(rot, np.array([1000, 0, 0])) + np.array(position)

        return self.p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=up_vector,
        )

    def capture_image(self):
        _, _, self.rgbImg, self.depthImg, self.segImg = self.p.getCameraImage(
            width=self.camera_frame_size[1],
            height=self.camera_frame_size[0],
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
        )
        self.rgbImg = np.array(self.rgbImg).reshape(-1, *self.camera_frame_size)
        self.depthImg = np.array(self.depthImg).reshape(-1, *self.camera_frame_size)
        self.segImg = np.array(self.segImg).reshape(-1, *self.camera_frame_size)

    def update(self):
        """
        updates state and control
        """
        self.update_state()
        self.update_control()

        if self.use_camera:
            self.capture_image()
