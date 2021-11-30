import multiprocessing
import numpy as np
import xml.etree.ElementTree as etxml

from env.PID import *

import pybullet as p
from pybullet_utils import bullet_client

class Drone(multiprocessing.Process):
    def __init__(self, p: bullet_client.BulletClient, drone_dir, start_pos: np.ndarray, start_orn: np.ndarray):
        multiprocessing.Process.__init__(self)

        # default physics looprate is 240 Hz
        self.p = p
        self.period = 1. / 240.

        """ SPAWN """
        self.start_pos = start_pos
        self.start_orn = self.p.getQuaternionFromEuler(start_orn)
        self.Id = self.p.loadURDF(
            drone_dir + "/cf2x.urdf",
            basePosition=self.start_pos,
            baseOrientation=self.start_orn,
            useFixedBase=False
        )

        """
        DRONE CONTROL
            motor_id corresponds to QuadrotorX in PX4
            control commands are in the form of pitch-roll-yaw-thrust
                using ENU convention
        """

        # All the params for the drone
        URDF_TREE = etxml.parse(drone_dir + "/cf2x.urdf").getroot()
        self.mass = float(URDF_TREE[1][0][1].attrib['value'])
        self.ixx = float(URDF_TREE[1][0][2].attrib['ixx'])
        self.iyy = float(URDF_TREE[1][0][2].attrib['iyy'])
        self.izz = float(URDF_TREE[1][0][2].attrib['izz'])
        self.arm = float(URDF_TREE[0].attrib['arm'])
        self.thrust2weight = float(URDF_TREE[0].attrib['thrust2weight'])
        self.kf = float(URDF_TREE[0].attrib['kf'])
        self.km = float(URDF_TREE[0].attrib['km'])
        self.max_speed_kmh = float(URDF_TREE[0].attrib['max_speed_kmh'])
        self.gnd_eff_coeff = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        self.prop_radius = float(URDF_TREE[0].attrib['prop_radius'])
        self.drag_coeff_xy = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        self.drag_coeff_z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        self.dw_coeff_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        self.dw_coeff_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        self.dw_coeff_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        self.length = float(URDF_TREE[1][2][1][0].attrib['length'])
        self.radius = float(URDF_TREE[1][2][1][0].attrib['radius'])
        self.collision_z_offset = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')][2]

        # the joint IDs corresponding to motorID 1234
        self.motor_id = np.array([0, 2, 1, 3])
        self.thr_coeff = np.array([[0., 0., 1.]]) * self.kf
        self.tor_coeff = np.array([[0., 0., 1.]]) * self.km
        self.tor_dir = np.array([[1.], [1.], [-1.], [-1.]])
        self.noise_ratio = 0.02

        # maximum motor RPM
        self.max_rpm = np.sqrt((self.thrust2weight * 9.81) / (4 * self.kf))
        # motor modelled with first order ode, below is time const
        self.motor_tau = 0.01
        # motor mapping from angular torque to individual motors
        self.motor_map = np.array(
            [
                [+1., -1., +1., +1.],
                [-1., +1., +1., +1.],
                [+1., +1., -1., +1.],
                [-1., -1., -1., +1.]
            ]
        )

        # outputs normalized body torque commands
        self.Kp_ang_vel = np.array([8e-3, 8e-3, 1e-2])
        self.Ki_ang_vel = np.array([1e-9, 1e-9, 5e-7])
        self.Kd_ang_vel = np.array([0., 0., 0.])
        self.lim_ang_vel = np.array([1., 1., 1.])

        # outputs angular rate
        self.Kp_ang_pos = np.array([.5, .5, 0.])
        self.Ki_ang_pos = np.array([0., 0., 0.])
        self.Kd_ang_pos = np.array([0., 0., 0.])
        self.lim_ang_pos = np.array([2., 2., 2.])

        # outputs angular position
        self.Kp_lin_vel = np.array([7., 7.])
        self.Ki_lin_vel = np.array([0., 0.])
        self.Kd_lin_vel = np.array([1., 1.])
        self.lim_lin_vel = np.array([.6, .6])

        # height controllers
        z_pos_PID = PID(3., 0., 0., 5., self.period)
        z_vel_PID = PID(0.2, .005, 0., 1., self.period)
        self.z_PIDs = [z_vel_PID, z_pos_PID]
        self.PIDs = []

        self.reset()


    def reset(self):
        self.set_mode(0)
        self.state = None
        self.rpm = np.array([0., 0., 0., 0.])
        self.setpoint = np.array([0., 0., 0., 0.])

        for PID in self.PIDs:
            PID.reset()

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)


    def rpm2forces(self, rpm):
        """ maps rpm to individual motor forces and torques"""
        rpm = np.expand_dims(rpm, axis=1)
        thrust = (rpm ** 2) * self.thr_coeff
        torque = (rpm ** 2) * self.tor_coeff * self.tor_dir

        # add some random noise to the motor outputs
        thrust += np.random.randn(*thrust.shape) * self.noise_ratio * thrust
        torque += np.random.randn(*torque.shape) * self.noise_ratio * torque

        for idx, thr, tor in zip(self.motor_id, thrust, torque):
            self.p.applyExternalForce(self.Id, idx, thr, [0., 0., 0.], self.p.LINK_FRAME)
            self.p.applyExternalTorque(self.Id, idx, tor, self.p.LINK_FRAME)


    def pwm2rpm(self, pwm):
        """ model the motor using first order ODE, y' = T/tau * (setpoint - y) """
        self.rpm += (self.period / self.motor_tau) * (self.max_rpm * pwm - self.rpm)

        return self.rpm


    def cmd2pwm(self, cmd):
        """ maps angular torque commands to motor rpms """
        pwm = np.matmul(self.motor_map, cmd)

        min = np.min(pwm)
        max = np.max(pwm)

        # deal with motor saturations
        if min < 0.:
            pwm = pwm - min
        if max > 1.:
            pwm = pwm / max

        return pwm


    def update_state(self):
        """ ang_vel, ang_pos, lin_vel, lin_pos """
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
        sets the flight mode:
            0 - vp, vq, vr, vz  :  \n
            1 - p, q, r, vz     :  \n
            2 - vp, vq, vr, z   :  \n
            3 - p, q, r, z      :  \n
            4 - u, v, vr, z     :  \n
            5 - u, v, vr, vz     :  \n
        """

        self.mode = mode
        if mode == 0 or mode == 2:
            ang_vel_PID = PID(self.Kp_ang_vel, self.Ki_ang_vel, self.Kd_ang_vel, self.lim_ang_vel, self.period)
            self.PIDs = [ang_vel_PID]
        elif mode == 1 or mode == 3:
            ang_vel_PID = PID(self.Kp_ang_vel, self.Ki_ang_vel, self.Kd_ang_vel, self.lim_ang_vel, self.period)
            ang_pos_PID = PID(self.Kp_ang_pos, self.Ki_ang_pos, self.Kd_ang_pos, self.lim_ang_pos, self.period)
            self.PIDs = [ang_vel_PID, ang_pos_PID]
        elif mode == 4 or mode == 5:
            ang_vel_PID = PID(self.Kp_ang_vel, self.Ki_ang_vel, self.Kd_ang_vel, self.lim_ang_vel, self.period)
            ang_pos_PID = PID(self.Kp_ang_pos[:2], self.Ki_ang_pos[:2], self.Kd_ang_pos[:2], self.lim_ang_pos[:2], self.period)
            lin_vel_PID = PID(self.Kp_lin_vel, self.Ki_lin_vel, self.Kd_lin_vel, self.lim_lin_vel, self.period)
            self.PIDs = [ang_vel_PID, ang_pos_PID, lin_vel_PID]


    def update_control(self):
        """ runs through PID controllers """
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
            output = self.PIDs[0].step(self.state[0], np.array([*output, self.setpoint[2]]))

        z_output = None
        # height controllers
        if self.mode == 0 or self.mode == 1 or self.mode == 5:
            z_output = self.z_PIDs[0].step(self.state[2][-1], self.setpoint[-1])
            z_output = np.clip(z_output, 0, 1)
        elif self.mode == 2 or self.mode == 3 or self.mode == 4:
            z_output = self.z_PIDs[1].step(self.state[3][-1], self.setpoint[-1])
            z_output = self.z_PIDs[0].step(self.state[2][-1], z_output)
            z_output = np.clip(z_output, 0, 1)

        # mix the commands
        command = np.array([*output, z_output])

        self.rpm2forces(self.pwm2rpm(self.cmd2pwm(command)))


    def update(self):
        """
        updates state and control
        """
        self.update_state()
        self.update_control()
