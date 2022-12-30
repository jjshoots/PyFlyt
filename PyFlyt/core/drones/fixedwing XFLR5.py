from __future__ import annotations

import math

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.mplot3d import axes3d
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
        aerofoil_df: pd.core.frame.DataFrame = pd.DataFrame()
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

        """Reads AeroData folder and create 3D surface for Cl & Cd values for interpolation"""
        for file in os.listdir(self.aero_data_path):
            txtfile = os.path.join(self.aero_data_path, file)
            with open(txtfile, "rb") as f:
                content = f.readlines()

                # Read line 3, decode text, and extract aerofoil name from index 23
                aerofoil_name = content[2].decode('utf-8')[23:-1]

                # Create dataframe entry for aerofoil if entry doesnt exist
                if aerofoil_name not in aerofoil_df:
                    df = pd.DataFrame(index=[aerofoil_name], columns=['Cl', 'Cd'])
                    aerofoil_df = pd.concat([aerofoil_df, df])

                # Read line 7, decode text, and extract Reynolds number from index 28
                re = float(content[7].decode('utf-8')[28:34])

                # Point to top of txt file
                f.seek(0)

                # Load values of alpha, Cl, and Cd
                alpha, Cl, Cd = np.loadtxt(f, skiprows=11, usecols=[
                                           0, 1, 2], unpack=True)
                Cl_df = pd.DataFrame(data=Cl, index=alpha, columns=[re])
                Cd_df = pd.DataFrame(data=Cd, index=alpha, columns=[re])

                # Get zero alpha index
                idx = np.where(alpha == 0)
                # Get alpha interval
                alpha_interval = np.absolute(alpha[idx]-alpha[idx-1])
                alpha_array = np.linspace(-90, 90, int(180/alpha_interval))
                print(alpha_array)
                # neg_alpha = np.linspace(-90, alpha[0], neg_elems)
                # neg_Cl = np.linspace(0, Cl[0], neg_elems)
                # neg_Cd = np.linspace(0, Cd[0], neg_elems)


                # # Calculate number of elements to be added on the positive side
                # pos_elems = int(np.absolute(90-alpha[-1]) / alpha_interval)
                # pos_alpha = np.linspace(alpha[-1], 90, pos_elems)
                # pos_Cl = np.linspace(0, Cl[0], pos_elems)
                # pos_Cd = np.linspace(0, Cd[0], pos_elems)

                

                # print(aerofoil_df.loc[aerofoil_name, 'Cl'])
                # print(aerofoil_df.insert(Cl_df, ))
                

                

        ax = plt.figure().add_subplot(projection='3d')
        X = []
        Y = np.array([])
        Z = np.array([])
        for i in aerofoil_df['NACA0009']:
            X.append(i)
            Y = np.append(Z, aerofoil_df['NACA0009'][i]['alpha'], axis=0)
            Z = np.append(Y, aerofoil_df['NACA0009'][i]['Cl'], axis=0)

        X, Y = np.meshgrid(X, Y)
        # X, Y, Z = axes3d.get_test_data(0.05)

        # Plot the 3D surface
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                        alpha=0.3)

        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

        ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
               xlabel='X', ylabel='Y', zlabel='Z')

        plt.show()

        """Reads fixedwing.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)
            motor_params = all_params["motor_params"]
            drag_params = all_params["drag_params"]
            ctrl_params = all_params["control_params"]

            # motor thrust and torque constants
            self.t2w = motor_params["thrust_to_weight"]
            self.kf = motor_params["thrust_const"]
            self.km = motor_params["torque_const"]

            # the joint IDs corresponding to motorID 1234
            self.thr_coeff = np.array([[0.0, 0.0, 1.0]]) * self.kf
            self.tor_coeff = np.array([[0.0, 0.0, 1.0]]) * self.km
            self.tor_dir = np.array([[1.0], [1.0], [-1.0], [-1.0]])
            self.noise_ratio = motor_params["motor_noise_ratio"]

            # pseudo drag coef
            self.drag_const_xyz = drag_params["drag_const_xyz"]
            self.drag_const_pqr = drag_params["drag_const_pqr"]

            # maximum motor RPM
            self.max_rpm = np.sqrt((self.t2w * 9.81) / (4 * self.kf))
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

    def reset(self):
        pass

    def set_mode(self, mode):
        pass

    def update_avionics(self):
        pass

    def update_physics(self):
        pass
