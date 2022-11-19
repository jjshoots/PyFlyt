import time
from multiprocessing import Pool

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from PyFlyt.core.drone import Drone


class Aviary(bullet_client.BulletClient):
    def __init__(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        render=False,
        use_camera=False,
        camera_frame_size=(128, 128),
    ):
        super().__init__(p.GUI if render else p.DIRECT)
        print("\033[A                             \033[A")

        # default physics looprate is 240 Hz
        # do not change because pybullet doesn't like it
        self.period = 1.0 / 240.0
        self.now = time.time()

        self.start_pos = start_pos
        self.start_orn = start_orn
        self.use_camera = use_camera
        self.camera_frame_size = camera_frame_size

        self.drone_model = "cf2x"
        self.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.render = render
        self.rtf_print_interval = 24
        self.rtf_debug_line = self.addUserDebugText(
            text="RTF here", textPosition=[0, 0, 0], textColorRGB=[1, 0, 0]
        )

        self.reset()

    def reset(self):
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.step_count = 0

        """ CONSTRUCT THE WORLD """
        self.planeId = self.loadURDF("plane.urdf", useFixedBase=True, globalScaling=1.0)
        # p.changeVisualShape(
        #     self.planeId,
        #     linkIndex=-1,
        #     rgbaColor=(0, 0, 0, 1),
        # )

        # spawn drones
        self.drones = []
        for start_pos, start_orn in zip(self.start_pos, self.start_orn):
            self.drones.append(
                Drone(
                    self,
                    start_pos=start_pos,
                    start_orn=start_orn,
                    drone_model=self.drone_model,
                    use_camera=self.use_camera,
                    camera_frame_size=self.camera_frame_size,
                )
            )

        self.armed = [1] * self.num_drones

    @property
    def num_drones(self):
        return len(self.drones)

    @property
    def states(self):
        """
        returns a list of states for each drone in the aviary
        """
        states = []
        for drone in self.drones:
            states.append(drone.state)

        states = np.stack(states, axis=0)

        return states

    def set_armed(self, settings):
        assert len(settings) == len(self.armed), "incorrect go length"
        self.armed = settings

    def set_mode(self, flight_mode):
        """
        sets the flight mode for each drone
        """
        for drone in self.drones:
            drone.set_mode(flight_mode)

    def set_setpoints(self, setpoints):
        """
        commands each drone to go to a setpoint as specified in a list
        """
        for i, drone in enumerate(self.drones):
            drone.setpoint = setpoints[i]

    def step(self):
        """
        Steps the environment
        """
        if self.render:
            elapsed = time.time() - self.now
            time.sleep(max(self.period - elapsed, 0.0))
            self.now = time.time()

            # calculate real time factor
            RTF = self.period / (elapsed + 1e-6)

            if self.step_count % self.rtf_print_interval == 0:
                # handle case where sometimes elapsed becomes 0
                if elapsed != 0.0:
                    self.rtf_debug_line = self.addUserDebugText(
                        text=f"RTF: {str(RTF)[:7]}",
                        textPosition=[0, 0, 0],
                        textColorRGB=[1, 0, 0],
                        replaceItemUniqueId=self.rtf_debug_line,
                    )

            # print(f'RTF: {RTF}')

        for drone, armed in zip(self.drones, self.armed):
            if armed:
                drone.update()

        self.stepSimulation()
        self.performCollisionDetection()
        self.step_count += 1
