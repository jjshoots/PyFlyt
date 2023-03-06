"""The Aviary class, the core of how PyFlyt handles UAVs in the PyBullet simulation environment."""
from __future__ import annotations

import time

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from .abstractions.base_drone import DroneClass
from .drones.fixedwing import FixedWing
from .drones.quadx import QuadX
from .drones.rocket import Rocket


class Aviary(bullet_client.BulletClient):
    """Aviary class, the core of how PyFlyt handles UAVs in the PyBullet simulation environment."""

    def __init__(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        drone_type: str | list[str],
        drone_type_mappings: None | dict[str, DroneClass] = None,
        render: bool = False,
        physics_hz: int = 240,
        ctrl_hz: int = 120,
        worldScale: float = 1.0,
        drone_options: dict | list[dict] = {},
        seed: None | int = None,
    ):
        """Initializes a PyBullet environment that hosts UAVs and other entities.

        The Aviary class itself inherits from a BulletClient, so any function that a PyBullet client has, this class will have.
        The Aviary also handles dealing with physics and control looprates, as well as automatic construction of several default UAVs and their corresponding cameras.

        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            drone_type (str | list[str]): drone_types
            drone_type_mappings (None | dict[str, DroneClass]): string to mapping of custom drone classes
            render (bool): render
            physics_hz (int): physics_hz
            ctrl_hz (int): ctrl_hz
            worldScale (float): worldScale
            drone_options (dict | list[dict]): drone_options
            seed (None | int): seed
        """
        super().__init__(p.GUI if render else p.DIRECT)
        print("\033[A                             \033[A")

        # assertations
        assert (
            len(start_pos.shape) == 2
        ), f"start_pos must be shape (n, 3), currently {start_pos.shape}."
        assert (
            start_pos.shape[-1] == 3
        ), f"start_pos must be shape (n, 3), currently {start_pos.shape}."
        assert (
            start_orn.shape == start_pos.shape
        ), f"start_orn must be same shape as start_pos, currently {start_orn.shape}."
        if isinstance(drone_type, list):
            assert (
                len(drone_type) == start_pos.shape[0]
            ), f"Must have same number of `drone_types` ({len(drone_type)}) as starting positions."

        # default physics looprate is 240 Hz
        # do not change because pybullet doesn't like it
        self.physics_hz = physics_hz
        self.physics_period = 1.0 / physics_hz
        self.ctrl_hz = ctrl_hz
        self.ctrl_period = 1.0 / ctrl_hz
        self.ctrl_update_ratio = int(physics_hz / ctrl_hz)
        self.now = time.time()

        # mapping of drone type string to the constructors
        self.drone_type_mappings = dict()
        self.drone_type_mappings["quadx"] = QuadX
        self.drone_type_mappings["fixedwing"] = FixedWing
        self.drone_type_mappings["rocket"] = Rocket
        if drone_type_mappings is not None:
            self.drone_type_mappings = {
                **self.drone_type_mappings,
                **drone_type_mappings,
            }

        # store all drone types
        self.drone_type = drone_type

        # pybullet stuff
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.worldScale = worldScale

        # directories and paths
        self.drone_options = drone_options
        self.setAdditionalSearchPath(pybullet_data.getDataPath())

        # render
        self.render = render
        self.rtf_debug_line = self.addUserDebugText(
            text="RTF here", textPosition=[0, 0, 0], textColorRGB=[1, 0, 0]
        )

        self.reset(seed)

    def reset(self, seed: None | int = None):
        """Resets the simulation.

        Args:
            seed (None | int): seed
        """
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.steps = 0

        # reset the camera position to a sane place
        self.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1],
        )

        # define new RNG
        self.np_random = np.random.RandomState(seed=seed)

        """ CONSTRUCT THE WORLD """
        self.planeId = self.loadURDF(
            "plane.urdf", useFixedBase=True, globalScaling=self.worldScale
        )

        # spawn drones
        self.drones: list[DroneClass] = []
        for start_pos, start_orn in zip(self.start_pos, self.start_orn):
            if isinstance(self.drone_type, list):
                # if we have a list of drone types, spawn them
                for dt in self.drone_type:
                    self.drones.append(
                        self.drone_type_mappings[dt](
                            self,
                            start_pos=start_pos,
                            start_orn=start_orn,
                            ctrl_hz=self.ctrl_hz,
                            physics_hz=self.physics_hz,
                            np_random=self.np_random,
                            **self.drone_options,
                        )
                    )
            else:
                # if we only have one drone type
                for _ in range(len(self.start_pos)):
                    self.drones.append(
                        self.drone_type_mappings[self.drone_type](
                            self,
                            start_pos=start_pos,
                            start_orn=start_orn,
                            ctrl_hz=self.ctrl_hz,
                            physics_hz=self.physics_hz,
                            np_random=self.np_random,
                            **self.drone_options,
                        )
                    )

        # arm everything
        self.register_all_new_bodies()
        self.set_armed(True)

    def register_all_new_bodies(self):
        """Registers all new bodies in the environment to be able to handle collisions later.

        Call this when there is an update in the number of bodies in the environment.
        """
        # collision array
        self.collision_array = np.zeros(
            (self.getNumBodies(), self.getNumBodies()), dtype=bool
        )

    @property
    def num_drones(self) -> int:
        """Returns the number of drones in the environment.

        Returns:
            int: number of drones in the environment
        """
        return len(self.drones)

    @property
    def states(self) -> np.ndarray:
        """Returns a list of states for all drones in the environment.

        Returns:
            np.ndarray: list of states
        """
        states = []
        for drone in self.drones:
            states.append(drone.state)

        states = np.stack(states, axis=0)

        return states

    @property
    def aux_states(self) -> np.ndarray:
        """Returns a list of auxiliary states for all drones in the environment.

        Returns:
            np.ndarray: list of auxiliary states
        """
        aux_states = []
        for drone in self.drones:
            aux_states.append(drone.aux_state)

        aux_states = np.stack(aux_states, axis=0)

        return aux_states

    def print_all_bodies(self):
        """Debugging function used to print out all bodies in the environment along with their IDs."""
        bodies = dict()
        for i in range(self.getNumBodies()):
            bodies[i] = self.getBodyInfo(i)[-1].decode("UTF-8")

        from pprint import pprint

        pprint(bodies)

    def set_armed(self, settings: int | bool | list[int | bool]):
        """Sets the arming state of each drone in the environment.

        Args:
            settings (int | bool | list[int | bool]): arm setting
        """
        if isinstance(settings, list):
            assert len(settings) == len(
                self.drones
            ), f"Expected {len(self.drones)} settings, got {len(settings)}."
            self.armed_drones = [
                drone for (drone, arm) in zip(self.drones, settings) if arm
            ]
        else:
            self.armed_drones = [drone for drone in self.drones] if settings else []

    def set_mode(self, flight_modes: int | list[int]):
        """Sets the flight mode of each drone in the environment.

        Args:
            flight_modes (int | list[int]): flight mode
        """
        if isinstance(flight_modes, list):
            assert len(flight_modes) == len(
                self.drones
            ), f"Expected {len(self.drones)} flight_modes, got {len(flight_modes)}."
            for drone, mode in zip(self.drones, flight_modes):
                drone.set_mode(mode)
        else:
            for drone in self.drones:
                drone.set_mode(flight_modes)

    def set_setpoints(self, setpoints: np.ndarray):
        """Sets the setpoints of each drone in the environment.

        Args:
            setpoints (np.ndarray): list of setpoints
        """
        for i, drone in enumerate(self.drones):
            drone.setpoint = setpoints[i]

    def step(self):
        """Steps the environment, this automatically handles physics and control looprates, one step is equivalent to one control loop step."""
        # compute rtf if we're rendering
        if self.render:
            elapsed = time.time() - self.now
            self.now = time.time()

            # sleep to maintain real time factor
            time.sleep(max(0, self.ctrl_period - elapsed))

            # calculate real time factor
            RTF = self.ctrl_period / (elapsed + 1e-6)

            # handle case where sometimes elapsed becomes 0
            if elapsed != 0.0:
                self.rtf_debug_line = self.addUserDebugText(
                    text=f"RTF: {str(RTF)[:7]}",
                    textPosition=[0, 0, 0],
                    textColorRGB=[1, 0, 0],
                    replaceItemUniqueId=self.rtf_debug_line,
                )

        # reset collisions
        self.collision_array &= False

        # update onboard avionics compute
        [drone.update_avionics() for drone in self.armed_drones]

        # step the environment enough times for one control loop
        for _ in range(self.ctrl_update_ratio):
            # compute physics
            [drone.update_physics() for drone in self.armed_drones]

            # advance pybullet
            self.stepSimulation()

            # splice out collisions
            for collision in self.getContactPoints():
                self.collision_array[collision[1], collision[2]] = True
                self.collision_array[collision[2], collision[1]] = True

        self.steps += 1
