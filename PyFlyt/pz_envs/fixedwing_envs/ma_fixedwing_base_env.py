"""Base Multiagent Fixedwing Environment."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from gymnasium import Space, spaces
from pettingzoo import ParallelEnv

from PyFlyt.core import Aviary
from PyFlyt.core.utils.compile_helpers import jitter


class MAFixedwingBaseEnv(ParallelEnv):
    """Base Dogfighting Environment for the Aggressor model using custom environment API."""

    metadata = dict(render_modes="human")

    def __init__(
        self,
        start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        assisted_flight: bool = True,
        flight_dome_size: float = 300.0,
        max_duration_seconds: float = 60.0,
        angle_representation: Literal["euler", "quaternion"] = "euler",
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            assisted_flight (bool): assisted_flight
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode

        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if render_mode is not None:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
        self.render_mode = render_mode is not None

        """SPACES"""
        # attitude size increases by 1 for quaternion
        if angle_representation == "euler":
            attitude_shape = 12
        elif angle_representation == "quaternion":
            attitude_shape = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        # action space
        high = np.ones(4 if assisted_flight else 6)
        low = high * -1.0
        self._action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # observation space
        self.auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                attitude_shape
                + self.auxiliary_space.shape[0]
                + self.action_space(None).shape[0],
            ),
        )

        """CONSTANTS"""
        # check the start_pos shapes
        assert (
            len(start_pos.shape) == 2
        ), f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        assert (
            start_pos.shape[-1] == 3
        ), f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        assert (
            start_pos.shape == start_orn.shape
        ), f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        self.start_pos = start_pos
        self.start_orn = start_orn

        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

        # select agents
        self.num_possible_agents = len(start_pos)
        self.possible_agents = [
            "uav_" + str(r) for r in range(self.num_possible_agents)
        ]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        """RUNTIME PARAMETERS"""
        self.current_actions = np.zeros(
            (
                self.num_possible_agents,
                *self.action_space(None).shape,
            ),
            dtype=np.float64,
        )
        self.past_actions = np.zeros(
            (
                self.num_possible_agents,
                *self.action_space(None).shape,
            ),
            dtype=np.float64,
        )

    def observation_space(self, agent: Any = None) -> Space:
        """observation_space.

        Returns:
            Space:

        """
        raise NotImplementedError

    def action_space(self, agent: Any = None) -> spaces.Box:
        """action_space.

        Returns:
            spaces.Box:

        """
        return self._action_space

    def close(self) -> None:
        """Close."""
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

    def reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """reset.

        Args:
            seed (None | int): seed
            options (dict | None): options

        Returns:
            tuple[dict[str, Any], dict[str, Any]]:

        """
        raise NotImplementedError

    def begin_reset(
        self,
        seed: None | int = None,
        options: None | dict[str, Any] = dict(),
        drone_options: None | dict[str, Any] | Sequence[dict[str, Any]] = dict(),
    ) -> None:
        """The first half of the reset function."""
        # if we already have an env, disconnect from it
        if hasattr(self, "aviary"):
            self.aviary.disconnect()
        self.step_count = 0
        self.agents = self.possible_agents[:]

        # handle drone options
        if drone_options is None:
            drone_options = [dict() for _ in range(self.num_possible_agents)]
        elif isinstance(drone_options, dict):
            drone_options = [drone_options for _ in range(self.num_possible_agents)]

        # set the model name
        for i in range(len(drone_options)):
            drone_options[i]["drone_model"] = (
                drone_options[i].get("drone_model") or "acrowing"
            )

        # if render, use onboard camera for the first aircraft
        if self.render_mode:
            drone_options[0]["use_camera"] = True
            drone_options[0]["camera_fps"] = int(120 / self.env_step_ratio)

        # rebuild the environment
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="fixedwing",
            render=bool(self.render_mode),
            drone_options=drone_options,
            seed=seed,
            world_scale=5.0,
        )

        # reset the camera position to a sane place
        self.aviary.resetDebugVisualizerCamera(
            cameraDistance=50,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1],
        )

    def end_reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> None:
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.aviary.register_all_new_bodies()

        # set flight mode
        self.aviary.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.aviary.step()
        self.update_states()

    def update_states(self) -> None:
        """Helper method to be called once after the internal aviary has stepped.

        Args:
            agent_id (int): agent_id

        Returns:
            Any:

        """
        pass

    def pop_obs_by_id(self, agent_id: int) -> Any:
        """Pops an observation at the `agent_id`.

        This will be called once per agent per RL step
        Feel free to reset values for the called `agent_id`.

        Args:
            agent_id (int): agent_id

        Returns:
            Any:

        """
        raise NotImplementedError

    def pop_term_trunc_rew_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Pops a term, trunc, reward, info at the `agent_id`.

        This will be called once per agent per RL step
        Feel free to reset values for the called `agent_id`.

        Args:
            agent_id (int): agent_id

        Returns:
            Tuple[bool, bool, float, dict[str, Any]]:

        """
        raise NotImplementedError

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """step.

        Args:
            actions (dict[str, np.ndarray]): actions

        Returns:
            tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:

        """
        # copy over the past actions
        self.past_actions = self.current_actions.copy()

        # set the new actions and send to aviary
        # this automatically sets terminated agent actions to 0
        self.current_actions = np.zeros_like(self.current_actions)
        for k, v in actions.items():
            if k in self.agents:
                self.current_actions[self.agent_name_mapping[k]] = v

        # pass things to the aviary, but clip throttle
        aviary_action = self.current_actions.copy()
        aviary_action[..., -1] = (aviary_action[..., -1] / 2.0) + 0.5
        self.aviary.set_all_setpoints(aviary_action)

        # step enough times for one RL step
        for _ in range(self.env_step_ratio):
            self.aviary.step()
            self.update_states()

        # observation and rewards dictionary
        obs = dict()
        term = dict()
        trunc = dict()
        rew = dict()
        infos = dict()

        # update reward, term, trunc, for each agent
        for ag in self.agents:
            ag_id = self.agent_name_mapping[ag]

            # compute observations reward term trunc info
            obs[ag] = self.pop_obs_by_id(ag_id)
            term[ag], trunc[ag], rew[ag], infos[ag] = (
                self.pop_term_trunc_rew_info_by_id(ag_id)
            )

        # increment step count and cull dead agents for the next round
        self.step_count += 1
        self.agents = [
            agent for agent in self.agents if not (term[agent] or trunc[agent])
        ]
        return obs, rew, term, trunc, infos

    @staticmethod
    def compute_rotation_forward(orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the rotation matrix and forward vector of an aircraft given its orientation.

        Args:
            orn (np.ndarray): an [n, 3] array of each drone's orientation

        Returns:
            np.ndarray: an [n, 3, 3] rotation matrix of each aircraft
            np.ndarray: an [n, 3] forward vector of each aircraft

        """
        # use the jitted component to generate all the memory intensive copies
        rx, ry, rz, forward_vector = (
            MAFixedwingBaseEnv._jitted_compute_unit_rotation_forward(orn)
        )

        # order of operations for multiplication matters here
        return rz @ ry @ rx, forward_vector

    @staticmethod
    @jitter
    def _jitted_compute_unit_rotation_forward(orn: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Jitted unit to compute rotation matrices and forward vectors.

        Args:
            orn (np.ndarray): orn

        Returns:
            tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ]:
        """
        # some general stuff
        c, s = np.cos(orn), np.sin(orn)
        eye = np.zeros((orn.shape[0], 3, 3), dtype=np.float64)
        eye[:] = np.eye(3)

        # create the rotation matrix
        rx = eye.copy()
        rx[:, 1, 1] = c[..., 0]
        rx[:, 1, 2] = -s[..., 0]
        rx[:, 2, 1] = s[..., 0]
        rx[:, 2, 2] = c[..., 0]
        ry = eye.copy()
        ry[:, 0, 0] = c[..., 1]
        ry[:, 0, 2] = s[..., 1]
        ry[:, 2, 0] = -s[..., 1]
        ry[:, 2, 2] = c[..., 1]
        rz = eye.copy()
        rz[:, 0, 0] = c[..., 2]
        rz[:, 0, 1] = -s[..., 2]
        rz[:, 1, 0] = s[..., 2]
        rz[:, 1, 1] = c[..., 2]

        # compute forward vector
        forward_vector = np.stack(
            (c[..., 2] * c[..., 1], s[..., 2] * c[..., 1], -s[..., 1]), axis=-1
        )

        return rx, ry, rz, forward_vector
