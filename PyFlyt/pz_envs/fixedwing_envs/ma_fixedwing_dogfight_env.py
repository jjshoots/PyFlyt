"""Multiagent Fixedwing Dogfighting Environment."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from PyFlyt.pz_envs.fixedwing_envs.ma_fixedwing_base_env import MAFixedwingBaseEnv


class MAFixedwingDogfightEnv(MAFixedwingBaseEnv):
    """Base Dogfighting Environment for the Acrowing model using the PettingZoo API.

    Args:
        spawn_height (float): how high to spawn the agents at the beginning of the simulation.
        damage_per_hit (float): how much damage per hit per physics step, each agent starts with a health of 1.0.
        lethal_distance (float): how close before weapons become effective.
        lethal_angle_radians (float): the width of the cone of fire.
        assisted_flight (bool): whether to use high level commands (RPYT) instead of full actuator commands.
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | str): can be "human" or None
    """

    metadata = {
        "render_modes": ["human"],
        "name": "ma_quadx_hover",
    }

    def __init__(
        self,
        spawn_height: float = 15.0,
        damage_per_hit: float = 0.02,
        lethal_distance: float = 15.0,
        lethal_angle_radians: float = 0.1,
        assisted_flight: bool = True,
        sparse_reward: bool = False,
        flight_dome_size: float = 150.0,
        max_duration_seconds: float = 60.0,
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            spawn_height (float): how high to spawn the agents at the beginning of the simulation.
            damage_per_hit (float): how much damage per hit per physics step, each agent starts with a health of 1.0.
            lethal_distance (float): how close before weapons become effective.
            lethal_angle_radians (float): the width of the cone of fire.
            assisted_flight (bool): whether to use high level commands (RPYT) instead of full actuator commands.
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | str): can be "human" or None
        """
        # placeholder starting positions
        super().__init__(
            start_pos=np.array([[10.0, 0.0, 10.0], [-10.0, 0.0, 10.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation="euler",
            agent_hz=agent_hz,
            assisted_flight=assisted_flight,
            render_mode=render_mode,
        )

        self.sparse_reward = sparse_reward
        self.damage_per_hit = damage_per_hit
        self.spawn_height = spawn_height
        self.lethal_distance = lethal_distance
        self.lethal_angle = lethal_angle_radians

        # observation_space
        # combined (state + aux) + health + enemy state
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + 1 + 12,),
        )

    def observation_space(self, agent: Any = None) -> spaces.Box:
        """observation_space.

        Args:
            agent:

        Returns:
            spaces.Box:
        """
        return self._observation_space

    def _get_start_pos_orn(self, seed: None | int) -> tuple[np.ndarray, np.ndarray]:
        """_get_start_pos_orn.

        Args:

        Returns:
            tuple[np.ndarray, np.ndarray]:
        """
        np_random = np.random.RandomState(seed=seed)
        start_pos = np.zeros((2, 3))
        while np.linalg.norm(start_pos[0] - start_pos[1]) < self.flight_dome_size * 0.2:
            start_pos = (np_random.rand(2, 3) - 0.5) * self.flight_dome_size * 0.5
            start_pos[:, -1] = self.spawn_height
        start_orn = (np_random.rand(2, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])

        return start_pos, start_orn

    def reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """reset.

        Args:
            seed (None | int): seed
            options (dict[str, Any]): options

        Returns:
            tuple[dict[str, Any], dict[str, Any]]:
        """
        self.start_pos, self.start_orn = self._get_start_pos_orn(seed)

        # define custom forward velocity
        _, start_vec = self.compute_rotation_forward(self.start_orn)
        start_vec *= 10.0
        drone_options = [dict(), dict()]
        for i in range(len(drone_options)):
            drone_options[i]["starting_velocity"] = start_vec[i]

        super().begin_reset(seed, options, drone_options=drone_options)

        # reset runtime parameters
        self.health = np.ones(2, dtype=np.float64)
        self.in_cone = np.zeros((2,), dtype=bool)
        self.in_range = np.zeros((2,), dtype=bool)
        self.chasing = np.zeros((2,), dtype=bool)
        self.current_hits = np.zeros((2), dtype=bool)
        self.current_angles = np.zeros(2, dtype=np.float64)
        self.current_offsets = np.zeros(2, dtype=np.float64)
        self.current_distance = np.zeros(2, dtype=np.float64)
        self.previous_hits = np.zeros((2), dtype=bool)
        self.previous_angles = np.zeros(2, dtype=np.float64)
        self.previous_offsets = np.zeros(2, dtype=np.float64)
        self.previous_distance = np.zeros(2, dtype=np.float64)
        self.observations = np.zeros((2, *self.observation_space(0).shape))
        self.last_obs_time = -1.0
        self.rewards = np.zeros((2,), dtype=np.float64)
        self.last_rew_time = -1.0

        super().end_reset(seed, options)

        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        }
        infos = {ag: dict() for ag in self.agents}
        return observations, infos

    def _compute_agent_states(self) -> None:
        """_compute_agent_states.

        Args:

        Returns:
            None:
        """
        # get the states of both drones
        self.attitudes = np.stack(self.aviary.all_states, axis=0)

        """COMPUTE HITS"""
        # get the rotation matrices and forward vectors
        # offset the position to be on top of the main wing
        rotation, forward_vecs = self.compute_rotation_forward(self.attitudes[:, 1])
        self.attitudes[:, -1] = self.attitudes[:, -1] - (forward_vecs * 0.35)  # pyright: ignore[reportGeneralTypeIssues]

        # compute the vectors of each drone to each drone
        separation = self.attitudes[::-1, -1] - self.attitudes[:, -1]  # pyright: ignore[reportGeneralTypeIssues]
        self.previous_distance = self.current_distance.copy()
        self.current_distance = np.linalg.norm(separation[0])

        # compute engagement angles
        self.previous_angles = self.current_angles.copy()
        self.current_angles = np.arccos(
            np.sum(separation * forward_vecs, axis=-1) / self.current_distance
        )

        # compute engagement offsets
        self.previous_offsets = self.current_offsets.copy()
        self.current_offsets = np.linalg.norm(
            np.cross(separation, forward_vecs), axis=-1
        )

        # whether we're lethal or chasing or have opponent in cone
        self.in_cone = self.current_angles < self.lethal_angle
        self.in_range = self.current_distance < self.lethal_distance
        self.chasing = np.abs(self.current_angles) < (np.pi / 2.0)

        # compute whether anyone hit anyone
        self.current_hits = self.in_cone & self.in_range & self.chasing

        # update health based on hits
        self.health -= self.damage_per_hit * self.current_hits[::-1]

        """COMPUTE THE STATE VECTOR"""
        # form the opponent state matrix
        opponent_attitudes = np.zeros_like(self.attitudes)

        # opponent angular rate is unchanged
        opponent_attitudes[:, 0] = self.attitudes[::-1, 0]

        # opponent angular position is relative to ours
        opponent_attitudes[:, 1] = self.attitudes[::-1, 1] - self.attitudes[:, 1]  # pyright: ignore[reportGeneralTypeIssues]

        # opponent velocity is relative to ours in our body frame
        ground_velocities: np.ndarray = (
            rotation @ np.expand_dims(self.attitudes[:, -2], axis=-1)
        ).reshape(2, 3)
        opponent_velocities = (
            np.expand_dims(ground_velocities, axis=1)[::-1] @ rotation
        ).reshape(2, 3)
        opponent_attitudes[:, 2] = opponent_velocities - self.attitudes[:, 2]  # pyright: ignore[reportGeneralTypeIssues]

        # opponent position is relative to ours in our body frame
        opponent_attitudes[:, 3] = (
            np.expand_dims(separation, axis=1) @ rotation
        ).reshape(2, 3)

        # flatten the attitude and opponent attitude, expand dim of health
        flat_attitude = self.attitudes.reshape(2, -1)
        flat_opponent_attitude = opponent_attitudes.reshape(2, -1)
        health = np.expand_dims(self.health, axis=-1)

        # form the state vector
        self.observations = np.concatenate(
            [
                flat_attitude,
                health,
                flat_opponent_attitude,
                health[::-1],
                self.past_actions,
            ],
            axis=-1,
        )

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        """compute_observation_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            np.ndarray:
        """
        # don't recompute if we've already done it
        if self.last_obs_time != self.aviary.elapsed_time:
            self.last_obs_time = self.aviary.elapsed_time
            self._compute_agent_states()
        return self.observations[agent_id]

    def _compute_engagement_rewards(self) -> None:
        """_compute_engagement_rewards."""
        # reset rewards
        self.rewards *= 0.0

        # sparse reward computation
        if not self.sparse_reward:
            # reward for closing the distance
            self.rewards += (
                np.clip(
                    self.previous_distance - self.current_distance,
                    a_min=0.0,
                    a_max=None,
                )
                * (~self.in_range & self.chasing)
                * 1.0
            )

            # reward for progressing to engagement
            self.rewards += (
                (self.previous_angles - self.current_angles) * self.in_range * 10.0
            )

            # reward for engaging the enemy
            self.rewards += 3.0 / (self.current_angles + 0.1) * self.in_range

        # reward for hits
        self.rewards += 30.0 * self.current_hits

        # penalty for being hit
        self.rewards -= 20.0 * self.current_hits[::-1]

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        term, trunc, info = super().compute_base_term_trunc_info_by_id(agent_id)

        # terminal if other agent is dead
        term |= self.num_agents < 2

        # don't recompute if we've already done it
        if self.last_rew_time != self.aviary.elapsed_time:
            self.last_rew_time = self.aviary.elapsed_time
            self._compute_engagement_rewards()

        reward = self.rewards[agent_id]
        reward -= bool(info.get("out_of_bounds")) * 3000.0
        reward -= bool(info.get("collision")) * 3000.0

        # all the info things
        info["wins"] = self.health <= 0.0
        info["healths"] = self.health

        return term, trunc, reward, info

    @staticmethod
    def compute_rotation_forward(orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the rotation matrix and forward vector of an aircraft given its orientation.

        Args:
            orn (np.ndarray): an [n, 3] array of each drone's orientation

        Returns:
            np.ndarray: an [n, 3, 3] rotation matrix of each aircraft
            np.ndarray: an [n, 3] forward vector of each aircraft
        """
        c, s = np.cos(orn), np.sin(orn)
        eye = np.stack([np.eye(3)] * orn.shape[0], axis=0)

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

        forward_vector = np.stack(
            [c[..., 2] * c[..., 1], s[..., 2] * c[..., 1], -s[..., 1]], axis=-1
        )

        # order of operations for multiplication matters here
        return rz @ ry @ rx, forward_vector

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
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
            tuple[
                dict[str, Any],
                dict[str, float],
                dict[str, bool],
                dict[str, bool],
                dict[str, dict[str, Any]],
            ]:
        """
        returns = super().step(actions=actions)

        # colour the gunsights conditionally
        if self.render_mode and not np.all(self.previous_hits == self.current_hits):
            self.previous_hits = self.current_hits.copy()
            hit_colour = np.array([1.0, 0.0, 0.0, 0.2])
            norm_colour = np.array([0.0, 0.0, 0.0, 0.2])
            for i in range(2):
                self.aviary.changeVisualShape(
                    self.aviary.drones[i].Id,
                    7,
                    rgbaColor=(hit_colour if self.current_hits[i] else norm_colour),
                )

        return returns
