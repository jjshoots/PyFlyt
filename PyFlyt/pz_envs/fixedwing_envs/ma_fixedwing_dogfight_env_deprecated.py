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
        "name": "ma_fixedwing_dogfight",
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
        self.hit_colour = np.array([1.0, 0.0, 0.0, 0.2])
        self.nohit_color = np.array([0.0, 0.0, 0.0, 0.2])

        # observation_space
        # combined (state + aux) + health + enemy state + enemy health
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + 1 + 12 + 1,),
        )

    def observation_space(self, agent: Any = None) -> spaces.Box:
        """observation_space.

        Args:
            agent (Any): agent

        Returns:
            spaces.Box:

        """
        return self._observation_space

    def _get_start_pos_orn(self, seed: None | int) -> tuple[np.ndarray, np.ndarray]:
        """_get_start_pos_orn.

        Args:
            seed (None | int): seed

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
        self.opponent_attitudes = np.zeros((2, 4, 3), dtype=np.float64)
        self.healths = np.ones(2, dtype=np.float64)
        self.opp_in_cone = np.zeros(2, dtype=bool)
        self.opp_in_range = np.zeros(2, dtype=bool)
        self.is_chasing = np.zeros(2, dtype=bool)
        self.current_hits = np.zeros(2, dtype=bool)
        self.current_angles = np.zeros(2, dtype=np.float64)
        self.current_offsets = np.zeros(2, dtype=np.float64)
        self.current_distance = np.zeros((), dtype=np.float64)
        self.previous_hits = np.zeros(2, dtype=bool)
        self.previous_angles = np.zeros(2, dtype=np.float64)
        self.previous_offsets = np.zeros(2, dtype=np.float64)
        self.previous_distance = np.zeros((), dtype=np.float64)
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

    def _compute_combat_state(self) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """compute_combat_state.

        Args:

        Returns:
            tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            - in_cone: an [n, n] array indicating which agents have which agents in the lethal cone.
            - in_range: an [n, n] array indicating which agents have which agents in range.
            - chasing: an [n, n] array indicating which are in pursuit of which agents.
            - current_hits: an [n, n] array indicating agent has shot which agent.
            - current_distances: an [n, n] array of distances between each agent and every other agent.
            - current_angles: an [n, n] array of engagement angles between each agent and every other agent.
            - current_offsets: an [n, n] array of engagement offsets between each agent and every other agent.
            - opponent_attitudes: an [n, n, 4, 3] of each agents states relative to self.
        """
        ############################################################################################
        # COMPUTE HITS
        ############################################################################################

        # get the rotation matrices and forward vectors
        # attitudes returns the position of the aircraft nose, shift it to the center of the body
        rotation, forward_vecs = self.compute_rotation_forward(self.attitudes[:, 1])
        self.attitudes[:, -1, :] = self.attitudes[:, -1, :] - (forward_vecs * 0.35)

        # separation here is a [self, other, 3] array of pointwise distance vectors
        separation = self.attitudes[None, :, -1, :] - self.attitudes[:, None, -1, :]

        # compute the vectors of each drone to each drone
        current_distances = np.linalg.norm(separation, axis=-1)

        # compute engagement angles
        # WARNING: this has NaNs on the diagonal, watch for everything downstream
        with np.errstate(divide="ignore", invalid="ignore"):
            current_angles = np.arccos(
                np.sum(separation * forward_vecs[:, None, :], axis=-1)
                / current_distances
            )

        # compute engagement offsets
        current_offsets = np.linalg.norm(np.cross(separation, forward_vecs), axis=-1)

        # whether we're lethal or chasing or have opponent in cone
        in_cone = current_angles < self.lethal_angle
        in_range = current_distances < self.lethal_distance
        chasing = np.abs(current_angles) < (np.pi / 2.0)

        # compute whether anyone hit anyone
        current_hits = in_cone & in_range & chasing

        ############################################################################################
        # COMPUTE STATES
        ############################################################################################

        # form the opponent state matrix
        # this is a [n, n, 4, 3] matrix since each agent needs to attend to every other agent
        opponent_attitudes = np.zeros(
            (self.attitudes.shape[0], *self.attitudes.shape), dtype=np.float64
        )

        # opponent angular rates are unchanged because already body frame
        opponent_attitudes[..., 0, :] = self.attitudes[:, 0, :]

        # opponent angular positions must convert to be relative to ours
        opponent_attitudes[..., 1, :] = (
            self.attitudes[None, :, 1] - self.attitudes[:, None, 1]
        )

        # rotate all velocities to be ground frame, this is [n, 3]
        ground_velocities = (rotation @ self.attitudes[:, -2, :][..., None])[..., 0]

        # then find all opponent velocities relative to our body frame
        # this is [self, other, 3]
        opponent_velocities = (
            ground_velocities[None, ..., None, :] @ rotation[:, None, ...]
        )[..., 0, :]

        # opponent velocities should be relative to our current velocity
        opponent_attitudes[..., 2, :] = (
            opponent_velocities - self.attitudes[:, 2, :][:, None, ...]
        )

        # opponent position is relative to ours in our body frame
        opponent_attitudes[..., 3, :] = separation @ rotation

        return (
            in_cone,
            in_range,
            chasing,
            current_hits,
            current_distances,
            current_angles,
            current_offsets,
            opponent_attitudes,
        )

    def _compute_agent_states(self) -> None:
        """_compute_agent_states.

        Returns:
            None:

        """
        # get the states of both drones
        self.attitudes = np.stack(self.aviary.all_states, axis=0)

        # record some past statistics
        self.previous_hits = self.current_hits.copy()
        self.previous_distance = self.current_distance.copy()
        self.previous_angles = self.current_angles.copy()
        self.previous_offsets = self.current_offsets.copy()

        # compute new states
        (
            in_cone,
            in_range,
            chasing,
            current_hits,
            current_distance,
            current_angles,
            current_offsets,
            self.opponent_attitudes,
        ) = self._compute_combat_state()

        # extract what we need
        idx = np.arange(2)
        self.opp_in_cone = in_cone[idx, idx[::-1]]
        self.opp_in_range = in_range[idx, idx[::-1]]
        self.is_chasing = chasing[idx, idx[::-1]]
        self.current_hits = current_hits.sum(axis=0) > 0.0
        self.current_distance = current_distance[0, 1]
        self.current_angles = current_angles[idx, idx[::-1]]
        self.current_offsets = current_offsets[idx[::-1], idx]

        # compute whether anyone hit anyone
        self.healths -= self.damage_per_hit * self.current_hits

        # flatten the attitude and opponent attitude
        flat_attitude = self.attitudes.reshape(2, -1)
        flat_opponent_attitude = self.opponent_attitudes[idx, idx[::-1]].reshape(2, -1)

        # form the state vector
        self.observations = np.concatenate(
            [
                flat_attitude,
                self.aviary.all_aux_states,
                self.healths[..., None],
                flat_opponent_attitude,
                self.healths[..., None][::-1],
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
                * (~self.opp_in_range & self.is_chasing)
                * 1.0
            )

            # reward for progressing to engagement
            self.rewards += (
                (self.previous_angles - self.current_angles) * self.opp_in_range * 10.0
            )

            # reward for engaging the enemy
            self.rewards += 3.0 / (self.current_angles + 0.1) * self.opp_in_range

        # reward for hits
        self.rewards += 30.0 * self.current_hits

        # penalty for being hit
        self.rewards -= 20.0 * self.current_hits[::-1]

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        # don't recompute if we've already done it
        if self.last_rew_time != self.aviary.elapsed_time:
            self.last_rew_time = self.aviary.elapsed_time
            self._compute_engagement_rewards()

        # initialize
        reward = self.rewards[agent_id]
        term = self.num_agents < 2
        trunc = self.step_count > self.max_steps
        info = dict()

        # collision
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id]):
            reward -= 3000.0
            info["collision"] = True
            term |= True

        # exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 3000.0
            info["out_of_bounds"] = True
            term |= True

        # out of health
        if self.healths[agent_id] <= 0.0:
            reward -= 100.0
            info["dead"] = True
            term |= True

        # all the info things
        info["healths"] = self.healths

        return term, trunc, reward, info

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
            for i in range(2):
                self.aviary.changeVisualShape(
                    self.aviary.drones[i].Id,
                    7,
                    rgbaColor=(
                        self.hit_colour if self.current_hits[i] else self.nohit_color
                    ),
                )

        return returns
