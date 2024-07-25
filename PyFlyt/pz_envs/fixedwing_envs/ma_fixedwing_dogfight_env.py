"""Multiagent Fixedwing Dogfighting Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.pz_envs.fixedwing_envs.ma_fixedwing_base_env import MAFixedwingBaseEnv


class MAFixedwingDogfightEnv(MAFixedwingBaseEnv):
    """Team Dogfighting Environment for the Acrowing model using the PettingZoo API.

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
        "name": "ma_fixedwing_team_dogfight",
    }

    def __init__(
        self,
        team_size: int = 2,
        spawn_radius: float = 10.0,
        spawn_height: float = 20.0,
        damage_per_hit: float = 0.01,
        lethal_distance: float = 20.0,
        lethal_angle_radians: float = 0.07,
        assisted_flight: bool = True,
        sparse_reward: bool = False,
        flatten_observation: bool = True,
        flight_dome_size: float = 500.0,
        max_duration_seconds: float = 60.0,
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            team_size (int): number of planes that comprises a team.
            spawn_radius (float): agents are spawned in a circle pointing outwards, this value is the radius of that circle.
            spawn_height (float): how high to spawn the agents at the beginning of the simulation.
            damage_per_hit (float): how much damage per hit per physics step, each agent starts with a health of 1.0.
            lethal_distance (float): how close before weapons become effective.
            lethal_angle_radians (float): the width of the cone of fire.
            assisted_flight (bool): whether to use high level commands (RPYT) instead of full actuator commands.
            sparse_reward (bool): whether to use sparse rewards or not.
            flatten_observation (bool): if False, this returns a Dict style observation.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | str): can be "human" or None

        """
        # placeholder starting positions
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 0.0]] * 2 * team_size),
            start_orn=np.array([[0.0, 0.0, 0.0]] * 2 * team_size),
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation="euler",
            agent_hz=agent_hz,
            assisted_flight=assisted_flight,
            render_mode=render_mode,
        )

        # some environment constants
        self.team_size = team_size
        self.spawn_radius = spawn_radius
        self.sparse_reward = sparse_reward
        self.flatten_observation = flatten_observation
        self.damage_per_hit = damage_per_hit
        self.spawn_height = spawn_height
        self.lethal_distance = lethal_distance
        self.lethal_angle = lethal_angle_radians
        self.team_flag = np.concatenate(
            (
                np.zeros((team_size,), dtype=bool),
                np.ones((team_size,), dtype=bool),
            ),
            axis=-1,
        )

        # the mask for friendly fire
        base_mask = np.ones((team_size, team_size), dtype=bool)
        self.friendly_fire_mask = np.concatenate(
            (
                np.concatenate((~base_mask, base_mask), axis=0),
                np.concatenate((base_mask, ~base_mask), axis=0),
            ),
            axis=1,
        )

        # observation_space
        self._observation_space = spaces.Dict(
            {
                # base + health
                "self": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self_space_shape := (self.combined_space.shape[0] + 1),),
                    dtype=np.float64,
                ),
                # attitude + health + team_mask
                "others": spaces.Sequence(
                    space=spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(others_space_shape := (12 + 1 + 1),),
                        dtype=np.float64,
                    ),
                    stack=True,
                ),
            }
        )

        # if flatten observation, then it's just a big box
        if flatten_observation:
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self_space_shape + (2 * team_size - 1) * others_space_shape,),
                dtype=np.float64,
            )

        # some rendering constants
        self.hit_colour = np.array([1.0, 0.0, 0.0, 0.2])
        self.nohit_color = np.array([0.0, 0.0, 0.0, 0.025])

    def observation_space(self, agent: Any = None) -> spaces.Dict | spaces.Box:
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
        # seed the RNG
        np_random = np.random.RandomState(seed=seed)

        # start out pointing in outward directions equally spaced
        start_z_radian = np.pi / self.team_size * np.arange(self.team_size * 2)

        # define the starting positions
        start_pos = np.zeros((self.num_possible_agents, 3))
        start_pos[:, 0] = self.spawn_radius * np.cos(start_z_radian)
        start_pos[:, 1] = self.spawn_radius * np.sin(start_z_radian)
        start_pos[:, 2] = (
            self.spawn_height + np_random.random(self.num_possible_agents) * 10.0
        )

        # define the starting orientations
        start_orn = np.zeros((self.team_size * 2, 3))
        start_orn[:, 2] = (
            start_z_radian + np_random.random(self.num_possible_agents) * np.pi / 8.0
        )

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
        _, start_velocity = self.compute_rotation_forward(self.start_orn)
        start_velocity *= 20.0
        drone_options = [dict() for _ in range(self.num_possible_agents)]
        for agent_id in range(len(drone_options)):
            drone_options[agent_id]["starting_velocity"] = start_velocity[agent_id]
            drone_options[agent_id]["drone_model"] = "acrowing"

        super().begin_reset(seed, options, drone_options=drone_options)

        # store all drone ids
        self.drone_ids = np.array([drone.Id for drone in self.aviary.drones])

        # inactive is a flag for when an aircraft has hit the ground and has 0 velocity
        self.inactive = np.zeros(self.num_possible_agents, dtype=bool)

        # attitudes and health of all drones
        self.healths = np.ones(self.num_possible_agents, dtype=np.float32)
        self.attitudes = np.zeros(
            (self.num_possible_agents, self.num_possible_agents, 4, 3), dtype=np.float32
        )
        self.other_attitudes = np.zeros(
            (self.num_possible_agents, self.num_possible_agents, 4, 3), dtype=np.float32
        )

        # engagement status
        self.in_cone = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )
        self.in_range = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )
        self.chasing = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )

        # engagement state
        self.current_hits = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )
        self.current_angles = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float32
        )
        self.current_offsets = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float32
        )
        self.current_distances = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float32
        )

        # past engagement state
        self.previous_hits = self.current_hits.copy()
        self.previous_angles = self.current_angles.copy()
        self.previous_offsets = self.current_offsets.copy()
        self.previous_distances = self.current_distances.copy()

        # accumulation for rew, term, trunc, info
        self.accumulated_rewards = np.zeros(
            (self.num_possible_agents,), dtype=np.float32
        )
        self.accumulated_terminations = np.zeros(
            (self.num_possible_agents,), dtype=bool
        )
        self.accumulated_truncations = np.zeros((self.num_possible_agents,), dtype=bool)
        self.accumulated_infos = [dict() for _ in range(self.num_possible_agents)]

        # some tracking statistics
        self.received_hits = np.zeros((self.num_possible_agents,), dtype=np.int32)

        # for rendering
        self.current_render_hits = np.zeros((self.num_possible_agents,), dtype=bool)
        self.previous_render_hits = np.zeros((self.num_possible_agents,), dtype=bool)

        # if we're rendering, set the colors of the wingtips and tail components
        if self.render_mode:
            for agent_id in range(self.num_possible_agents):
                # wingtips and tail component IDs
                for component_id in [1, 2, 3, 4]:
                    self.aviary.changeVisualShape(
                        self.aviary.drones[agent_id].Id,
                        component_id,
                        rgbaColor=(
                            np.array([1.0, 0.0, 0.0, 1.0])
                            if self.team_flag[agent_id]
                            else np.array([0.0, 0.0, 1.0, 1.0])
                        ),
                    )

        super().end_reset(seed, options)

        # initialize the observations and infos
        observations = {
            ag: self.pop_obs_by_id(self.agent_name_mapping[ag]) for ag in self.agents
        }
        infos = {ag: dict() for ag in self.agents}
        return observations, infos

    def update_states(self) -> None:
        """Updates all states and rewards after the aviary has stepped.

        Args:

        Returns:
            None:
        """
        self._compute_observation()
        self._compute_term_trunc_rew_info()

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

    def _compute_observation(self) -> None:
        """_compute_observation.

        Returns:
            None:

        """
        # get the states of both drones
        self.attitudes = np.stack(self.aviary.all_states, axis=0)

        # record some past statistics
        self.previous_hits = self.current_hits.copy()
        self.previous_distances = self.current_distances.copy()
        self.previous_angles = self.current_angles.copy()
        self.previous_offsets = self.current_offsets.copy()

        # compute new states
        (
            self.in_cone,
            self.in_range,
            self.chasing,
            self.current_hits,
            self.current_distances,
            self.current_angles,
            self.current_offsets,
            self.other_attitudes,
        ) = self._compute_combat_state()

        # mask out friendly fire - we don't shoot our friends
        self.current_hits &= self.friendly_fire_mask

        # compute whether anyone hit anyone
        received_hits_per_agent = self.current_hits.sum(axis=0)
        self.received_hits += received_hits_per_agent
        self.healths -= self.damage_per_hit * received_hits_per_agent
        self.healths[self.healths < 0.0] = 0.0

        # deactivate aircraft if they're dead, have heights close to the ground and have velocity less than 0.1
        self.inactive = (
            (self.healths <= 0.0)
            & (self.attitudes[:, -1, -1] < 2.0)
            & (np.linalg.norm(self.attitudes[:, -2, :], axis=-1) < 0.1)
        )

        # flatten things
        flat_attitudes = self.attitudes.reshape(self.num_possible_agents, -1)
        flat_healths = self.healths.reshape(self.num_possible_agents, -1)
        flat_other_attitudes = self.other_attitudes.reshape(
            self.num_possible_agents, self.num_possible_agents, -1
        )

        # start stacking the observations
        self.observations: list[dict[Literal["self", "others"], np.ndarray]] = []
        for i in range(self.num_possible_agents):
            # flag for still active aircraft since we want to eliminate redundant observations
            relevant = np.ones(self.num_possible_agents, dtype=bool)
            relevant[i] = False
            relevant &= ~self.inactive

            # build the observation
            self.observations.append(
                {
                    # formulate the observation of self
                    "self": np.concatenate(
                        (
                            flat_attitudes[i],
                            self.aviary.aux_state(i),
                            flat_healths[i],
                            self.past_actions[i],
                        ),
                        axis=-1,
                    ),
                    # formulate the others status: select perspective, then pick out if active
                    "others": np.concatenate(
                        (
                            flat_other_attitudes[i][relevant, ...],
                            flat_healths[relevant, ...],
                            (self.team_flag[relevant, None] == self.team_flag[i]),
                        ),
                        axis=-1,
                    ),
                }
            )

    def _compute_engagement_rewards(self) -> np.ndarray:
        """_compute_engagement_rewards.

        Args:

        Returns:
            np.ndarray: [n, ] array of rewards for engagement.

        """
        # init engagement rewards
        engagement_rewards = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float32
        )

        # sparse reward computation
        if not self.sparse_reward:
            # reward for closing the distance
            engagement_rewards += (
                np.clip(
                    self.previous_distances - self.current_distances,
                    a_min=0.0,
                    a_max=None,
                )
                * (~self.in_range & self.chasing)
                * 1.0
            )

            # reward for progressing to engagement, penalty for losing angles is less
            # WARNING: NaN introduced here
            delta_angles = self.previous_angles - self.current_angles
            delta_angles[delta_angles > 0.0] *= 0.8
            engagement_rewards += delta_angles * self.in_range * 7.0

            # reward for engaging the enemy
            # WARNING: NaN introduced here
            engagement_rewards += 3.0 / (self.current_angles + 0.1) * self.in_range

        # reward for hits and being hit, more reward for hits so the agents are less self preserving
        hits_rewards = (12.0 * self.current_hits) + (-8.0 * self.current_hits.T)
        engagement_rewards += 1.0 * hits_rewards

        # remove the nans, and sum rewards along axes
        np.fill_diagonal(engagement_rewards, 0.0)
        engagement_rewards = engagement_rewards.sum(axis=1)

        # team-based rewards
        # reward for your friends hitting enemies, and penalty otherwise
        # we get 0.5x what our teammates receive
        engagement_rewards[self.team_flag] += (
            0.5 * (hits_rewards * self.team_flag[:, None]).sum()
        )
        engagement_rewards[~self.team_flag] += (
            0.5 * (hits_rewards * ~self.team_flag[:, None]).sum()
        )

        return engagement_rewards

    def _compute_term_trunc_rew_info(self) -> None:
        """_compute_term_trunc_rew_info.

        Args:

        Returns:
            None:
        """
        # accumulate engagement rewards and handle truncation
        self.accumulated_rewards += self._compute_engagement_rewards()
        self.accumulated_truncations |= self.step_count > self.max_steps

        # out of health
        zero_healths = self.healths <= 0.0
        self.accumulated_terminations |= zero_healths

        # collision, override reward, not add
        collisions = self.aviary.contact_array[self.drone_ids].sum(axis=-1) > 0
        self.accumulated_terminations |= collisions
        self.accumulated_rewards[collisions] = -1500.0
        self.healths[collisions] = 0.0

        # exceed flight dome, override reward, not add
        out_of_bounds = (
            np.linalg.norm(
                np.stack(
                    [s[-1] for s in self.aviary.all_states],
                    axis=0,
                ),
                axis=-1,
            )
            > self.flight_dome_size
        )
        self.accumulated_terminations |= out_of_bounds
        self.accumulated_rewards[out_of_bounds] = -1500.0
        self.healths[out_of_bounds] = 0.0

        # all opponents deactivated, override reward, not add
        # this is hardcoded to have 2 teams by default
        team_wins = np.zeros_like(self.team_flag, dtype=bool)
        for team in [True, False]:
            team_wins[self.team_flag == team] = (
                self.healths[self.team_flag != team] <= 0.0
            ) & np.any(self.healths[self.team_flag == team] > 0.0)
        self.accumulated_terminations |= team_wins
        self.accumulated_rewards[team_wins] = 300.0

        # splice out infos
        for (
            info,
            health,
            received_hits,
            no_hp,
            coll,
            oob,
            team_win,
        ) in zip(
            self.accumulated_infos,
            self.healths,
            self.received_hits,
            zero_healths,
            collisions,
            out_of_bounds,
            team_wins,
        ):
            info.update({"health": health})
            info.update({"received_hits": received_hits})
            if no_hp:
                info.update({"dead": True})
            if coll:
                info.update({"collision": True})
            if oob:
                info.update({"out_of_bounds": True})
            if team_win:
                info.update({"team_win": True})

    def pop_obs_by_id(
        self, agent_id: int
    ) -> dict[Literal["self", "others"], np.ndarray] | np.ndarray:
        """pop_obs_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            np.ndarray:

        """
        if not self.flatten_observation:
            return self.observations[agent_id]
        else:
            flat_observation = np.concatenate(
                (
                    self.observations[agent_id]["self"],
                    self.observations[agent_id]["others"].flatten(),
                ),
                axis=-1,
            )
            flat_observation = np.concatenate(
                (
                    flat_observation,
                    np.zeros(
                        self._observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript] # fmt: skip
                        - flat_observation.shape[0],
                    ),
                )
            )
            return flat_observation

    def pop_term_trunc_rew_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Pops the termination, truncation, and reward of the current timestep."""
        reward = self.accumulated_rewards[agent_id]
        info = self.accumulated_infos[agent_id]

        # reset reward and info since this is a pop operation
        self.accumulated_rewards[agent_id] = 0.0
        self.accumulated_infos[agent_id] = dict()

        return (
            self.accumulated_terminations[agent_id],
            self.accumulated_truncations[agent_id],
            reward,
            info,
        )

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
        if self.render_mode:
            self.previous_render_hits = self.current_render_hits.copy()
            self.current_render_hits = self.current_hits.sum(axis=1) > 0
            for i in np.nonzero(
                self.current_render_hits != self.previous_render_hits,
            )[0]:
                self.aviary.changeVisualShape(
                    self.aviary.drones[i].Id,
                    7,
                    rgbaColor=(
                        self.hit_colour
                        if self.current_render_hits[i]
                        else self.nohit_color
                    ),
                )

        return returns
