from copy import deepcopy

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from PyFlyt.core import Aviary


def make_ma_hover_env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = MAQuadXHoverEnv(render_mode=render_mode)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class MAQuadXHoverEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "ma_quadx_hover"}

    def __init__(self, render_mode=None):
        self._start_pos = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        self._start_orn = np.zeros_like(self._start_pos)
        num_agents = len(self._start_pos)
        self.step_limit = 300

        self.render_mode = render_mode is not None
        self.aviary = Aviary(
            start_pos=self._start_pos,
            start_orn=self._start_orn,
            drone_type="quadx",
            render=self.render_mode,
        )
        # self.aviary.set_mode(6)

        # optional: a mapping between agent name and ID
        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # we define the action space
        high = np.array(
            [
                3.0,
                3.0,
                3.0,
                1.0,
            ]
        )
        low = np.array(
            [
                -3.0,
                -3.0,
                -3.0,
                0.0,
            ]
        )
        self._action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # we define the observation space
        auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                # base 12 states
                12
                # target positions
                + 3
                + self.action_space(None).shape[0]  # pyright: ignore
                + auxiliary_space.shape[0],  # pyright: ignore
            ),
            dtype=np.float64,
        )

        # runtime parameters
        self._actions = np.zeros(
            (num_agents, *self.action_space(None).shape)
        )  # pyright: ignore
        self._past_actions = np.zeros(
            (num_agents, *self.action_space(None).shape)
        )  # pyright: ignore

        self.render_mode = render_mode

    def observation_space(self, _):
        return self._observation_space

    def action_space(self, _):
        return self._action_space

    def observe(self, agent):
        agent_id = self.agent_name_mapping[agent]
        raw_state = self.aviary.state(agent_id)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]
        ang_vel, ang_pos, lin_vel, lin_pos = raw_state

        # combine everything
        return np.array(
            [
                *ang_vel,
                *ang_pos,
                *lin_vel,
                *lin_pos,
                *self._start_pos[agent_id],
                *self._past_actions[agent_id],
                *self.aviary.aux_state(0),
            ]
        )

    def close(self):
        self.aviary.disconnect()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Our agent_selector utility allows easy cyclic stepping through the agents list.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # disconnect and rebuilt the environment
        self.aviary.disconnect()
        self.aviary = Aviary(
            start_pos=self._start_pos,
            start_orn=self._start_orn,
            drone_type="quadx",
            render=bool(self.render_mode),
            seed=seed,
        )
        # self.aviary.set_mode(6)

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._actions[self.agent_name_mapping[agent]] = action
        self._cumulative_rewards[agent] = 0

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # send the action to the aviary and update
            self.aviary.set_all_setpoints(self._actions)
            self.aviary.step()
            self._past_actions = deepcopy(self._actions)

            # rewards for all agents are placed in the .rewards dictionary
            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]
                linear_distance = np.linalg.norm(
                    self.aviary.state(ag_id)[-1] - self._start_pos[ag_id]
                )

                # how far are we from 0 roll pitch
                angular_distance = np.linalg.norm(self.aviary.state(ag_id)[1][:2])

                # add reward to s_agent
                self.rewards[ag] = -float(linear_distance + angular_distance)

                # truncations and terminations
                self.truncations[ag] = bool(
                    self.aviary.elapsed_time > self.step_limit
                )
                self.terminations[ag] = bool(self.aviary.contact_array[ag_id].sum() > 0)
        else:
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


if __name__ == "__main__":
    env = make_ma_hover_env(render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        print(reward)

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            # action = env.action_space(agent).sample()
            action = np.array([0.0, 0.0, 0.0, 0.0])

        env.step(action)
    env.close()
