"""Enhanced PettingZoo environment wrapper with parallel environment support."""

from typing import Any, Dict, List, Literal, Optional, Union
import warnings

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv, ParallelEnv
from pettingzoo.utils.wrappers import BaseWrapper

from tianshou.env.pettingzoo_env import PettingZooEnv


class EnhancedPettingZooEnv(PettingZooEnv):
    """Enhanced wrapper supporting both AEC and Parallel PettingZoo environments.

    This wrapper extends the original PettingZooEnv to support ParallelEnv
    where all agents act simultaneously, while maintaining backward compatibility
    with AEC (Agent Environment Cycle) environments.

    Example usage for parallel environment:
    ::
        from pettingzoo.mpe import simple_spread_v3

        # Create parallel environment
        parallel_env = simple_spread_v3.parallel_env()
        env = EnhancedPettingZooEnv(parallel_env, mode="parallel")

        # All agents receive observations simultaneously
        obs, info = env.reset()
        # obs["observations"] contains all agents' observations

        # All agents act simultaneously
        actions = {"agent_0": 0, "agent_1": 1, "agent_2": 0}
        obs, rewards, term, trunc, info = env.step(actions)

    Example usage for AEC environment (backward compatible):
    ::
        from pettingzoo.mpe import simple_spread_v3

        # Create AEC environment
        aec_env = simple_spread_v3.env()
        env = EnhancedPettingZooEnv(aec_env)  # auto-detects AEC

        # Works like original PettingZooEnv
        obs, info = env.reset()
        action = policy(obs)
        obs, rewards, term, trunc, info = env.step(action)
    """

    def __init__(
        self,
        env: Union[BaseWrapper, ParallelEnv, AECEnv],
        mode: Literal["aec", "parallel", "auto"] = "auto",
    ):
        """Initialize enhanced PettingZoo environment wrapper.

        Args:
            env: PettingZoo environment (AEC or Parallel)
            mode: Environment mode
                - "aec": Agent Environment Cycle (sequential actions)
                - "parallel": Parallel environment (simultaneous actions)
                - "auto": Auto-detect from environment type
        """
        # Auto-detect environment type
        if mode == "auto":
            # ParallelEnv has observation_spaces (plural) attribute
            if hasattr(env, "observation_spaces"):
                self.mode = "parallel"
            else:
                self.mode = "aec"
        else:
            self.mode = mode

        # For parallel env, we need different initialization
        if self.mode == "parallel":
            self._init_parallel(env)
        else:
            # Use parent class initialization for AEC
            super().__init__(env)
            self.is_parallel = False
            self._num_agents = len(self.agents)  # Store as private attribute
            self.metadata = getattr(env, "metadata", {})

    @property
    def num_agents(self):
        """Number of agents in the environment."""
        return self._num_agents

    def _init_parallel(self, env: ParallelEnv):
        """Initialize for parallel environment."""
        self.env = env
        self.agents = env.possible_agents
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}
        self.rewards = [0] * len(self.agents)

        # Add attributes for compatibility with tests
        self.is_parallel = True
        self._num_agents = len(self.agents)  # Store as private attribute
        self.metadata = getattr(env, "metadata", {})

        # For parallel env, get spaces from observation_spaces dict
        first_agent = self.agents[0]
        self.observation_space = env.observation_spaces[first_agent]
        self.action_space = env.action_spaces[first_agent]

        # Verify all agents have identical spaces
        assert all(
            env.observation_spaces[agent] == self.observation_space for agent in self.agents
        ), "All agents must have identical observation spaces"

        assert all(
            env.action_spaces[agent] == self.action_space for agent in self.agents
        ), "All agents must have identical action spaces"

    def reset(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        """Reset environment.

        Returns:
            For AEC mode: (obs_dict, info) where obs_dict contains agent_id, obs, and optionally mask
            For parallel mode: (obs_dict, info) where obs_dict contains observations for all agents
        """
        if self.mode == "parallel":
            return self._parallel_reset(*args, **kwargs)
        else:
            return super().reset(*args, **kwargs)

    def _parallel_reset(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        """Reset parallel environment."""
        observations, infos = self.env.reset(*args, **kwargs)

        # Convert to format compatible with collectors
        obs_dict = {
            "observations": observations,
            "agent_ids": list(self.agents),
        }

        # Add action masks if present
        if isinstance(self.action_space, spaces.Discrete):
            # Create masks for all agents
            masks = {}
            for agent in self.agents:
                if agent in observations:
                    if (
                        isinstance(observations[agent], dict)
                        and "action_mask" in observations[agent]
                    ):
                        masks[agent] = [m == 1 for m in observations[agent]["action_mask"]]
                    else:
                        masks[agent] = [True] * self.action_space.n
                else:
                    masks[agent] = [False] * self.action_space.n
            obs_dict["masks"] = masks

        return obs_dict, infos

    def step(
        self, action: Union[int, np.ndarray, Dict[str, Any]]
    ) -> tuple[dict, list, Union[bool, list], Union[bool, list], dict]:
        """Take a step in the environment.

        Args:
            action: For AEC mode, single action for current agent.
                   For parallel mode, dict mapping agent_id to action, or array of actions.

        Returns:
            obs: Observation dict
            rewards: List of rewards (indexed by agent_idx)
            terminated: Termination flag
            truncated: Truncation flag
            info: Additional info
        """
        if self.mode == "parallel":
            return self._parallel_step(action)
        else:
            return super().step(action)

    def _parallel_step(
        self, action: Union[Dict[str, Any], np.ndarray]
    ) -> tuple[dict, list, list, list, dict]:
        """Step in parallel environment."""
        # Convert array actions to dict if needed
        if isinstance(action, (np.ndarray, list)):
            action_dict = {agent: action[i] for i, agent in enumerate(self.agents)}
        else:
            action_dict = action

        # All agents step simultaneously
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)

        # Convert rewards to list format (indexed by agent_idx)
        reward_list = [0.0] * len(self.agents)
        for agent, reward in rewards.items():
            reward_list[self.agent_idx[agent]] = reward

        # Convert terminations/truncations to list format (indexed by agent_idx)
        term_list = [False] * len(self.agents)
        trunc_list = [False] * len(self.agents)
        for agent, term in terminations.items():
            term_list[self.agent_idx[agent]] = term
        for agent, trunc in truncations.items():
            trunc_list[self.agent_idx[agent]] = trunc

        # Format observations
        obs_dict = {
            "observations": observations,
            "agent_ids": list(self.agents),
        }

        # Add action masks if present
        if isinstance(self.action_space, spaces.Discrete):
            masks = {}
            for agent in self.agents:
                # Check if agent still has observations (not terminated)
                if agent in observations:
                    if (
                        isinstance(observations[agent], dict)
                        and "action_mask" in observations[agent]
                    ):
                        masks[agent] = [m == 1 for m in observations[agent]["action_mask"]]
                    else:
                        masks[agent] = [True] * self.action_space.n
                else:
                    # Agent is terminated, no valid actions
                    masks[agent] = [False] * self.action_space.n
            obs_dict["masks"] = masks

        return obs_dict, reward_list, term_list, trunc_list, infos

    def close(self) -> None:
        """Close environment."""
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        """Seed environment."""
        if self.mode == "parallel":
            # ParallelEnv uses reset(seed=seed)
            self._seed = seed
        else:
            super().seed(seed)

    def render(self) -> Any:
        """Render environment."""
        return self.env.render()
