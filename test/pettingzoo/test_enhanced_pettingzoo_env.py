"""Tests for enhanced PettingZoo environment wrapper with parallel support."""


import numpy as np
import pytest
from gymnasium import spaces

from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv


class MockParallelEnv:
    """Mock PettingZoo ParallelEnv for testing."""

    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32) for agent in self.agents
        }
        self.action_spaces = {agent: spaces.Discrete(2) for agent in self.agents}
        self.observations = None
        self.infos = None

    def reset(self, seed=None, options=None):
        """Reset and return observations for all agents."""
        self.agents = self.possible_agents.copy()
        self.observations = {agent: np.random.rand(4).astype(np.float32) for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return self.observations, self.infos

    def step(self, actions):
        """All agents step simultaneously."""
        assert set(actions.keys()) == set(self.agents)

        observations = {agent: np.random.rand(4).astype(np.float32) for agent in self.agents}
        rewards = {agent: np.random.rand() for agent in self.agents}
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def close(self):
        pass

    def render(self):
        pass


class MockAECEnv:
    """Mock PettingZoo AEC environment for testing."""

    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()
        self.agent_selection = None
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = None

    def reset(self, seed=None, options=None):
        """Reset environment and select first agent."""
        self.agents = self.possible_agents.copy()
        self.agent_selection = self.agents[0]
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action):
        """Single agent takes action."""
        if self.agent_selection:
            # Process action for current agent
            self.rewards[self.agent_selection] = np.random.rand()

            # Move to next agent
            current_idx = self.agents.index(self.agent_selection)
            next_idx = (current_idx + 1) % len(self.agents)
            self.agent_selection = self.agents[next_idx]

    def last(self, observer=None):
        """Get last observation, reward, termination, truncation, info."""
        obs = np.random.rand(4).astype(np.float32)
        rew = self.rewards.get(self.agent_selection, 0.0)
        term = self.terminations.get(self.agent_selection, False)
        trunc = self.truncations.get(self.agent_selection, False)
        info = self.infos.get(self.agent_selection, {})
        return obs, rew, term, trunc, info

    def observation_space(self, agent):
        return spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Discrete(2)

    def close(self):
        pass

    def render(self):
        pass


@pytest.mark.slow
class TestEnhancedPettingZooEnv:
    """Test enhanced PettingZoo environment wrapper."""

    def test_auto_detect_parallel_env(self):
        """Test automatic detection of parallel environment."""
        parallel_env = MockParallelEnv()
        env = EnhancedPettingZooEnv(parallel_env, mode="auto")
        assert env.mode == "parallel"

    def test_auto_detect_aec_env(self):
        """Test automatic detection of AEC environment."""
        aec_env = MockAECEnv()
        env = EnhancedPettingZooEnv(aec_env, mode="auto")
        assert env.mode == "aec"

    def test_parallel_reset(self):
        """Test reset for parallel environment."""
        parallel_env = MockParallelEnv()
        env = EnhancedPettingZooEnv(parallel_env, mode="parallel")

        obs, info = env.reset()

        # Should return dict with all agents' observations
        assert "observations" in obs
        assert len(obs["observations"]) == 3
        assert all(agent in obs["observations"] for agent in ["agent_0", "agent_1", "agent_2"])

    def test_parallel_step(self):
        """Test step for parallel environment."""
        parallel_env = MockParallelEnv()
        env = EnhancedPettingZooEnv(parallel_env, mode="parallel")

        env.reset()

        # All agents act simultaneously
        actions = {"agent_0": 0, "agent_1": 1, "agent_2": 0}
        obs, rewards, term, trunc, info = env.step(actions)

        # Should return observations for all agents
        assert "observations" in obs
        assert len(obs["observations"]) == 3

        # Rewards should be array format for compatibility
        assert isinstance(rewards, (list, np.ndarray))
        assert len(rewards) == 3

    def test_aec_step(self):
        """Test step for AEC environment."""
        aec_env = MockAECEnv()
        env = EnhancedPettingZooEnv(aec_env, mode="aec")

        obs, info = env.reset()

        # Single agent acts
        action = 0
        obs, rewards, term, trunc, info = env.step(action)

        # Should return observation for current agent
        assert "agent_id" in obs
        assert "obs" in obs
        assert isinstance(rewards, (list, np.ndarray))

    def test_parallel_dict_actions(self):
        """Test handling of dict actions in parallel mode."""
        parallel_env = MockParallelEnv()
        env = EnhancedPettingZooEnv(parallel_env, mode="parallel")

        env.reset()

        # Test with dict actions
        actions = {"agent_0": 0, "agent_1": 1, "agent_2": 0}
        obs, rewards, term, trunc, info = env.step(actions)

        assert obs is not None
        assert rewards is not None

    def test_parallel_array_actions(self):
        """Test handling of array actions in parallel mode."""
        parallel_env = MockParallelEnv()
        env = EnhancedPettingZooEnv(parallel_env, mode="parallel")

        env.reset()

        # Test with array actions (assuming agent order)
        actions = np.array([0, 1, 0])
        obs, rewards, term, trunc, info = env.step(actions)

        assert obs is not None
        assert rewards is not None

    def test_compatibility_with_existing_code(self):
        """Test backward compatibility with existing PettingZooEnv usage."""
        aec_env = MockAECEnv()
        env = EnhancedPettingZooEnv(aec_env)  # Should default to AEC mode

        # Should work like original PettingZooEnv
        obs, info = env.reset()
        assert "agent_id" in obs
        assert "obs" in obs

        obs, rewards, term, trunc, info = env.step(0)
        assert "agent_id" in obs
        assert isinstance(rewards, (list, np.ndarray))
