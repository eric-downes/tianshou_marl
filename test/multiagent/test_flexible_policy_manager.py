"""Tests for flexible multi-agent policy manager."""

import numpy as np
import pytest
import torch.nn as nn
from gymnasium import spaces

from tianshou.algorithm.algorithm_base import Policy
from tianshou.algorithm.multiagent.flexible_policy import (
    FlexibleMultiAgentPolicyManager,
)
from tianshou.data import Batch


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, policy_id: str):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
        )
        self.policy_id = policy_id
        self.parameters = nn.Linear(4, 2)  # Mock parameters

    def forward(self, batch: Batch, state=None, **kwargs):
        """Mock forward pass."""
        return Batch(act=np.random.randint(0, 2, size=batch.obs.shape[0]))

    def add_exploration_noise(self, act, batch):
        """Mock exploration noise."""
        return act

    def state_dict(self):
        """Get state dict for parameter checking."""
        return self.parameters.state_dict()


class MockEnv:
    """Mock PettingZoo environment."""

    def __init__(self, num_agents: int = 3):
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents.copy()
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}


@pytest.mark.slow
class TestFlexibleMultiAgentPolicyManager:
    """Test flexible multi-agent policy manager."""

    def test_independent_mode(self):
        """Test independent mode where each agent has its own policy."""
        env = MockEnv(num_agents=3)
        policies = [MockPolicy(f"policy_{i}") for i in range(3)]

        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        # Each agent should have a different policy
        assert len(manager.policy_map) == 3
        assert manager.policy_map["agent_0"] is policies[0]
        assert manager.policy_map["agent_1"] is policies[1]
        assert manager.policy_map["agent_2"] is policies[2]

    def test_shared_mode(self):
        """Test shared mode where all agents share the same policy."""
        env = MockEnv(num_agents=3)
        shared_policy = MockPolicy("shared_policy")

        manager = FlexibleMultiAgentPolicyManager(policies=shared_policy, env=env, mode="shared")

        # All agents should share the same policy instance
        assert len(manager.policy_map) == 3
        assert manager.policy_map["agent_0"] is shared_policy
        assert manager.policy_map["agent_1"] is shared_policy
        assert manager.policy_map["agent_2"] is shared_policy

    def test_grouped_mode(self):
        """Test grouped mode where agents in same group share policy."""
        env = MockEnv(num_agents=6)

        # Define two groups
        agent_groups = {
            "team_a": ["agent_0", "agent_1", "agent_2"],
            "team_b": ["agent_3", "agent_4", "agent_5"],
        }

        policies = {"team_a": MockPolicy("team_a_policy"), "team_b": MockPolicy("team_b_policy")}

        manager = FlexibleMultiAgentPolicyManager(
            policies=policies, env=env, mode="grouped", agent_groups=agent_groups
        )

        # Team A agents should share team_a_policy
        assert manager.policy_map["agent_0"] is policies["team_a"]
        assert manager.policy_map["agent_1"] is policies["team_a"]
        assert manager.policy_map["agent_2"] is policies["team_a"]

        # Team B agents should share team_b_policy
        assert manager.policy_map["agent_3"] is policies["team_b"]
        assert manager.policy_map["agent_4"] is policies["team_b"]
        assert manager.policy_map["agent_5"] is policies["team_b"]

    def test_custom_mode(self):
        """Test custom mode with user-defined policy mapping."""
        env = MockEnv(num_agents=4)

        policies = {
            "explorer": MockPolicy("explorer_policy"),
            "defender": MockPolicy("defender_policy"),
        }

        # Custom mapping function
        def policy_mapping_fn(agent_id: str) -> str:
            # Even numbered agents are explorers, odd are defenders
            agent_num = int(agent_id.split("_")[1])
            return "explorer" if agent_num % 2 == 0 else "defender"

        manager = FlexibleMultiAgentPolicyManager(
            policies=policies, env=env, mode="custom", policy_mapping_fn=policy_mapping_fn
        )

        # Check custom mapping
        assert manager.policy_map["agent_0"] is policies["explorer"]
        assert manager.policy_map["agent_1"] is policies["defender"]
        assert manager.policy_map["agent_2"] is policies["explorer"]
        assert manager.policy_map["agent_3"] is policies["defender"]

    def test_parameter_sharing_memory_efficiency(self):
        """Test that parameter sharing reduces memory usage."""
        env = MockEnv(num_agents=10)

        # Independent policies
        independent_policies = [MockPolicy(f"policy_{i}") for i in range(10)]
        independent_manager = FlexibleMultiAgentPolicyManager(
            policies=independent_policies, env=env, mode="independent"
        )

        # Shared policy
        shared_policy = MockPolicy("shared_policy")
        shared_manager = FlexibleMultiAgentPolicyManager(
            policies=shared_policy, env=env, mode="shared"
        )

        # Count unique policy instances
        independent_policies_set = set()
        for policy in independent_manager.policy_map.values():
            independent_policies_set.add(id(policy))

        shared_policies_set = set()
        for policy in shared_manager.policy_map.values():
            shared_policies_set.add(id(policy))

        # Independent should have 10 unique policies, shared should have only 1
        assert len(independent_policies_set) == 10
        assert len(shared_policies_set) == 1

    def test_forward_with_shared_policy(self):
        """Test forward pass with shared policy."""
        env = MockEnv(num_agents=3)
        shared_policy = MockPolicy("shared_policy")

        manager = FlexibleMultiAgentPolicyManager(policies=shared_policy, env=env, mode="shared")

        # Create batch with observations for all agents
        batch = Batch(
            obs=Batch(
                agent_id=np.array(
                    ["agent_0", "agent_1", "agent_2", "agent_0", "agent_1", "agent_2"]
                ),
                obs=np.random.rand(6, 4).astype(np.float32),
            )
        )

        # Forward should work correctly
        result = manager.forward(batch)
        assert "act" in result
        assert len(result.act) == 6

    def test_dict_policies_input(self):
        """Test initialization with dict of policies."""
        env = MockEnv(num_agents=3)

        policies = {
            "agent_0": MockPolicy("policy_0"),
            "agent_1": MockPolicy("policy_1"),
            "agent_2": MockPolicy("policy_2"),
        }

        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        # Should map correctly
        assert manager.policy_map["agent_0"] is policies["agent_0"]
        assert manager.policy_map["agent_1"] is policies["agent_1"]
        assert manager.policy_map["agent_2"] is policies["agent_2"]

    def test_invalid_configuration(self):
        """Test that invalid configurations raise errors."""
        env = MockEnv(num_agents=3)

        # Independent mode requires multiple policies
        with pytest.raises(ValueError):
            FlexibleMultiAgentPolicyManager(
                policies=MockPolicy("single"), env=env, mode="independent"
            )

        # Grouped mode requires agent_groups
        with pytest.raises(ValueError):
            FlexibleMultiAgentPolicyManager(
                policies=[MockPolicy("p1"), MockPolicy("p2")],
                env=env,
                mode="grouped",
                agent_groups=None,
            )

        # Custom mode requires policy_mapping_fn
        with pytest.raises(ValueError):
            FlexibleMultiAgentPolicyManager(
                policies={"p1": MockPolicy("p1")}, env=env, mode="custom", policy_mapping_fn=None
            )
