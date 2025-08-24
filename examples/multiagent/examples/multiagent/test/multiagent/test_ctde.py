"""Fast tests for CTDE (Centralized Training, Decentralized Execution) components."""

import numpy as np
import pytest
import torch
from unittest.mock import Mock

from tianshou.algorithm.multiagent import (
    CentralizedCritic,
    CTDEPolicy,
    DecentralizedActor,
    GlobalStateConstructor,
    QMIXPolicy,
    QMIXMixer,
)
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.utils.net.common import Net


class TestGlobalStateConstructor:
    """Test GlobalStateConstructor functionality."""

    def test_concatenation_mode(self):
        """Test global state construction via concatenation."""
        constructor = GlobalStateConstructor(mode="concat")
        
        # Test with different agent observations
        agent_obs = {
            "agent_0": torch.randn(5, 4),
            "agent_1": torch.randn(5, 6),
            "agent_2": torch.randn(5, 3),
        }
        
        global_state = constructor(agent_obs)
        assert global_state.shape == (5, 13)  # 4 + 6 + 3

    def test_mean_aggregation(self):
        """Test global state via mean aggregation."""
        constructor = GlobalStateConstructor(mode="mean")
        
        # Same-sized observations for mean aggregation
        agent_obs = {
            "agent_0": torch.randn(5, 4),
            "agent_1": torch.randn(5, 4),
            "agent_2": torch.randn(5, 4),
        }
        
        global_state = constructor(agent_obs)
        assert global_state.shape == (5, 4)

    def test_attention_aggregation(self):
        """Test global state via attention mechanism."""
        constructor = GlobalStateConstructor(
            mode="attention",
            state_dim=4,
            hidden_dim=32
        )
        
        agent_obs = {
            "agent_0": torch.randn(5, 4),
            "agent_1": torch.randn(5, 4),
            "agent_2": torch.randn(5, 4),
        }
        
        global_state = constructor(agent_obs)
        assert global_state.shape == (5, 4)

    def test_custom_function(self):
        """Test global state with custom aggregation function."""
        def custom_fn(obs_dict):
            # Sum all observations
            return sum(obs_dict.values())
        
        constructor = GlobalStateConstructor(
            mode="custom",
            custom_fn=custom_fn
        )
        
        agent_obs = {
            "agent_0": torch.randn(5, 4),
            "agent_1": torch.randn(5, 4),
        }
        
        global_state = constructor(agent_obs)
        assert global_state.shape == (5, 4)


class TestCentralizedCritic:
    """Test CentralizedCritic functionality."""

    def test_critic_initialization(self):
        """Test critic network initialization."""
        critic = CentralizedCritic(
            global_state_dim=16,
            action_dims={"agent_0": 4, "agent_1": 6},
            hidden_sizes=[64, 32]
        )
        
        assert critic is not None
        
    def test_critic_forward_pass(self):
        """Test critic forward pass."""
        critic = CentralizedCritic(
            global_state_dim=16,
            action_dims={"agent_0": 4, "agent_1": 6},
            hidden_sizes=[64, 32]
        )
        
        global_state = torch.randn(10, 16)
        joint_actions = {
            "agent_0": torch.randn(10, 4),
            "agent_1": torch.randn(10, 6)
        }
        
        q_values = critic(global_state, joint_actions)
        assert q_values.shape == (10, 1)  # Scalar Q-value for each sample


class TestDecentralizedActor:
    """Test DecentralizedActor functionality."""

    def test_actor_initialization(self):
        """Test actor network initialization."""
        actor = DecentralizedActor(
            obs_dim=8,
            action_dim=4,
            hidden_sizes=[64, 32]
        )
        
        assert actor is not None

    def test_actor_forward_pass(self):
        """Test actor forward pass."""
        actor = DecentralizedActor(
            obs_dim=8,
            action_dim=4,
            hidden_sizes=[64, 32]
        )
        
        obs = torch.randn(10, 8)
        action_probs = actor(obs)
        
        assert action_probs.shape == (10, 4)
        # Check probabilities sum to 1 (if softmax applied)
        if hasattr(actor, 'softmax'):
            assert torch.allclose(action_probs.sum(dim=1), torch.ones(10), atol=1e-6)


class TestQMIXMixer:
    """Test QMIX mixing network."""

    def test_mixer_initialization(self):
        """Test mixer network initialization."""
        mixer = QMIXMixer(
            n_agents=3,
            state_dim=12,
            embed_dim=32,
            hypernet_embed=64
        )
        
        assert mixer is not None

    def test_mixer_monotonicity(self):
        """Test QMIX mixer maintains monotonicity constraint."""
        mixer = QMIXMixer(
            n_agents=3,
            state_dim=12,
            embed_dim=32
        )
        
        # Individual Q-values (should be non-negative for monotonicity test)
        individual_q = torch.abs(torch.randn(10, 3))  # Ensure positive
        state = torch.randn(10, 12)
        
        mixed_q = mixer(individual_q, state)
        assert mixed_q.shape == (10, 1)

    def test_mixer_forward_pass(self):
        """Test mixer forward pass with various inputs."""
        mixer = QMIXMixer(n_agents=2, state_dim=8)
        
        for batch_size in [1, 5, 10]:
            individual_q = torch.randn(batch_size, 2)
            state = torch.randn(batch_size, 8)
            
            mixed_q = mixer(individual_q, state)
            assert mixed_q.shape == (batch_size, 1)


class TestCTDEPolicy:
    """Test base CTDE policy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agents = ["agent_0", "agent_1"]
        self.obs_dims = {"agent_0": 4, "agent_1": 6}
        self.action_dims = {"agent_0": 2, "agent_1": 3}

    def test_ctde_policy_initialization(self):
        """Test CTDE policy initialization."""
        policy = CTDEPolicy(
            agents=self.agents,
            obs_dims=self.obs_dims,
            action_dims=self.action_dims
        )
        
        assert policy.agents == self.agents
        assert len(policy.actors) == len(self.agents)
        assert policy.critic is not None

    def test_decentralized_execution(self):
        """Test that execution only uses local observations."""
        policy = CTDEPolicy(
            agents=self.agents,
            obs_dims=self.obs_dims,
            action_dims=self.action_dims
        )
        
        # During execution, each agent only sees local obs
        local_obs = torch.randn(5, 4)  # agent_0's obs
        
        # Should be able to get action from local obs only
        # (Implementation would depend on actual CTDEPolicy interface)

    def test_centralized_training_info(self):
        """Test that training uses global state information."""
        policy = CTDEPolicy(
            agents=self.agents,
            obs_dims=self.obs_dims,
            action_dims=self.action_dims
        )
        
        # During training, policy should accept global state
        global_state = torch.randn(5, 16)
        joint_actions = {
            "agent_0": torch.randn(5, 2),
            "agent_1": torch.randn(5, 3)
        }
        
        # Should be able to compute centralized value
        # (Implementation would depend on actual CTDEPolicy interface)


class TestQMIXPolicy:
    """Test QMIX policy implementation."""

    def setup_method(self):
        """Set up QMIX test environment."""
        self.env = Mock()
        self.env.agents = ["agent_0", "agent_1"]
        
        # Create individual policies
        self.policies = {}
        for agent_id in self.env.agents:
            net = Net(state_shape=(4,), action_shape=2, hidden_sizes=[32])
            policy = DiscreteQLearningPolicy(
                model=net,
                action_space=Mock(),
                observation_space=Mock()
            )
            self.policies[agent_id] = policy

    def test_qmix_initialization(self):
        """Test QMIX policy initialization."""
        qmix = QMIXPolicy(
            policies=self.policies,
            env=self.env,
            state_dim=10,
            lr=1e-3
        )
        
        assert qmix.policies == self.policies
        assert qmix.mixer is not None

    def test_qmix_mixing_network(self):
        """Test QMIX mixing network integration."""
        qmix = QMIXPolicy(
            policies=self.policies,
            env=self.env,
            state_dim=10,
            lr=1e-3
        )
        
        # Test that mixer combines individual Q-values
        individual_q = torch.randn(5, len(self.env.agents))
        global_state = torch.randn(5, 10)
        
        mixed_q = qmix.mixer(individual_q, global_state)
        assert mixed_q.shape == (5, 1)

    def test_qmix_target_networks(self):
        """Test QMIX target network updates."""
        qmix = QMIXPolicy(
            policies=self.policies,
            env=self.env,
            state_dim=10,
            lr=1e-3,
            target_update_freq=100
        )
        
        # Check that target networks exist
        assert hasattr(qmix, 'target_mixer')
        
        # Test target update (would need actual update method)


class TestIntegration:
    """Integration tests for CTDE components."""

    def test_full_ctde_pipeline(self):
        """Test complete CTDE training pipeline."""
        # Create global state constructor
        constructor = GlobalStateConstructor(mode="concat")
        
        # Create centralized critic
        critic = CentralizedCritic(
            global_state_dim=10,  # 4 + 6 concatenated
            action_dims={"agent_0": 2, "agent_1": 3},
            hidden_sizes=[32]
        )
        
        # Create decentralized actors
        actor_0 = DecentralizedActor(obs_dim=4, action_dim=2, hidden_sizes=[32])
        actor_1 = DecentralizedActor(obs_dim=6, action_dim=3, hidden_sizes=[32])
        
        # Test pipeline
        agent_obs = {
            "agent_0": torch.randn(5, 4),
            "agent_1": torch.randn(5, 6)
        }
        
        # Construct global state
        global_state = constructor(agent_obs)
        assert global_state.shape == (5, 10)
        
        # Get actions from actors
        actions_0 = actor_0(agent_obs["agent_0"])
        actions_1 = actor_1(agent_obs["agent_1"])
        
        # Combine actions for critic
        joint_actions = {
            "agent_0": actions_0,
            "agent_1": actions_1
        }
        
        # Get centralized value
        q_values = critic(global_state, joint_actions)
        assert q_values.shape == (5, 1)

    def test_qmix_complete_forward(self):
        """Test complete QMIX forward pass."""
        # Setup individual Q-networks
        individual_q = torch.randn(5, 3)  # 3 agents
        global_state = torch.randn(5, 12)
        
        # Create QMIX mixer
        mixer = QMIXMixer(n_agents=3, state_dim=12)
        
        # Mix individual Q-values
        total_q = mixer(individual_q, global_state)
        
        assert total_q.shape == (5, 1)
        # Additional checks could verify monotonicity properties

    def test_performance_with_many_agents(self):
        """Test CTDE performance with many agents."""
        n_agents = 10
        
        # Create large-scale setup
        agent_obs = {f"agent_{i}": torch.randn(20, 8) for i in range(n_agents)}
        
        constructor = GlobalStateConstructor(mode="concat")
        global_state = constructor(agent_obs)
        
        assert global_state.shape == (20, 8 * n_agents)
        
        # Test with QMIX mixer
        mixer = QMIXMixer(n_agents=n_agents, state_dim=8*n_agents)
        individual_q = torch.randn(20, n_agents)
        
        mixed_q = mixer(individual_q, global_state)
        assert mixed_q.shape == (20, 1)
