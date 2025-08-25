"""Tests for simplified MARL API."""

import pytest
import torch
from gymnasium import spaces
from pettingzoo.classic import tictactoe_v3

from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv


class TestSimplifiedImports:
    """Test simplified import structure."""

    def test_simplified_imports(self):
        """Test that simplified imports work correctly."""
        # These imports should work after implementation
        from tianshou_marl import (
            PolicyManager,
            QMIXPolicy,
            DQNPolicy,
            PPOPolicy,
            AutoPolicy,
        )
        
        # Verify they are the correct classes
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            QMIXPolicy as OriginalQMIXPolicy,
        )
        from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
        from tianshou.algorithm.modelfree.ppo import PPO as OriginalPPOPolicy
        
        assert PolicyManager is FlexibleMultiAgentPolicyManager
        assert QMIXPolicy is OriginalQMIXPolicy
        assert DQNPolicy is DiscreteQLearningPolicy
        assert PPOPolicy is OriginalPPOPolicy
        
    def test_backward_compatibility(self):
        """Test that original imports still work."""
        # Original imports should still work
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        from tianshou.algorithm.multiagent import QMIXPolicy
        from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
        
        assert FlexibleMultiAgentPolicyManager is not None
        assert QMIXPolicy is not None
        assert DiscreteQLearningPolicy is not None


class TestAutoPolicy:
    """Test AutoPolicy with automatic environment detection."""
    
    def test_auto_policy_discrete_action_space(self):
        """Test AutoPolicy correctly selects DQN for discrete action spaces."""
        from tianshou_marl import AutoPolicy
        
        # Create a mock environment with discrete action space
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # AutoPolicy should detect discrete action space and create DQN
        policy = AutoPolicy.from_env(
            env,
            config={
                "learning_rate": 1e-3,
                "hidden_sizes": [64, 64],
                "epsilon": 0.1,
            }
        )
        
        # Verify it created a DQN-based policy manager
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        assert isinstance(policy, FlexibleMultiAgentPolicyManager)
        
    def test_auto_policy_continuous_action_space(self):
        """Test AutoPolicy correctly selects appropriate policy for continuous action spaces."""
        from tianshou_marl import AutoPolicy
        
        # Create a mock environment with continuous action space
        class MockContinuousEnv:
            def __init__(self):
                self.agents = ["agent_0", "agent_1"]
                self.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
                self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
                
        env = MockContinuousEnv()
        
        # AutoPolicy should detect continuous action space
        policy = AutoPolicy.from_env(
            env,
            config={
                "learning_rate": 1e-3,
                "hidden_sizes": [64, 64],
            }
        )
        
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        assert isinstance(policy, FlexibleMultiAgentPolicyManager)
        
    def test_auto_policy_with_mode_selection(self):
        """Test AutoPolicy respects mode parameter."""
        from tianshou_marl import AutoPolicy
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Test independent mode
        policy_independent = AutoPolicy.from_env(
            env,
            mode="independent",
            config={"learning_rate": 1e-3}
        )
        assert policy_independent.mode == "independent"
        
        # Test shared mode (parameter sharing)
        policy_shared = AutoPolicy.from_env(
            env,
            mode="shared",
            config={"learning_rate": 1e-3}
        )
        assert policy_shared.mode == "shared"
        
    def test_auto_policy_default_network_architecture(self):
        """Test AutoPolicy creates reasonable default network architectures."""
        from tianshou_marl import AutoPolicy
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Create policy without specifying hidden_sizes
        policy = AutoPolicy.from_env(env)
        
        # Should have created policies with reasonable defaults
        assert policy is not None
        assert len(policy.policies) > 0


class TestFactoryMethods:
    """Test factory methods for simplified policy creation."""
    
    def test_policy_manager_from_env(self):
        """Test PolicyManager.from_env factory method."""
        from tianshou_marl import PolicyManager
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Create policy manager from environment
        manager = PolicyManager.from_env(
            env,
            algorithm="DQN",
            mode="independent",
            config={
                "learning_rate": 1e-3,
                "hidden_sizes": [64, 64],
                "epsilon": 0.1,
            }
        )
        
        assert manager is not None
        assert manager.mode == "independent"
        assert len(manager.policies) == len(env.agents)
        
    def test_policy_manager_shared_mode(self):
        """Test PolicyManager with parameter sharing."""
        from tianshou_marl import PolicyManager
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Create shared policy
        manager = PolicyManager.from_env(
            env,
            algorithm="DQN",
            mode="shared",
            config={"learning_rate": 1e-3}
        )
        
        assert manager.mode == "shared"
        # All agents should share the same policy instance
        policies = list(manager.policies.values())
        if policies:
            first_policy = policies[0]
            for policy in policies[1:]:
                assert policy is first_policy
                
    @pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()), 
        reason="No GPU available (CUDA or MPS)"
    )
    def test_device_selection(self):
        """Test automatic device selection (CUDA or Metal)."""
        from tianshou_marl import AutoPolicy
        from tianshou_marl.device_utils import get_default_device
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Test auto device selection
        policy_auto = AutoPolicy.from_env(env, device="auto")
        expected_device = get_default_device()
        
        # Check if models are on correct device
        for agent_policy in policy_auto.policies.values():
            if hasattr(agent_policy, 'model'):
                for param in agent_policy.model.parameters():
                    assert param.device.type == expected_device
                    
        # Test explicit MPS if available
        if torch.backends.mps.is_available():
            policy_mps = AutoPolicy.from_env(env, device="mps")
            for agent_policy in policy_mps.policies.values():
                if hasattr(agent_policy, 'model'):
                    for param in agent_policy.model.parameters():
                        assert param.device.type == "mps"