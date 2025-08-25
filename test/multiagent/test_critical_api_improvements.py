"""Test critical API improvements needed for RealTianshouTrainer and CTDETrainer."""

import pytest
import torch
from pettingzoo.classic import tictactoe_v3
from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv


class TestCriticalImports:
    """Test critical import fixes for RealTianshouTrainer."""
    
    def test_ppo_policy_import(self):
        """Test PPOPolicy can be imported (needed by RealTianshouTrainer)."""
        # This was broken: from tianshou.policy import PPOPolicy
        # Now should work via simplified imports
        from tianshou_marl import PPOPolicy
        
        # Verify it's the correct class
        from tianshou.algorithm.modelfree.ppo import PPO
        assert PPOPolicy is PPO
        
    def test_policy_manager_import(self):
        """Test PolicyManager import (alias for FlexibleMultiAgentPolicyManager)."""
        from tianshou_marl import PolicyManager
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        assert PolicyManager is FlexibleMultiAgentPolicyManager


class TestPolicyManagerFromEnv:
    """Test FlexibleMultiAgentPolicyManager.from_env() for CTDETrainer."""
    
    def test_flexible_policy_manager_from_env(self):
        """Test FlexibleMultiAgentPolicyManager.from_env() works."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # This is what CTDETrainer needs
        policy_manager = FlexibleMultiAgentPolicyManager.from_env(
            env, 
            algorithm="DQN",
            mode="independent"
        )
        
        assert isinstance(policy_manager, FlexibleMultiAgentPolicyManager)
        assert policy_manager.mode == "independent"
        assert len(policy_manager.policies) == len(env.agents)
        
    def test_policy_manager_alias_from_env(self):
        """Test PolicyManager.from_env() alias works."""
        from tianshou_marl import PolicyManager
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Using the simpler alias
        policy_manager = PolicyManager.from_env(
            env,
            algorithm="DQN", 
            mode="shared"
        )
        
        assert policy_manager.mode == "shared"
        # In shared mode, all agents use the same policy
        policies = list(policy_manager.policies.values())
        if len(policies) > 1:
            assert all(p is policies[0] for p in policies[1:])
            
    def test_policy_manager_with_config(self):
        """Test PolicyManager.from_env() with configuration."""
        from tianshou_marl import PolicyManager
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        config = {
            "learning_rate": 1e-4,
            "hidden_sizes": [128, 128],
            "epsilon": 0.05,
        }
        
        policy_manager = PolicyManager.from_env(
            env,
            algorithm="DQN",
            mode="independent",
            config=config
        )
        
        assert policy_manager is not None
        assert len(policy_manager.policies) == len(env.agents)


class TestQMIXFromEnv:
    """Test QMIXPolicy.from_env() for Phase 4 CTDE implementation."""
    
    def test_qmix_from_env_basic(self):
        """Test basic QMIXPolicy.from_env() functionality."""
        from tianshou_marl import QMIXPolicy
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Simple one-line QMIX creation (vs 50+ lines before)
        qmix = QMIXPolicy.from_env(env)
        
        assert isinstance(qmix, QMIXPolicy)
        assert qmix.n_agents == len(env.agents)
        assert len(qmix.actors) == len(env.agents)
        assert qmix.mixer is not None
        
    def test_qmix_from_env_with_config(self):
        """Test QMIXPolicy.from_env() with custom configuration."""
        from tianshou_marl import QMIXPolicy
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        config = {
            "learning_rate": 5e-4,
            "hidden_sizes": [128, 64],
            "mixing_embed_dim": 64,
            "hypernet_embed_dim": 128,
            "epsilon": 0.05,
            "discount_factor": 0.95,
        }
        
        qmix = QMIXPolicy.from_env(env, config=config)
        
        assert qmix.epsilon == 0.05
        assert qmix.discount_factor == 0.95
        
    def test_qmix_device_support(self):
        """Test QMIXPolicy.from_env() with device selection."""
        from tianshou_marl import QMIXPolicy
        from tianshou_marl.device_utils import get_default_device
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Test auto device
        qmix = QMIXPolicy.from_env(env, device="auto")
        expected_device = get_default_device()
        
        # Check actors are on correct device
        for actor in qmix.actors:
            device = next(actor.parameters()).device
            assert device.type == expected_device
            
        # Check mixer is on correct device
        mixer_device = next(qmix.mixer.parameters()).device
        assert mixer_device.type == expected_device


class TestEndToEndUsage:
    """Test complete usage scenarios matching trainer requirements."""
    
    def test_realtianshou_trainer_compatibility(self):
        """Test imports and API needed by RealTianshouTrainer."""
        # All imports that RealTianshouTrainer needs
        from tianshou_marl import PPOPolicy, PolicyManager
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Create policy manager as trainer would
        policy_manager = PolicyManager.from_env(
            env,
            algorithm="DQN",  # PPO would need actor-critic setup
            mode="independent"
        )
        
        assert policy_manager is not None
        
    def test_ctde_trainer_compatibility(self):
        """Test imports and API needed by CTDETrainer."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        from tianshou_marl import QMIXPolicy
        
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # Option 1: Use FlexibleMultiAgentPolicyManager for independent policies
        policy_manager = FlexibleMultiAgentPolicyManager.from_env(
            env,
            algorithm="DQN",
            mode="independent"
        )
        assert policy_manager is not None
        
        # Option 2: Use QMIX for CTDE
        qmix = QMIXPolicy.from_env(env)
        assert qmix is not None
        
    def test_comparison_old_vs_new(self):
        """Compare old boilerplate vs new simplified API."""
        env = tictactoe_v3.env()
        env = EnhancedPettingZooEnv(env)
        
        # NEW WAY: 2 lines
        from tianshou_marl import QMIXPolicy
        qmix_new = QMIXPolicy.from_env(env, config={"learning_rate": 1e-3})
        
        # OLD WAY: 30+ lines (abbreviated)
        from tianshou.algorithm.multiagent.ctde import QMIXMixer, QMIXPolicy as OrigQMIX
        from tianshou.utils.net.common import Net
        import numpy as np
        
        # Manual setup...
        n_agents = len(env.agents)
        obs_space = env.observation_space
        
        # Handle Dict spaces manually...
        if hasattr(obs_space, 'spaces') and 'observation' in obs_space.spaces:
            state_shape = obs_space.spaces['observation'].shape
        else:
            state_shape = (4,)  # Fallback
            
        action_shape = env.action_space.n
        
        # Create actors manually...
        actors = []
        for _ in range(n_agents):
            actors.append(Net(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_sizes=[64, 64]
            ))
            
        # Create mixer manually...
        global_state_dim = int(np.prod(state_shape)) * n_agents
        mixer = QMIXMixer(n_agents, global_state_dim, 32, 64)
        
        # Setup optimizer manually...
        import torch.optim as optim
        all_params = []
        for actor in actors:
            all_params.extend(actor.parameters())
        all_params.extend(mixer.parameters())
        optimizer = optim.Adam(all_params, lr=1e-3)
        
        # Finally create policy...
        qmix_old = OrigQMIX(
            actors=actors,
            mixer=mixer,
            observation_space=obs_space,
            action_space=env.action_space,
            n_agents=n_agents,
            optimizer=optimizer,
            discount_factor=0.99,
            epsilon=0.1,
        )
        
        # Both should work
        assert isinstance(qmix_new, OrigQMIX)
        assert isinstance(qmix_old, OrigQMIX)
        
        # New way is MUCH simpler!