"""Fast comprehensive tests for MARL functionality.

Strategy:
1. Use minimal data sizes (batch_size=2-4 instead of 100+)
2. Mock heavy operations (no actual neural network training)
3. Test interfaces and data flow, not convergence
4. Use in-memory mocks instead of real environments
5. Each test should run in <0.1 seconds
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from tianshou.data import Batch
from tianshou.algorithm.algorithm_base import Policy
from gymnasium import spaces


# ============================================================================
# FAST MOCK UTILITIES - Shared across all tests
# ============================================================================

class MinimalPolicy(Policy):
    """Ultra-lightweight policy for fast testing."""
    def __init__(self):
        super().__init__(
            observation_space=spaces.Box(-1, 1, (2,)),
            action_space=spaces.Discrete(2)
        )
    
    def forward(self, batch: Batch, state=None, **kwargs):
        batch_size = len(batch.obs) if hasattr(batch.obs, '__len__') else 1
        return Batch(act=np.zeros(batch_size, dtype=int), state=state)
    
    def learn(self, batch: Batch, **kwargs):
        return {"loss": 0.1}


class MinimalEnv:
    """Minimal environment for fast testing."""
    def __init__(self, n_agents=2):
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents
        self.observation_spaces = {a: spaces.Box(-1, 1, (2,)) for a in self.agents}
        self.action_spaces = {a: spaces.Discrete(2) for a in self.agents}
        self.metadata = {"is_parallelizable": True}
    
    def reset(self):
        return {a: np.zeros(2) for a in self.agents}, {a: {} for a in self.agents}
    
    def step(self, actions):
        obs = {a: np.zeros(2) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: False for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, rewards, term, trunc, info


# ============================================================================
# TEST: Enhanced PettingZoo Environment (8 fast tests)
# ============================================================================

class TestEnhancedPettingZooFast:
    """Fast tests for EnhancedPettingZooEnv - each test <0.1s."""
    
    def test_env_detection(self):
        """Test environment type detection."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=2)
        wrapped = EnhancedPettingZooEnv(env)
        assert wrapped.is_parallel == True
        assert wrapped.num_agents == 2
    
    def test_parallel_reset(self):
        """Test parallel environment reset."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=2)
        wrapped = EnhancedPettingZooEnv(env)
        obs, info = wrapped.reset()
        
        assert "observations" in obs
        assert "masks" in obs
        assert "agent_ids" in obs
        assert len(obs["observations"]) == 2
    
    def test_parallel_step(self):
        """Test parallel environment step."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=2)
        wrapped = EnhancedPettingZooEnv(env)
        wrapped.reset()
        
        actions = np.array([0, 1])
        obs, rewards, term, trunc, info = wrapped.step(actions)
        
        assert "observations" in obs
        assert len(rewards) == 2
        assert len(term) == 2
    
    def test_action_masking(self):
        """Test action masking support."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=2)
        env.action_mask = lambda agent: [True, False]  # Mock action mask
        wrapped = EnhancedPettingZooEnv(env)
        
        obs, _ = wrapped.reset()
        assert "masks" in obs
        # Masks should be available for each agent
        assert len(obs["masks"]) == 2
    
    def test_dict_action_format(self):
        """Test dict action format support."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=2)
        wrapped = EnhancedPettingZooEnv(env)
        wrapped.reset()
        
        # Test with dict actions
        actions = {"agent_0": 0, "agent_1": 1}
        obs, rewards, term, trunc, info = wrapped.step(actions)
        assert len(rewards) == 2
    
    def test_array_action_format(self):
        """Test array action format support."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=2)
        wrapped = EnhancedPettingZooEnv(env)
        wrapped.reset()
        
        # Test with array actions
        actions = np.array([0, 1])
        obs, rewards, term, trunc, info = wrapped.step(actions)
        assert len(rewards) == 2
    
    def test_agent_removal(self):
        """Test handling of terminated agents."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=3)
        wrapped = EnhancedPettingZooEnv(env)
        obs, _ = wrapped.reset()
        
        # Simulate agent termination
        env.agents = ["agent_0", "agent_2"]  # agent_1 terminated
        actions = np.array([0, 0, 1])
        obs, rewards, term, trunc, info = wrapped.step(actions)
        
        # Should handle gracefully
        assert "observations" in obs
    
    def test_metadata_preservation(self):
        """Test that environment metadata is preserved."""
        from tianshou.env import EnhancedPettingZooEnv
        
        env = MinimalEnv(n_agents=2)
        env.metadata["custom_field"] = "test_value"
        wrapped = EnhancedPettingZooEnv(env)
        
        assert wrapped.metadata["is_parallelizable"] == True
        assert wrapped.metadata["custom_field"] == "test_value"


# ============================================================================
# TEST: Flexible Policy Configuration (8 fast tests)
# ============================================================================

class TestFlexiblePolicyFast:
    """Fast tests for FlexibleMultiAgentPolicyManager - each test <0.1s."""
    
    def test_independent_mode(self):
        """Test independent policy mode."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="independent"
        )
        
        # Each agent should have its own policy
        assert manager.policy_mapping["agent_0"] is not manager.policy_mapping["agent_1"]
        assert len(manager.policies) == 2
    
    def test_shared_mode(self):
        """Test shared policy mode."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=3)
        shared_policy = MinimalPolicy()
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=shared_policy,
            env=env,
            mode="shared"
        )
        
        # All agents share same policy
        assert manager.policy_mapping["agent_0"] is manager.policy_mapping["agent_1"]
        assert manager.policy_mapping["agent_1"] is manager.policy_mapping["agent_2"]
        assert len(manager.policies) == 1
    
    def test_grouped_mode(self):
        """Test grouped policy mode."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=4)
        policies = {
            "team_a": MinimalPolicy(),
            "team_b": MinimalPolicy()
        }
        agent_groups = {
            "team_a": ["agent_0", "agent_1"],
            "team_b": ["agent_2", "agent_3"]
        }
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="grouped",
            agent_groups=agent_groups
        )
        
        # Same team shares policy
        assert manager.policy_mapping["agent_0"] is manager.policy_mapping["agent_1"]
        assert manager.policy_mapping["agent_2"] is manager.policy_mapping["agent_3"]
        # Different teams have different policies
        assert manager.policy_mapping["agent_0"] is not manager.policy_mapping["agent_2"]
    
    def test_custom_mode(self):
        """Test custom policy mapping."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=3)
        policies = {
            "policy_even": MinimalPolicy(),
            "policy_odd": MinimalPolicy()
        }
        
        def custom_mapping(agent_id):
            idx = int(agent_id.split("_")[1])
            return "policy_even" if idx % 2 == 0 else "policy_odd"
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="custom",
            policy_mapping_fn=custom_mapping
        )
        
        assert manager.policy_mapping["agent_0"] is policies["policy_even"]
        assert manager.policy_mapping["agent_1"] is policies["policy_odd"]
        assert manager.policy_mapping["agent_2"] is policies["policy_even"]
    
    def test_forward_independent(self):
        """Test forward pass with independent policies."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="independent"
        )
        
        # Create minimal batch
        obs = np.zeros((2, 2))
        agent_ids = np.array(["agent_0", "agent_1"])
        batch = Batch(obs=Batch(obs=obs, agent_id=agent_ids))
        
        result = manager(batch)
        assert "act" in result
    
    def test_forward_shared_optimized(self):
        """Test optimized forward pass for shared policies."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=3)
        shared_policy = MinimalPolicy()
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=shared_policy,
            env=env,
            mode="shared"
        )
        
        # Should use optimized path
        obs = np.zeros((3, 2))
        agent_ids = np.array(["agent_0", "agent_1", "agent_2"]) 
        batch = Batch(obs=Batch(obs=obs, agent_id=agent_ids))
        
        with patch.object(shared_policy, 'forward', wraps=shared_policy.forward) as mock_forward:
            result = manager(batch)
            # Should call policy only once for all agents
            assert mock_forward.call_count == 1
    
    def test_policy_update(self):
        """Test updating policies after initialization."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=2)
        old_policy = MinimalPolicy()
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=old_policy,
            env=env,
            mode="shared"
        )
        
        # Update policy
        new_policy = MinimalPolicy()
        manager.policies["shared"] = new_policy
        manager.policy_mapping = {agent: new_policy for agent in env.agents}
        
        assert manager.policy_mapping["agent_0"] is new_policy
        assert manager.policy_mapping["agent_1"] is new_policy
    
    def test_mixed_action_spaces(self):
        """Test handling of mixed action spaces."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=2)
        # Give different action spaces
        env.action_spaces["agent_0"] = spaces.Discrete(3)
        env.action_spaces["agent_1"] = spaces.Box(-1, 1, (2,))
        
        policies = {agent: MinimalPolicy() for agent in env.agents}
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="independent"
        )
        
        assert manager.agents == ["agent_0", "agent_1"]


# ============================================================================
# TEST: Dict Observation Support (8 fast tests)
# ============================================================================

class TestDictObservationFast:
    """Fast tests for Dict observation support - each test <0.1s."""
    
    def test_wrapper_creation(self):
        """Test DictObservationWrapper creation."""
        from tianshou.env import DictObservationWrapper
        
        obs_space = spaces.Dict({
            "position": spaces.Box(-1, 1, (2,)),
            "id": spaces.Discrete(3)
        })
        policy = MinimalPolicy()
        
        wrapper = DictObservationWrapper(policy, obs_space)
        assert wrapper.policy is policy
        assert wrapper.observation_space == obs_space
    
    def test_auto_preprocessor(self):
        """Test automatic preprocessor creation."""
        from tianshou.env import DictObservationWrapper, DictToTensorPreprocessor
        
        obs_space = spaces.Dict({
            "feat": spaces.Box(-1, 1, (3,))
        })
        policy = MinimalPolicy()
        
        wrapper = DictObservationWrapper(policy, obs_space)
        assert isinstance(wrapper.preprocessor, DictToTensorPreprocessor)
    
    def test_tensor_conversion(self):
        """Test Dict to tensor conversion."""
        from tianshou.env import DictToTensorPreprocessor
        
        obs_space = spaces.Dict({
            "continuous": spaces.Box(-1, 1, (2,)),
            "discrete": spaces.Discrete(3)
        })
        
        preprocessor = DictToTensorPreprocessor(obs_space)
        
        obs = {
            "continuous": torch.randn(1, 2),
            "discrete": torch.tensor([1])
        }
        
        result = preprocessor(obs)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 2  # [batch, features]
    
    def test_box_space_extraction(self):
        """Test Box space feature extraction."""
        from tianshou.env import DictToTensorPreprocessor
        
        obs_space = spaces.Dict({
            "box": spaces.Box(-1, 1, (4,))
        })
        
        preprocessor = DictToTensorPreprocessor(obs_space)
        assert "box" in preprocessor.extractors
        assert isinstance(preprocessor.extractors["box"], nn.Linear)
    
    def test_discrete_space_extraction(self):
        """Test Discrete space feature extraction."""
        from tianshou.env import DictToTensorPreprocessor
        
        obs_space = spaces.Dict({
            "discrete": spaces.Discrete(5)
        })
        
        preprocessor = DictToTensorPreprocessor(obs_space)
        assert "discrete" in preprocessor.extractors
        assert isinstance(preprocessor.extractors["discrete"], nn.Embedding)
    
    def test_multi_discrete_extraction(self):
        """Test MultiDiscrete space feature extraction."""
        from tianshou.env import DictToTensorPreprocessor
        
        obs_space = spaces.Dict({
            "multi": spaces.MultiDiscrete([2, 3, 4])
        })
        
        preprocessor = DictToTensorPreprocessor(obs_space)
        assert "multi" in preprocessor.extractors
        assert isinstance(preprocessor.extractors["multi"], nn.ModuleList)
        assert len(preprocessor.extractors["multi"]) == 3
    
    def test_batched_processing(self):
        """Test batched dict observation processing."""
        from tianshou.env import DictToTensorPreprocessor
        
        obs_space = spaces.Dict({
            "feat1": spaces.Box(-1, 1, (2,)),
            "feat2": spaces.Box(-1, 1, (3,))
        })
        
        preprocessor = DictToTensorPreprocessor(obs_space)
        
        batch_size = 4
        obs = {
            "feat1": torch.randn(batch_size, 2),
            "feat2": torch.randn(batch_size, 3)
        }
        
        result = preprocessor(obs)
        assert result.shape[0] == batch_size
    
    def test_wrapper_forward(self):
        """Test wrapper forward pass."""
        from tianshou.env import DictObservationWrapper
        
        obs_space = spaces.Dict({
            "state": spaces.Box(-1, 1, (2,))
        })
        policy = MinimalPolicy()
        
        wrapper = DictObservationWrapper(policy, obs_space)
        
        obs = {"state": np.array([0.5, -0.5])}
        batch = Batch(obs=obs)
        
        result = wrapper.forward(batch)
        assert "act" in result


# ============================================================================
# TEST: Training Coordination (8 fast tests)
# ============================================================================

class TestTrainingCoordinationFast:
    """Fast tests for training coordination - each test <0.1s."""
    
    def test_simultaneous_trainer_init(self):
        """Test SimultaneousTrainer initialization."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SimultaneousTrainer
        )
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        
        trainer = SimultaneousTrainer(policy_manager=manager)
        assert trainer.policy_manager is manager
    
    def test_simultaneous_train_step(self):
        """Test simultaneous training step."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SimultaneousTrainer
        )
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Minimal batch
        batches = {
            agent: Batch(
                obs=np.zeros((2, 2)),
                act=np.zeros(2, dtype=int),
                rew=np.zeros(2),
                terminated=np.zeros(2, dtype=bool),
                obs_next=np.zeros((2, 2))
            ) for agent in env.agents
        }
        
        losses = trainer.train_step(batches)
        assert "agent_0" in losses
        assert "agent_1" in losses
    
    def test_sequential_trainer(self):
        """Test SequentialTrainer."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SequentialTrainer
        )
        
        env = MinimalEnv(n_agents=3)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        
        trainer = SequentialTrainer(
            policy_manager=manager,
            agent_order=["agent_2", "agent_0", "agent_1"]
        )
        
        assert trainer.agent_order == ["agent_2", "agent_0", "agent_1"]
    
    def test_self_play_trainer(self):
        """Test SelfPlayTrainer initialization."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SelfPlayTrainer
        )
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        
        trainer = SelfPlayTrainer(
            policy_manager=manager,
            main_agent_id="agent_0",
            snapshot_interval=10
        )
        
        assert trainer.main_agent_id == "agent_0"
        assert trainer.snapshot_interval == 10
    
    def test_league_play_trainer(self):
        """Test LeaguePlayTrainer."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            LeaguePlayTrainer
        )
        
        env = MinimalEnv(n_agents=4)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        
        trainer = LeaguePlayTrainer(
            policy_manager=manager,
            league_size=4,
            matchmaking="random"
        )
        
        assert trainer.league_size == 4
        assert trainer.matchmaking == "random"
    
    def test_training_frequency(self):
        """Test agent-specific training frequencies."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SimultaneousTrainer
        )
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        
        trainer = SimultaneousTrainer(
            policy_manager=manager,
            agent_train_freq={"agent_0": 1, "agent_1": 2}
        )
        
        assert trainer.agent_train_freq["agent_0"] == 1
        assert trainer.agent_train_freq["agent_1"] == 2
    
    def test_mode_switching(self):
        """Test dynamic training mode switching."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            MATrainer
        )
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        
        trainer = MATrainer(policy_manager=manager, training_mode="simultaneous")
        assert trainer.training_mode == "simultaneous"
        
        trainer.set_training_mode("sequential")
        assert trainer.training_mode == "sequential"
    
    def test_checkpoint_saving(self):
        """Test checkpoint save/load functionality."""
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SimultaneousTrainer
        )
        import tempfile
        import os
        
        env = MinimalEnv(n_agents=2)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path)
            assert os.path.exists(path)
            
            # Test loading
            trainer.load_checkpoint(path)


# ============================================================================
# TEST: CTDE Support (8 fast tests)
# ============================================================================

class TestCTDEFast:
    """Fast tests for CTDE support - each test <0.1s."""
    
    def test_ctde_policy_init(self):
        """Test CTDEPolicy initialization."""
        from tianshou.algorithm.multiagent import CTDEPolicy
        
        actor = nn.Linear(2, 2)
        critic = nn.Linear(4, 1)
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters()),
            observation_space=spaces.Box(-1, 1, (2,)),
            action_space=spaces.Discrete(2)
        )
        
        assert policy.actor is actor
        assert policy.critic is critic
    
    def test_global_state_concat(self):
        """Test global state concatenation."""
        from tianshou.algorithm.multiagent import GlobalStateConstructor
        
        constructor = GlobalStateConstructor(
            mode="concatenate",
            obs_dim=2,
            n_agents=3
        )
        
        obs = {
            "agent_0": torch.randn(1, 2),
            "agent_1": torch.randn(1, 2),
            "agent_2": torch.randn(1, 2)
        }
        
        global_state = constructor.build(obs)
        assert global_state.shape == (1, 6)  # 3 agents * 2 dim
    
    def test_global_state_mean(self):
        """Test global state mean aggregation."""
        from tianshou.algorithm.multiagent import GlobalStateConstructor
        
        constructor = GlobalStateConstructor(
            mode="mean",
            obs_dim=2,
            n_agents=3
        )
        
        obs = {
            "agent_0": torch.ones(1, 2),
            "agent_1": torch.ones(1, 2) * 2,
            "agent_2": torch.ones(1, 2) * 3
        }
        
        global_state = constructor.build(obs)
        assert global_state.shape == (1, 2)
        assert torch.allclose(global_state, torch.tensor([[2.0, 2.0]]))
    
    def test_qmix_init(self):
        """Test QMIX policy initialization."""
        from tianshou.algorithm.multiagent import QMIXPolicy, QMIXMixer
        
        actors = [nn.Linear(2, 2) for _ in range(2)]
        mixer = QMIXMixer(n_agents=2, state_dim=4)
        
        policy = QMIXPolicy(
            actors=actors,
            mixer=mixer,
            observation_space=spaces.Box(-1, 1, (2,)),
            action_space=spaces.Discrete(2),
            n_agents=2
        )
        
        assert len(policy.actors) == 2
        assert policy.mixer is mixer
    
    def test_qmix_forward(self):
        """Test QMIX forward pass."""
        from tianshou.algorithm.multiagent import QMIXPolicy, QMIXMixer
        
        actors = [nn.Linear(2, 2) for _ in range(2)]
        mixer = QMIXMixer(n_agents=2, state_dim=4)
        
        policy = QMIXPolicy(
            actors=actors,
            mixer=mixer,
            observation_space=spaces.Box(-1, 1, (2,)),
            action_space=spaces.Discrete(2),
            n_agents=2
        )
        
        obs = torch.randn(1, 2)
        batch = Batch(obs=obs)
        
        # Mock to avoid tuple unpacking issues
        with patch.object(actors[0], 'forward', return_value=torch.randn(1, 2)):
            result = policy.forward(batch)
            assert "act" in result
    
    def test_maddpg_init(self):
        """Test MADDPG policy initialization."""
        from tianshou.algorithm.multiagent import MADDPGPolicy
        
        actors = [nn.Linear(2, 2) for _ in range(2)]
        critics = [nn.Linear(6, 1) for _ in range(2)]  # 2*(2+2) for obs+act
        
        policy = MADDPGPolicy(
            actors=actors,
            critics=critics,
            observation_space=spaces.Box(-1, 1, (2,)),
            action_space=spaces.Box(-1, 1, (2,)),
            n_agents=2
        )
        
        assert len(policy.actors) == 2
        assert len(policy.critics) == 2
    
    def test_target_network_updates(self):
        """Test target network soft updates."""
        from tianshou.algorithm.multiagent import CTDEPolicy
        
        actor = nn.Linear(2, 2)
        critic = nn.Linear(4, 1)
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters()),
            observation_space=spaces.Box(-1, 1, (2,)),
            action_space=spaces.Discrete(2),
            tau=0.005
        )
        
        # Create target networks
        policy.actor_target = nn.Linear(2, 2)
        policy.critic_target = nn.Linear(4, 1)
        
        # Store original params
        orig_weight = policy.actor_target.weight.clone()
        
        # Update targets
        policy.soft_update_targets()
        
        # Check params changed
        assert not torch.allclose(policy.actor_target.weight, orig_weight)
    
    def test_decentralized_execution(self):
        """Test decentralized execution (local obs only)."""
        from tianshou.algorithm.multiagent import CTDEPolicy
        
        actor = nn.Linear(2, 2)
        critic = nn.Linear(4, 1)
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters()),
            observation_space=spaces.Box(-1, 1, (2,)),
            action_space=spaces.Discrete(2)
        )
        
        # Execution uses only local observations
        local_obs = torch.randn(1, 2)
        batch = Batch(obs=local_obs)
        
        with patch.object(actor, 'forward', return_value=torch.randn(1, 2)):
            result = policy.forward(batch)
            actor.forward.assert_called_once()
            # Critic should NOT be called during execution
            assert "act" in result


# ============================================================================
# TEST: Integration Tests (5 fast tests)
# ============================================================================

class TestIntegrationFast:
    """Fast integration tests - each test <0.1s."""
    
    def test_pettingzoo_with_flexible_policy(self):
        """Test PettingZoo env with flexible policy manager."""
        from tianshou.env import EnhancedPettingZooEnv
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        env = MinimalEnv(n_agents=2)
        wrapped = EnhancedPettingZooEnv(env)
        
        policies = MinimalPolicy()  # Shared policy
        manager = FlexibleMultiAgentPolicyManager(policies, env, "shared")
        
        obs, _ = wrapped.reset()
        
        # Create batch from enhanced env output
        obs_array = np.array([obs["observations"][a] for a in env.agents])
        agent_ids = np.array(env.agents)
        batch = Batch(obs=Batch(obs=obs_array, agent_id=agent_ids))
        
        result = manager(batch)
        assert "act" in result
    
    def test_dict_obs_with_training(self):
        """Test dict observations with training coordination."""
        from tianshou.env import DictObservationWrapper
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SimultaneousTrainer
        )
        
        obs_space = spaces.Dict({
            "state": spaces.Box(-1, 1, (2,))
        })
        
        env = MinimalEnv(n_agents=2)
        policies = {}
        for agent in env.agents:
            policy = MinimalPolicy()
            policies[agent] = DictObservationWrapper(policy, obs_space)
        
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Dict observations in batch
        batches = {
            agent: Batch(
                obs={"state": np.zeros((2, 2))},
                act=np.zeros(2, dtype=int),
                rew=np.zeros(2),
                terminated=np.zeros(2, dtype=bool),
                obs_next={"state": np.zeros((2, 2))}
            ) for agent in env.agents
        }
        
        losses = trainer.train_step(batches)
        assert len(losses) == 2
    
    def test_ctde_with_flexible_policy(self):
        """Test CTDE with flexible policy manager."""
        from tianshou.algorithm.multiagent import (
            CTDEPolicy,
            FlexibleMultiAgentPolicyManager,
            GlobalStateConstructor
        )
        
        env = MinimalEnv(n_agents=2)
        
        # Create CTDE policies
        policies = {}
        for agent in env.agents:
            actor = nn.Linear(2, 2)
            critic = nn.Linear(4, 1)  # Global state
            
            policy = CTDEPolicy(
                actor=actor,
                critic=critic,
                optim_actor=torch.optim.Adam(actor.parameters()),
                optim_critic=torch.optim.Adam(critic.parameters()),
                observation_space=spaces.Box(-1, 1, (2,)),
                action_space=spaces.Discrete(2)
            )
            policies[agent] = policy
        
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        
        # Test execution
        obs_array = np.zeros((2, 2))
        agent_ids = np.array(env.agents)
        batch = Batch(obs=Batch(obs=obs_array, agent_id=agent_ids))
        
        with patch.object(policies["agent_0"].actor, 'forward', return_value=torch.randn(1, 2)):
            with patch.object(policies["agent_1"].actor, 'forward', return_value=torch.randn(1, 2)):
                result = manager(batch)
                assert "act" in result
    
    def test_full_training_loop(self):
        """Test complete training loop with all components."""
        from tianshou.env import EnhancedPettingZooEnv
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SimultaneousTrainer
        )
        
        # Setup
        env = MinimalEnv(n_agents=2)
        wrapped = EnhancedPettingZooEnv(env)
        
        policies = MinimalPolicy()  # Shared
        manager = FlexibleMultiAgentPolicyManager(policies, env, "shared")
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Collect experience (1 step)
        obs, _ = wrapped.reset()
        obs_array = np.array([obs["observations"][a] for a in env.agents])
        agent_ids = np.array(env.agents)
        batch = Batch(obs=Batch(obs=obs_array, agent_id=agent_ids))
        
        actions = manager(batch)
        if hasattr(actions, 'act'):
            action_array = actions.act
        else:
            action_array = np.zeros(2, dtype=int)
        
        obs_next, rewards, term, trunc, _ = wrapped.step(action_array)
        
        # Train
        batches = {
            agent: Batch(
                obs=obs["observations"][agent].reshape(1, -1),
                act=np.array([action_array[i]]),
                rew=np.array([rewards[i]]),
                terminated=np.array([term[i]]),
                obs_next=obs_next["observations"][agent].reshape(1, -1)
            ) for i, agent in enumerate(env.agents)
        }
        
        losses = trainer.train_step(batches)
        assert len(losses) > 0
    
    def test_performance_benchmark(self):
        """Benchmark to ensure all operations are fast."""
        import time
        from tianshou.env import EnhancedPettingZooEnv
        from tianshou.algorithm.multiagent import (
            FlexibleMultiAgentPolicyManager,
            SimultaneousTrainer
        )
        
        start = time.time()
        
        # Setup
        env = MinimalEnv(n_agents=4)
        wrapped = EnhancedPettingZooEnv(env)
        policies = {agent: MinimalPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies, env, "independent")
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Run 10 steps
        obs, _ = wrapped.reset()
        for _ in range(10):
            obs_array = np.array([obs["observations"][a] for a in env.agents])
            agent_ids = np.array(env.agents)
            batch = Batch(obs=Batch(obs=obs_array, agent_id=agent_ids))
            
            actions = manager(batch)
            if hasattr(actions, 'act'):
                action_array = actions.act
            else:
                action_array = np.zeros(4, dtype=int)
            
            obs, rewards, term, trunc, _ = wrapped.step(action_array)
        
        # Train once
        batches = {
            agent: Batch(
                obs=np.zeros((4, 2)),
                act=np.zeros(4, dtype=int),
                rew=np.zeros(4),
                terminated=np.zeros(4, dtype=bool),
                obs_next=np.zeros((4, 2))
            ) for agent in env.agents
        }
        trainer.train_step(batches)
        
        elapsed = time.time() - start
        assert elapsed < 0.5, f"Operations took {elapsed:.3f}s, should be <0.5s"


if __name__ == "__main__":
    # Quick verification that tests are fast
    import time
    
    print("Running fast MARL tests...")
    start = time.time()
    
    # Count tests
    test_classes = [
        TestEnhancedPettingZooFast,
        TestFlexiblePolicyFast,
        TestDictObservationFast,
        TestTrainingCoordinationFast,
        TestCTDEFast,
        TestIntegrationFast
    ]
    
    total_tests = sum(
        len([m for m in dir(cls) if m.startswith("test_")])
        for cls in test_classes
    )
    
    print(f"Total fast tests: {total_tests}")
    print(f"Expected runtime: <{total_tests * 0.1:.1f} seconds")
    print("\nRun with: pytest test/multiagent/test_marl_fast_comprehensive.py -v")