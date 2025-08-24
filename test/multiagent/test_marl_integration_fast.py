"""Fast integration tests for MARL functionality.

These tests cover critical MARL features and run in <30 seconds total.
They are NOT marked with @pytest.mark.slow to ensure they run in default test suite.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
from unittest.mock import Mock

from tianshou.data import Batch
from tianshou.env import EnhancedPettingZooEnv
from tianshou.env.dict_observation import DictObservationWrapper, DictToTensorPreprocessor
from tianshou.algorithm.multiagent import (
    FlexibleMultiAgentPolicyManager,
    SimultaneousTrainer,
    CTDEPolicy,
    GlobalStateConstructor
)
from tianshou.algorithm.algorithm_base import Policy
from gymnasium import spaces


class FastMockPolicy(Policy):
    """Lightweight mock policy for fast tests."""
    
    def __init__(self, observation_space=None, action_space=None):
        if action_space is None:
            action_space = spaces.Discrete(2)
        if observation_space is None:
            observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        super().__init__(observation_space=observation_space, action_space=action_space)
        
    def forward(self, batch: Batch, state=None, **kwargs):
        if hasattr(batch.obs, 'shape'):
            batch_size = batch.obs.shape[0]
        else:
            batch_size = 1
        return Batch(act=np.zeros(batch_size, dtype=int), state=state)
    
    def learn(self, batch: Batch, **kwargs):
        return {"loss": 0.0}


class FastMockEnv:
    """Lightweight mock environment for fast tests."""
    
    def __init__(self, n_agents=2, parallel=True):
        self.n_agents = n_agents
        self.parallel = parallel
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents.copy()
        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(4,))
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(2)
            for agent in self.agents
        }
        self.metadata = {"is_parallelizable": parallel}
        
    def reset(self, *args, **kwargs):
        obs = {agent: np.zeros(4) for agent in self.agents}
        info = {agent: {} for agent in self.agents}
        return obs, info
        
    def step(self, actions):
        obs = {agent: np.zeros(4) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        term = {agent: False for agent in self.agents}
        trunc = {agent: False for agent in self.agents}
        info = {agent: {} for agent in self.agents}
        return obs, rewards, term, trunc, info


class TestFastMARL:
    """Fast integration tests for MARL - run in <30 seconds total."""
    
    def test_enhanced_pettingzoo_env_basic(self):
        """Test basic EnhancedPettingZooEnv functionality - fast version."""
        env = FastMockEnv(n_agents=2, parallel=True)
        wrapped = EnhancedPettingZooEnv(env)
        
        # Quick test of core functionality
        obs, info = wrapped.reset()
        # Enhanced env returns dict with observations, masks, agent_ids
        assert "observations" in obs
        assert len(obs["observations"]) == 2
        
        actions = np.array([0, 1])
        obs, rewards, term, trunc, info = wrapped.step(actions)
        assert "observations" in obs
        assert len(obs["observations"]) == 2
        assert len(rewards) == 2
        
    def test_flexible_policy_shared_mode(self):
        """Test shared policy mode - fast version."""
        env = FastMockEnv(n_agents=3)
        shared_policy = FastMockPolicy()
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=shared_policy,
            env=env,
            mode="shared"
        )
        
        # Verify all agents use same policy instance
        assert manager.policy_mapping["agent_0"] is manager.policy_mapping["agent_1"]
        assert manager.policy_mapping["agent_1"] is manager.policy_mapping["agent_2"]
        
        # Quick forward pass - need proper batch format
        obs_array = np.zeros((3, 4))  # [n_agents, obs_dim]
        agent_ids = np.array(["agent_0", "agent_1", "agent_2"])
        batch = Batch(obs=Batch(obs=obs_array, agent_id=agent_ids))
        actions = manager(batch)
        assert "act" in actions
        
    def test_dict_observation_processing(self):
        """Test dict observation handling - fast version."""
        # Simple dict observation space
        obs_space = spaces.Dict({
            "position": spaces.Box(low=-1, high=1, shape=(2,)),
            "state": spaces.Discrete(3)
        })
        
        policy = FastMockPolicy()
        wrapper = DictObservationWrapper(policy, obs_space)
        
        # Single observation
        obs = {
            "position": np.array([0.5, -0.5]),
            "state": np.array(1)
        }
        batch = Batch(obs=obs)
        result = wrapper.forward(batch)
        assert "act" in result
        
    def test_training_coordinator_basic(self):
        """Test basic training coordination - fast version."""
        env = FastMockEnv(n_agents=2)
        policies = {
            "agent_0": FastMockPolicy(),
            "agent_1": FastMockPolicy()
        }
        
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="independent"
        )
        
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Quick training step - format batch per agent
        batches = {}
        for agent in env.agents:
            batches[agent] = Batch(
                obs=np.zeros((10, 4)),
                act=np.zeros(10, dtype=int),
                rew=np.zeros(10),
                terminated=np.zeros(10, dtype=bool),
                truncated=np.zeros(10, dtype=bool),
                obs_next=np.zeros((10, 4)),
                info=[{} for _ in range(10)]
            )
        
        losses = trainer.train_step(batches)
        assert "agent_0" in losses
        assert "agent_1" in losses
        
    def test_ctde_basic_functionality(self):
        """Test basic CTDE functionality - fast version."""
        # Simple actor and critic
        class SimpleActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)
            
            def forward(self, obs, state=None):
                return self.fc(obs), state
        
        actor = SimpleActor()
        critic = nn.Linear(8, 1)  # Global state dimension
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters()),
            observation_space=spaces.Box(low=-1, high=1, shape=(4,)),
            action_space=spaces.Discrete(2)
        )
        
        # Quick forward pass (decentralized)
        local_obs = torch.randn(1, 4)
        batch = Batch(obs=local_obs)
        result = policy.forward(batch)
        assert "act" in result
        
        # Quick learn step (centralized)
        batch_train = Batch(
            obs=torch.randn(10, 4),
            act=torch.randint(0, 2, (10,)),
            rew=torch.randn(10),
            terminated=torch.zeros(10, dtype=torch.bool),
            truncated=torch.zeros(10, dtype=torch.bool),
            obs_next=torch.randn(10, 4),
            global_obs=torch.randn(10, 8),  # Global state for critic
            global_obs_next=torch.randn(10, 8)  # Next global state for critic
        )
        
        losses = policy.learn(batch_train)
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        
    def test_global_state_construction(self):
        """Test global state construction - fast version."""
        constructor = GlobalStateConstructor(
            mode="concatenate",
            obs_dim=4,
            n_agents=2
        )
        
        observations = {
            "agent_0": torch.randn(10, 4),
            "agent_1": torch.randn(10, 4)
        }
        
        global_state = constructor.build(observations)
        assert global_state.shape == (10, 8)  # Concatenated
        
    def test_integration_dict_obs_with_ctde(self):
        """Test Dict observations with CTDE - fast integration."""
        # Dict observation space
        obs_space = spaces.Dict({
            "local": spaces.Box(low=-1, high=1, shape=(3,)),
            "id": spaces.Discrete(2)
        })
        
        # Create preprocessor
        preprocessor = DictToTensorPreprocessor(obs_space)
        
        # Simple CTDE setup
        actor = nn.Sequential(
            nn.Linear(preprocessor.output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        critic = nn.Linear(preprocessor.output_dim * 2, 1)  # For 2 agents
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters()),
            observation_space=obs_space,
            action_space=spaces.Discrete(2)
        )
        
        # Process dict observation
        obs_dict = {
            "local": torch.randn(1, 3),
            "id": torch.tensor([0])
        }
        processed = preprocessor(obs_dict)
        
        # Forward through policy
        batch = Batch(obs=processed)
        result = policy.forward(batch)
        assert "act" in result
        
    def test_multi_agent_workflow(self):
        """Test complete multi-agent workflow - fast version."""
        # Setup environment
        env = FastMockEnv(n_agents=2, parallel=True)
        wrapped_env = EnhancedPettingZooEnv(env)
        
        # Setup policies with parameter sharing
        shared_policy = FastMockPolicy()
        manager = FlexibleMultiAgentPolicyManager(
            policies=shared_policy,
            env=env,
            mode="shared"
        )
        
        # Setup trainer
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Quick episode
        obs, info = wrapped_env.reset()
        for _ in range(3):  # Just 3 steps for speed
            # Extract observations from the enhanced env format
            obs_array = np.array([obs["observations"][agent] for agent in env.agents])
            agent_ids = np.array(env.agents)
            batch = Batch(obs=Batch(obs=obs_array, agent_id=agent_ids))
            actions = manager(batch)
            
            # Convert actions to proper format
            if hasattr(actions, 'act'):
                action_array = actions.act
            else:
                action_array = np.array([0, 0])  # Default actions
                
            obs, rewards, term, trunc, info = wrapped_env.step(action_array)
            
        # Quick training - format per agent
        batches = {}
        for agent in env.agents:
            batches[agent] = Batch(
                obs=np.zeros((5, 4)),
                act=np.zeros(5, dtype=int),
                rew=np.zeros(5),
                terminated=np.zeros(5, dtype=bool),
                truncated=np.zeros(5, dtype=bool),
                obs_next=np.zeros((5, 4)),
                info=[{} for _ in range(5)]
            )
        
        losses = trainer.train_step(batches)
        assert len(losses) > 0
        
    def test_performance_critical_path(self):
        """Test performance of critical MARL path - must be <1 second."""
        import time
        
        start = time.time()
        
        # Setup
        env = FastMockEnv(n_agents=4)
        policies = {agent: FastMockPolicy() for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="independent"
        )
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Run 100 forward passes
        for _ in range(100):
            obs_array = np.zeros((4, 4))  # [n_agents, obs_dim]
            agent_ids = np.array(env.agents)
            batch = Batch(obs=Batch(obs=obs_array, agent_id=agent_ids))
            actions = manager(batch)
            
        # Run 10 training steps
        for _ in range(10):
            batches = {}
            for agent in env.agents:
                batches[agent] = Batch(
                    obs=np.zeros((32, 4)),
                    act=np.zeros(32, dtype=int),
                    rew=np.zeros(32),
                    terminated=np.zeros(32, dtype=bool),
                    truncated=np.zeros(32, dtype=bool),
                    obs_next=np.zeros((32, 4)),
                    info=[{} for _ in range(32)]
                )
            losses = trainer.train_step(batches)
            
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Critical path took {elapsed:.2f}s, should be <1s"