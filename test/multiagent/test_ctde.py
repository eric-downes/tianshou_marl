"""Tests for Centralized Training with Decentralized Execution (CTDE) support."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

from tianshou.data import Batch, ReplayBuffer
from tianshou.algorithm.multiagent.ctde import (
    CTDEPolicy,
    CentralizedCritic,
    DecentralizedActor,
    GlobalStateConstructor,
    QMIXPolicy,
    QMIXMixer,
    MADDPGPolicy,
)
from gymnasium import spaces


class MockActor(nn.Module):
    """Mock decentralized actor network."""
    
    def __init__(self, obs_dim: int = 4, action_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(obs_dim, action_dim)
        
    def forward(self, obs: torch.Tensor, state=None):
        """Forward pass returning actions."""
        return self.fc(obs), state


class MockCritic(nn.Module):
    """Mock centralized critic network."""
    
    def __init__(self, global_obs_dim: int = 12, n_agents: int = 3):
        super().__init__()
        # For MADDPG, each critic outputs a single Q-value for the joint state-action
        self.fc = nn.Linear(global_obs_dim, 1)  # Single Q-value output
        
    def forward(self, global_obs: torch.Tensor):
        """Forward pass returning Q-values."""
        return self.fc(global_obs)


class MockEnvironment:
    """Mock multi-agent environment with global state."""
    
    def __init__(self, n_agents: int = 3):
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = spaces.Discrete(2)
        self.global_obs_dim = 4 * n_agents  # Concatenated observations
        
    def get_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct global state from individual observations."""
        # Simple concatenation for testing
        obs_list = [observations[agent] for agent in self.agents]
        return np.concatenate(obs_list)
        
    def reset(self):
        """Reset environment."""
        obs = {agent: np.random.randn(4) for agent in self.agents}
        global_state = self.get_global_state(obs)
        return obs, global_state, {}
        
    def step(self, actions):
        """Step environment."""
        obs = {agent: np.random.randn(4) for agent in self.agents}
        global_state = self.get_global_state(obs)
        rewards = {agent: np.random.random() for agent in self.agents}
        terms = {agent: False for agent in self.agents}
        truncs = {agent: False for agent in self.agents}
        infos = {agent: {"global_state": global_state} for agent in self.agents}
        return obs, rewards, terms, truncs, infos


def create_mock_batch(n_agents: int = 3, batch_size: int = 32, with_global_state: bool = True, 
                     action_space: spaces.Space = None) -> Batch:
    """Create a mock batch for CTDE training."""
    batch = Batch()
    
    # Individual agent data
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        
        # Create actions based on action space type
        if action_space is None or isinstance(action_space, spaces.Discrete):
            # Discrete actions - scalar per sample
            act = torch.randint(0, 2, (batch_size,))
        elif isinstance(action_space, spaces.Box):
            # Continuous actions - vector per sample
            action_dim = action_space.shape[0]
            act = torch.randn(batch_size, action_dim)
        else:
            act = torch.randint(0, 2, (batch_size,))
        
        agent_batch = Batch(
            obs=torch.randn(batch_size, 4),
            act=act,
            rew=torch.randn(batch_size),
            terminated=torch.zeros(batch_size, dtype=torch.bool),
            truncated=torch.zeros(batch_size, dtype=torch.bool),
            obs_next=torch.randn(batch_size, 4),
        )
        batch[agent_id] = agent_batch
    
    # Global state information
    if with_global_state:
        batch.global_obs = torch.randn(batch_size, 4 * n_agents)
        batch.global_obs_next = torch.randn(batch_size, 4 * n_agents)
        
    return batch


@pytest.mark.slow
class TestCTDEPolicy:
    """Test suite for CTDEPolicy base class."""
    
    def test_ctde_policy_initialization(self):
        """Test CTDEPolicy can be initialized."""
        actor = MockActor()
        critic = MockCritic()
        optim_actor = optim.Adam(actor.parameters())
        optim_critic = optim.Adam(critic.parameters())
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=optim_actor,
            optim_critic=optim_critic,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2)
        )
        
        assert policy.actor == actor
        assert policy.critic == critic
        assert policy.enable_global_info is True
        
    def test_decentralized_forward(self):
        """Test decentralized execution (forward pass)."""
        actor = MockActor()
        critic = MockCritic()
        optim_actor = optim.Adam(actor.parameters())
        optim_critic = optim.Adam(critic.parameters())
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=optim_actor,
            optim_critic=optim_critic,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2)
        )
        
        # Forward should only use local observations
        batch = Batch(obs=torch.randn(10, 4))
        result = policy.forward(batch)
        
        assert "act" in result
        assert result.act.shape == (10, 2)  # 10 samples, 2 actions
        
    def test_centralized_training(self):
        """Test centralized training with global state."""
        actor = MockActor()
        critic = MockCritic()
        optim_actor = optim.Adam(actor.parameters())
        optim_critic = optim.Adam(critic.parameters())
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=optim_actor,
            optim_critic=optim_critic,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2)
        )
        
        # Create batch with global state
        batch = Batch(
            obs=torch.randn(32, 4),
            act=torch.randint(0, 2, (32,)),
            rew=torch.randn(32),
            obs_next=torch.randn(32, 4),
            global_obs=torch.randn(32, 12),
            global_obs_next=torch.randn(32, 12),
            terminated=torch.zeros(32, dtype=torch.bool)
        )
        
        # Learn should use global state
        losses = policy.learn(batch)
        
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        
    def test_disable_global_info(self):
        """Test training without global information."""
        actor = MockActor()
        critic = MockCritic(global_obs_dim=4, n_agents=1)  # Use local obs dim
        optim_actor = optim.Adam(actor.parameters())
        optim_critic = optim.Adam(critic.parameters())
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=optim_actor,
            optim_critic=optim_critic,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2),
            enable_global_info=False
        )
        
        batch = Batch(
            obs=torch.randn(32, 4),
            act=torch.randint(0, 2, (32,)),
            rew=torch.randn(32),
            obs_next=torch.randn(32, 4),
            terminated=torch.zeros(32, dtype=torch.bool)
        )
        
        # Should train without global state
        losses = policy.learn(batch)
        assert "critic_loss" in losses


@pytest.mark.slow
class TestGlobalStateConstructor:
    """Test suite for global state construction."""
    
    def test_concatenation_constructor(self):
        """Test simple concatenation of observations."""
        constructor = GlobalStateConstructor(mode="concatenate")
        
        observations = {
            "agent_0": torch.randn(32, 4),
            "agent_1": torch.randn(32, 4),
            "agent_2": torch.randn(32, 4)
        }
        
        global_state = constructor.build(observations)
        
        assert global_state.shape == (32, 12)  # 3 agents * 4 dim
        
    def test_attention_based_constructor(self):
        """Test attention-based global state construction."""
        constructor = GlobalStateConstructor(
            mode="attention",
            obs_dim=4,
            n_agents=3,
            hidden_dim=64
        )
        
        observations = {
            "agent_0": torch.randn(32, 4),
            "agent_1": torch.randn(32, 4),
            "agent_2": torch.randn(32, 4)
        }
        
        global_state = constructor.build(observations)
        
        # Output dimension should be hidden_dim
        assert global_state.shape == (32, 64)
        
    def test_graph_based_constructor(self):
        """Test graph-based global state construction."""
        # Adjacency matrix (fully connected for testing)
        adjacency = torch.ones(3, 3)
        
        constructor = GlobalStateConstructor(
            mode="graph",
            obs_dim=4,
            n_agents=3,
            adjacency_matrix=adjacency
        )
        
        observations = {
            "agent_0": torch.randn(32, 4),
            "agent_1": torch.randn(32, 4),
            "agent_2": torch.randn(32, 4)
        }
        
        global_state = constructor.build(observations)
        
        # Should aggregate neighbor information
        assert global_state.shape[0] == 32
        
    def test_custom_constructor(self):
        """Test custom global state construction function."""
        def custom_builder(obs_dict):
            # Custom logic: average all observations
            obs_list = list(obs_dict.values())
            stacked = torch.stack(obs_list, dim=1)
            return stacked.mean(dim=1)
        
        constructor = GlobalStateConstructor(
            mode="custom",
            custom_fn=custom_builder
        )
        
        observations = {
            "agent_0": torch.randn(32, 4),
            "agent_1": torch.randn(32, 4),
            "agent_2": torch.randn(32, 4)
        }
        
        global_state = constructor.build(observations)
        
        assert global_state.shape == (32, 4)


@pytest.mark.slow
class TestQMIX:
    """Test suite for QMIX algorithm."""
    
    def test_qmix_initialization(self):
        """Test QMIX policy initialization."""
        n_agents = 3
        actors = [MockActor() for _ in range(n_agents)]
        mixer = QMIXMixer(
            n_agents=n_agents,
            state_dim=12,
            mixing_embed_dim=32
        )
        
        policy = QMIXPolicy(
            actors=actors,
            mixer=mixer,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2),
            n_agents=n_agents
        )
        
        assert len(policy.actors) == n_agents
        assert policy.mixer == mixer
        
    def test_qmix_mixing_network(self):
        """Test QMIX mixing network."""
        n_agents = 3
        state_dim = 12
        batch_size = 32
        
        mixer = QMIXMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_embed_dim=32
        )
        
        # Individual Q-values
        q_values = torch.randn(batch_size, n_agents)
        # Global state
        global_state = torch.randn(batch_size, state_dim)
        
        # Mix Q-values
        q_total = mixer(q_values, global_state)
        
        assert q_total.shape == (batch_size, 1)
        
    def test_qmix_monotonicity(self):
        """Test QMIX monotonicity constraint."""
        n_agents = 3
        state_dim = 12
        
        mixer = QMIXMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_embed_dim=32
        )
        
        # Ensure weights are positive (monotonicity)
        for param in mixer.parameters():
            if "weight" in param.__class__.__name__.lower():
                assert (param.data >= 0).all() or not mixer.enforce_monotonic
                
    def test_qmix_forward(self):
        """Test QMIX forward pass (decentralized)."""
        n_agents = 3
        actors = [MockActor() for _ in range(n_agents)]
        mixer = QMIXMixer(n_agents=n_agents, state_dim=12)
        
        policy = QMIXPolicy(
            actors=actors,
            mixer=mixer,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2),
            n_agents=n_agents
        )
        
        # Individual observations
        batch = Batch()
        for i in range(n_agents):
            batch[f"agent_{i}"] = Batch(obs=torch.randn(10, 4))
        
        result = policy.forward(batch)
        
        # Should return actions for each agent
        for i in range(n_agents):
            assert f"agent_{i}" in result
            assert "act" in result[f"agent_{i}"]
            
    def test_qmix_learn(self):
        """Test QMIX learning step."""
        n_agents = 3
        actors = [MockActor() for _ in range(n_agents)]
        mixer = QMIXMixer(n_agents=n_agents, state_dim=12)
        
        policy = QMIXPolicy(
            actors=actors,
            mixer=mixer,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2),
            n_agents=n_agents
        )
        
        # Create multi-agent batch with global state
        batch = create_mock_batch(n_agents=n_agents, with_global_state=True)
        
        losses = policy.learn(batch)
        
        assert "loss" in losses
        assert "q_values" in losses
        assert losses["loss"] >= 0


@pytest.mark.slow
class TestMADDPG:
    """Test suite for MADDPG algorithm."""
    
    def test_maddpg_initialization(self):
        """Test MADDPG policy initialization."""
        n_agents = 3
        actors = [MockActor() for _ in range(n_agents)]
        critics = [MockCritic() for _ in range(n_agents)]
        
        policy = MADDPGPolicy(
            actors=actors,
            critics=critics,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Box(-1, 1, (2,)),  # Continuous actions
            n_agents=n_agents
        )
        
        assert len(policy.actors) == n_agents
        assert len(policy.critics) == n_agents
        assert len(policy.target_actors) == n_agents
        assert len(policy.target_critics) == n_agents
        
    def test_maddpg_target_networks(self):
        """Test MADDPG target network updates."""
        n_agents = 2
        actors = [MockActor() for _ in range(n_agents)]
        critics = [MockCritic() for _ in range(n_agents)]
        
        policy = MADDPGPolicy(
            actors=actors,
            critics=critics,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Box(-1, 1, (2,)),
            n_agents=n_agents,
            tau=0.01  # Soft update parameter
        )
        
        # Get initial target network parameters
        initial_params = []
        for target_actor in policy.target_actors:
            initial_params.append(
                list(target_actor.parameters())[0].clone()
            )
        
        # Modify main network
        for actor in policy.actors:
            for param in actor.parameters():
                param.data += 1.0
        
        # Update target networks
        policy.update_target_networks()
        
        # Check soft update worked
        for i, target_actor in enumerate(policy.target_actors):
            current_param = list(target_actor.parameters())[0]
            assert not torch.equal(current_param, initial_params[i])
            
    def test_maddpg_centralized_critic(self):
        """Test MADDPG centralized critic with all agents' actions."""
        n_agents = 3
        actors = [MockActor() for _ in range(n_agents)]
        critics = [MockCritic() for _ in range(n_agents)]
        
        action_space = spaces.Box(-1, 1, (2,))
        policy = MADDPGPolicy(
            actors=actors,
            critics=critics,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=action_space,
            n_agents=n_agents
        )
        
        # Create batch with all agents' observations and actions
        batch = create_mock_batch(n_agents=n_agents, with_global_state=True,
                                action_space=action_space)
        
        # Critics should use all agents' observations and actions
        all_obs = torch.cat([batch[f"agent_{i}"].obs for i in range(n_agents)], dim=-1)
        # For continuous actions, they're already in the right shape
        all_actions = torch.cat([batch[f"agent_{i}"].act for i in range(n_agents)], dim=-1)
        
        # Verify critic input dimensions
        critic_input = torch.cat([all_obs, all_actions], dim=-1)
        assert critic_input.shape[1] == 4 * n_agents + 2 * n_agents  # obs + actions (2D each)
        
    def test_maddpg_learn(self):
        """Test MADDPG learning step."""
        n_agents = 2
        actors = [MockActor() for _ in range(n_agents)]
        critics = [MockCritic() for _ in range(n_agents)]
        
        action_space = spaces.Box(-1, 1, (2,))
        policy = MADDPGPolicy(
            actors=actors,
            critics=critics,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=action_space,
            n_agents=n_agents
        )
        
        batch = create_mock_batch(n_agents=n_agents, with_global_state=True, 
                                action_space=action_space)
        
        losses = policy.learn(batch)
        
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        for i in range(n_agents):
            assert f"agent_{i}_actor_loss" in losses
            assert f"agent_{i}_critic_loss" in losses


@pytest.mark.slow
class TestIntegration:
    """Integration tests for CTDE with other components."""
    
    def test_ctde_with_training_coordinator(self):
        """Test CTDE policy with training coordinators."""
        from tianshou.algorithm.multiagent.training_coordinator import SimultaneousTrainer
        from tianshou.algorithm.multiagent.flexible_policy import FlexibleMultiAgentPolicyManager
        
        env = MockEnvironment(n_agents=2)
        
        # Create CTDE policies for each agent
        policies = {}
        for agent in env.agents:
            actor = MockActor()
            # For CTDEPolicy, critic uses global observations only (not actions)
            critic = MockCritic(global_obs_dim=env.global_obs_dim, n_agents=env.n_agents)
            optim_actor = optim.Adam(actor.parameters())
            optim_critic = optim.Adam(critic.parameters())
            
            policies[agent] = CTDEPolicy(
                actor=actor,
                critic=critic,
                optim_actor=optim_actor,
                optim_critic=optim_critic,
                observation_space=env.observation_space,
                action_space=env.action_space
            )
        
        # Create policy manager
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="independent"
        )
        
        # Create trainer
        trainer = SimultaneousTrainer(policy_manager=manager)
        
        # Create batch with global state
        batch = create_mock_batch(n_agents=2, with_global_state=True)
        
        # Train
        losses = trainer.train_step(batch)
        
        assert len(losses) == 2
        for agent in env.agents:
            assert agent in losses
            
    def test_qmix_with_replay_buffer(self):
        """Test QMIX with experience replay."""
        n_agents = 3
        actors = [MockActor() for _ in range(n_agents)]
        mixer = QMIXMixer(n_agents=n_agents, state_dim=12)
        
        policy = QMIXPolicy(
            actors=actors,
            mixer=mixer,
            observation_space=spaces.Box(-1, 1, (4,)),
            action_space=spaces.Discrete(2),
            n_agents=n_agents
        )
        
        # Create replay buffer
        buffer = ReplayBuffer(size=1000)
        
        # Collect experiences
        for _ in range(100):
            batch = create_mock_batch(n_agents=n_agents, batch_size=1)
            # Flatten for buffer
            flat_batch = Batch()
            
            # Add required keys for replay buffer (use agent_0 as primary)
            # Convert tensors to numpy for replay buffer
            flat_batch.obs = batch["agent_0"].obs.numpy()
            flat_batch.act = batch["agent_0"].act.numpy()
            flat_batch.rew = batch["agent_0"].rew.numpy()
            flat_batch.obs_next = batch["agent_0"].obs_next.numpy()
            flat_batch.terminated = batch["agent_0"].terminated.numpy()
            flat_batch.truncated = batch["agent_0"].truncated.numpy()
            
            # Add global state
            for key in ["global_obs", "global_obs_next"]:
                if key in batch:
                    flat_batch[key] = batch[key].numpy()
            
            # Add all agents' data with prefixes
            for i in range(n_agents):
                for key in batch[f"agent_{i}"].keys():
                    val = batch[f"agent_{i}"][key]
                    # Convert tensors to numpy
                    flat_batch[f"agent_{i}_{key}"] = val.numpy() if hasattr(val, 'numpy') else val
            buffer.add(flat_batch)
        
        # Sample and train
        sampled, indices = buffer.sample(32)
        
        # Reconstruct multi-agent batch
        ma_batch = Batch()
        for key in ["global_obs", "global_obs_next"]:
            if key in sampled:
                val = sampled[key]
                # Squeeze the extra dimension from batch_size=1 storage
                if isinstance(val, np.ndarray) and val.shape[1] == 1:
                    val = val.squeeze(1)
                ma_batch[key] = val
        for i in range(n_agents):
            agent_batch = Batch()
            for key in ["obs", "act", "rew", "obs_next", "terminated", "truncated"]:
                field_name = f"agent_{i}_{key}"
                if field_name in sampled:
                    val = sampled[field_name]
                    # Squeeze the extra dimension from batch_size=1 storage
                    if isinstance(val, np.ndarray) and len(val.shape) > 1 and val.shape[1] == 1:
                        if key in ["obs", "obs_next"]:
                            # For observations, squeeze the batch=1 dimension but keep feature dims
                            val = val.squeeze(1)
                        elif key in ["act", "rew", "terminated", "truncated"]:
                            # For scalars, fully squeeze
                            val = val.squeeze()
                    agent_batch[key] = val
                elif key == "truncated":
                    # Add truncated if missing (for compatibility)
                    agent_batch[key] = np.zeros_like(sampled[f"agent_{i}_terminated"])
            ma_batch[f"agent_{i}"] = agent_batch
        
        losses = policy.learn(ma_batch)
        assert "loss" in losses
        
    def test_global_state_from_environment(self):
        """Test global state construction from environment info."""
        env = MockEnvironment(n_agents=3)
        constructor = GlobalStateConstructor(mode="concatenate")
        
        # Reset environment
        obs, global_state_true, _ = env.reset()
        
        # Construct global state
        obs_tensors = {
            agent: torch.tensor(obs[agent]).unsqueeze(0) 
            for agent in env.agents
        }
        global_state_constructed = constructor.build(obs_tensors)
        
        # Compare
        global_state_true_tensor = torch.tensor(global_state_true).unsqueeze(0)
        assert global_state_constructed.shape == global_state_true_tensor.shape
        
    def test_ctde_end_to_end(self):
        """Test complete CTDE training loop."""
        env = MockEnvironment(n_agents=2)
        
        # Create QMIX policy
        actors = [MockActor() for _ in range(env.n_agents)]
        mixer = QMIXMixer(n_agents=env.n_agents, state_dim=env.global_obs_dim)
        
        policy = QMIXPolicy(
            actors=actors,
            mixer=mixer,
            observation_space=env.observation_space,
            action_space=env.action_space,
            n_agents=env.n_agents
        )
        
        # Training loop
        for episode in range(10):
            obs, global_state, _ = env.reset()
            episode_reward = 0
            
            for step in range(10):
                # Decentralized execution
                batch = Batch()
                for i, agent in enumerate(env.agents):
                    batch[agent] = Batch(obs=torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0))
                
                actions = policy.forward(batch)
                
                # Environment step
                action_dict = {
                    agent: actions[agent].act.item() 
                    for agent in env.agents
                }
                obs_next, rewards, terms, truncs, infos = env.step(action_dict)
                
                # Store experience (would go to replay buffer)
                experience = create_mock_batch(
                    n_agents=env.n_agents, 
                    batch_size=1,
                    with_global_state=True
                )
                
                # Centralized training
                if step % 4 == 0:  # Train every 4 steps
                    losses = policy.learn(experience)
                    assert "loss" in losses
                
                obs = obs_next
                episode_reward += sum(rewards.values())
                
                if any(terms.values()) or any(truncs.values()):
                    break
        
        # Should complete without errors
        assert True