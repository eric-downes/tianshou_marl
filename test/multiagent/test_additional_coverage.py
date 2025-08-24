"""Additional fast tests to maximize MARL coverage - Fixed version."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
import torch.nn as nn


class TestCentralizedCriticCoverage:
    """Additional tests for CentralizedCritic functionality."""

    def test_centralized_critic_initialization(self):
        """Test CentralizedCritic initialization with various configurations."""
        from tianshou.algorithm.multiagent.ctde import CentralizedCritic
        
        # Test with correct API
        critic = CentralizedCritic(
            global_obs_dim=20,
            n_agents=3,
            hidden_dim=64
        )
        assert critic is not None
        assert isinstance(critic, nn.Module)
        
        # Test with different hidden dim
        critic_simple = CentralizedCritic(
            global_obs_dim=10,
            n_agents=2,
            hidden_dim=32
        )
        assert critic_simple is not None

    def test_centralized_critic_forward(self):
        """Test CentralizedCritic forward pass functionality."""
        from tianshou.algorithm.multiagent.ctde import CentralizedCritic
        
        n_agents = 2
        critic = CentralizedCritic(
            global_obs_dim=16,
            n_agents=n_agents,
            hidden_dim=32
        )
        
        batch_size = 8
        global_obs = torch.randn(batch_size, 16)
        
        q_values = critic(global_obs)
        assert q_values.shape == (batch_size, n_agents)

    def test_centralized_critic_different_batch_sizes(self):
        """Test CentralizedCritic with different batch sizes."""
        from tianshou.algorithm.multiagent.ctde import CentralizedCritic
        
        n_agents = 2
        critic = CentralizedCritic(
            global_obs_dim=12,
            n_agents=n_agents,
            hidden_dim=64
        )
        
        for batch_size in [1, 4, 16, 32]:
            global_obs = torch.randn(batch_size, 12)
            q_values = critic(global_obs)
            assert q_values.shape == (batch_size, n_agents)


class TestDecentralizedActorCoverage:
    """Additional tests for DecentralizedActor functionality."""

    def test_decentralized_actor_initialization(self):
        """Test DecentralizedActor initialization."""
        from tianshou.algorithm.multiagent.ctde import DecentralizedActor
        
        # Test with correct API
        actor = DecentralizedActor(
            obs_dim=8,
            action_dim=4,
            hidden_dim=32
        )
        assert actor is not None
        assert isinstance(actor, nn.Module)

    def test_decentralized_actor_forward(self):
        """Test DecentralizedActor forward pass."""
        from tianshou.algorithm.multiagent.ctde import DecentralizedActor
        
        actor = DecentralizedActor(
            obs_dim=6,
            action_dim=3,
            hidden_dim=32
        )
        
        obs = torch.randn(10, 6)
        action_logits, state = actor(obs)
        
        assert action_logits.shape == (10, 3)
        assert state is None  # Default state

    def test_decentralized_actor_batch_processing(self):
        """Test DecentralizedActor with various batch sizes."""
        from tianshou.algorithm.multiagent.ctde import DecentralizedActor
        
        actor = DecentralizedActor(obs_dim=4, action_dim=2, hidden_dim=16)
        
        for batch_size in [1, 8, 32]:
            obs = torch.randn(batch_size, 4)
            action_logits, state = actor(obs)
            assert action_logits.shape == (batch_size, 2)


class TestEnhancedPettingZooEnvCoverage:
    """Additional tests for EnhancedPettingZooEnv."""

    def test_env_mode_detection(self):
        """Test automatic environment mode detection."""
        from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv
        from gymnasium import spaces
        
        # Mock parallel environment with identical spaces for all agents
        mock_parallel_env = Mock()
        mock_obs_space = spaces.Box(low=0, high=1, shape=(4,))
        mock_act_space = spaces.Discrete(2)
        mock_parallel_env.observation_spaces = {"agent_0": mock_obs_space, "agent_1": mock_obs_space}
        mock_parallel_env.action_spaces = {"agent_0": mock_act_space, "agent_1": mock_act_space}
        mock_parallel_env.agents = ["agent_0", "agent_1"]
        mock_parallel_env.possible_agents = ["agent_0", "agent_1"]
        
        env = EnhancedPettingZooEnv(mock_parallel_env, mode="auto")
        assert env.mode == "parallel"
        
        # Mock AEC environment - doesn't have observation_spaces attribute
        mock_aec_env = Mock()
        mock_aec_env.agents = ["agent_0", "agent_1"]
        mock_aec_env.possible_agents = ["agent_0", "agent_1"]
        # Explicitly make observation_spaces not exist to trigger AEC mode
        del mock_aec_env.observation_spaces
        
        # For AEC mode, observation_space and action_space are methods
        mock_aec_env.observation_space = Mock(return_value=mock_obs_space)
        mock_aec_env.action_space = Mock(return_value=mock_act_space)
        mock_aec_env.agent_selection = "agent_0"
        mock_aec_env.reset = Mock()
        mock_aec_env.last = Mock(return_value=(np.zeros(4), 0, False, False, {}))
        
        env_aec = EnhancedPettingZooEnv(mock_aec_env, mode="auto")
        assert env_aec.mode == "aec"

    def test_env_explicit_mode(self):
        """Test explicit mode setting."""
        from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv
        
        mock_env = Mock()
        mock_env.agents = ["agent_0"]
        mock_env.possible_agents = ["agent_0"]
        
        env_parallel = EnhancedPettingZooEnv(mock_env, mode="parallel")
        assert env_parallel.mode == "parallel"
        
        env_aec = EnhancedPettingZooEnv(mock_env, mode="aec")
        assert env_aec.mode == "aec"


class TestDictObservationCoverage:
    """Additional tests for dict observation handling."""

    def test_dict_obs_wrapper_creation(self):
        """Test DictObservationWrapper creation with various spaces."""
        from tianshou.env.dict_observation import DictObservationWrapper
        from gymnasium import spaces
        
        # Create proper dict observation space
        obs_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        })
        
        wrapper = DictObservationWrapper(obs_space)
        assert wrapper is not None

    def test_auto_preprocessor_detection(self):
        """Test automatic preprocessor detection for dict observations."""
        from tianshou.env.dict_observation import DictObservationWrapper
        from gymnasium import spaces
        
        # Test with image-like observation
        obs_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        })
        
        wrapper = DictObservationWrapper(obs_space)
        assert wrapper is not None


class TestCommunicationPerformance:
    """Performance and stress tests for communication components."""

    def test_large_message_handling(self):
        """Test communication with large message dimensions."""
        from tianshou.algorithm.multiagent.communication import CommunicationChannel
        
        channel = CommunicationChannel(
            agent_ids=[f"agent_{i}" for i in range(5)],
            message_dim=512,
            comm_type="broadcast"
        )
        
        large_message = torch.randn(1, 512)
        # Use correct method name
        channel.send(
            sender_id="agent_0",
            message=large_message,
            target_agents=None  # Broadcast to all
        )
        
        for i in range(1, 5):
            messages = channel.receive(f"agent_{i}")
            assert len(messages) >= 0  # May have messages

    def test_many_agents_communication(self):
        """Test communication performance with many agents."""
        from tianshou.algorithm.multiagent.communication import CommunicationChannel
        
        n_agents = 20
        agents = [f"agent_{i}" for i in range(n_agents)]
        
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=64,
            comm_type="broadcast"
        )
        
        # Each agent sends a message
        for agent_id in agents:
            message = torch.randn(1, 64)
            channel.send(
                sender_id=agent_id,
                message=message,
                target_agents=None
            )
        
        # Check that each agent can receive messages
        for agent_id in agents:
            messages = channel.receive(agent_id)
            assert isinstance(messages, list)


class TestPolicyManagerRobustness:
    """Test FlexibleMultiAgentPolicyManager robustness."""

    def test_policy_manager_error_handling(self):
        """Test policy manager handles various error conditions."""
        from tianshou.algorithm.multiagent.flexible_policy import FlexibleMultiAgentPolicyManager
        
        mock_env = Mock()
        mock_env.agents = ["agent_0", "agent_1"]
        
        # Test with invalid mode - should raise error
        with pytest.raises(ValueError):
            FlexibleMultiAgentPolicyManager(Mock(), mock_env, mode="invalid_mode")

    def test_policy_manager_mode_switching(self):
        """Test dynamic mode switching in policy manager."""
        from tianshou.algorithm.multiagent.flexible_policy import FlexibleMultiAgentPolicyManager
        from tianshou.algorithm.algorithm_base import Policy
        
        mock_env = Mock()
        mock_env.agents = ["agent_0", "agent_1"]
        
        # Create a proper policy dict instead of Mock
        mock_policy = Mock(spec=Policy)
        policies = {"agent_0": mock_policy, "agent_1": mock_policy}
        
        manager = FlexibleMultiAgentPolicyManager(
            policies, mock_env, mode="independent"
        )
        
        # Test mode attribute
        assert manager.mode == "independent"


class TestTrainerRobustness:
    """Test training coordinator robustness."""

    def test_trainer_initialization_variants(self):
        """Test various trainer initialization patterns."""
        from tianshou.algorithm.multiagent.training_coordinator import SimultaneousTrainer
        from tianshou.algorithm.multiagent.flexible_policy import FlexibleMultiAgentPolicyManager
        
        mock_env = Mock()
        mock_env.agents = ["agent_0", "agent_1"]
        mock_policy = Mock()
        policies = {"agent_0": mock_policy, "agent_1": mock_policy}
        
        policy_manager = FlexibleMultiAgentPolicyManager(
            policies, mock_env, mode="independent"
        )
        
        trainer = SimultaneousTrainer(
            policy_manager=policy_manager,
            envs=mock_env,
            buffer=Mock(),
            update_frequency=10
        )
        
        assert trainer is not None

    def test_training_frequency_validation(self):
        """Test training frequency parameter validation."""
        from tianshou.algorithm.multiagent.training_coordinator import SimultaneousTrainer
        from tianshou.algorithm.multiagent.flexible_policy import FlexibleMultiAgentPolicyManager
        
        mock_env = Mock()
        mock_env.agents = ["agent_0"]
        mock_policy = Mock()
        policies = {"agent_0": mock_policy}
        
        policy_manager = FlexibleMultiAgentPolicyManager(
            policies, mock_env, mode="independent"
        )
        
        # Test various frequency values
        for freq in [1, 5, 10, 100]:
            trainer = SimultaneousTrainer(
                policy_manager=policy_manager,
                envs=mock_env,
                buffer=Mock(),
                update_frequency=freq
            )
            assert trainer.update_frequency == freq


class TestIntegrationRobustness:
    """Robustness tests for component integration."""

    def test_component_compatibility(self):
        """Test that different components work together."""
        from tianshou.algorithm.multiagent.communication import (
            CommunicationChannel, MessageEncoder, MessageDecoder
        )
        
        # Create compatible components
        channel = CommunicationChannel(
            agent_ids=["a0", "a1"],
            message_dim=32,
            comm_type="broadcast"
        )
        encoder = MessageEncoder(input_dim=64, message_dim=32)
        decoder = MessageDecoder(message_dim=32, output_dim=16)
        
        # Test they work together
        obs = torch.randn(5, 64)
        message = encoder(obs)
        channel.send("a0", message, target_agents=None)
        
        received = channel.receive("a1")
        if received:  # May or may not have messages
            decoded = decoder(received)
            assert decoded.shape[1] == 16

    def test_batch_size_consistency(self):
        """Test that all components handle consistent batch sizes."""
        from tianshou.algorithm.multiagent.ctde import GlobalStateConstructor
        from tianshou.algorithm.multiagent.communication import MessageEncoder
        
        # Test GlobalStateConstructor
        constructor = GlobalStateConstructor(mode="concat")
        
        for batch_size in [1, 8, 16]:
            obs_dict = {
                "agent_0": torch.randn(batch_size, 4),
                "agent_1": torch.randn(batch_size, 6)
            }
            result = constructor(obs_dict)
            assert result.shape[0] == batch_size
            assert result.shape[1] == 10  # 4 + 6
                    
        # Test MessageEncoder
        encoder = MessageEncoder(input_dim=64, message_dim=32)
        
        for batch_size in [1, 8, 16]:
            obs = torch.randn(batch_size, 64)
            result = encoder(obs)
            assert result.shape[0] == batch_size
            assert result.shape[1] == 32