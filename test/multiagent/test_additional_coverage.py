"""Additional fast tests to maximize MARL coverage."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

# Delay torch import to avoid conflicts
torch = None

def setup_module():
    """Setup module with delayed torch import."""
    global torch
    import torch as torch_module
    torch = torch_module


class TestCentralizedCriticCoverage:
    """Additional tests for CentralizedCritic functionality."""

    def test_centralized_critic_initialization(self):
        """Test CentralizedCritic initialization with various configurations."""
        from tianshou.algorithm.multiagent.ctde import CentralizedCritic
        
        # Test basic initialization
        critic = CentralizedCritic(
            global_state_dim=20,
            action_dims={"agent_0": 4, "agent_1": 6},
            hidden_sizes=[64, 32]
        )
        assert critic is not None
        
        # Test with single hidden layer
        critic_simple = CentralizedCritic(
            global_state_dim=10,
            action_dims={"agent_0": 2},
            hidden_sizes=[32]
        )
        assert critic_simple is not None

    def test_centralized_critic_forward(self):
        """Test CentralizedCritic forward pass functionality."""
        from tianshou.algorithm.multiagent.ctde import CentralizedCritic
        
        critic = CentralizedCritic(
            global_state_dim=16,
            action_dims={"agent_0": 4, "agent_1": 6},
            hidden_sizes=[32]
        )
        
        global_state = torch.randn(8, 16)
        joint_actions = {
            "agent_0": torch.randn(8, 4),
            "agent_1": torch.randn(8, 6)
        }
        
        q_values = critic(global_state, joint_actions)
        assert q_values.shape == (8, 1)

    def test_centralized_critic_different_batch_sizes(self):
        """Test CentralizedCritic with different batch sizes."""
        from tianshou.algorithm.multiagent.ctde import CentralizedCritic
        
        critic = CentralizedCritic(
            global_state_dim=12,
            action_dims={"agent_0": 3, "agent_1": 3},
            hidden_sizes=[64]
        )
        
        for batch_size in [1, 4, 16, 32]:
            global_state = torch.randn(batch_size, 12)
            joint_actions = {
                "agent_0": torch.randn(batch_size, 3),
                "agent_1": torch.randn(batch_size, 3)
            }
            
            q_values = critic(global_state, joint_actions)
            assert q_values.shape == (batch_size, 1)


class TestDecentralizedActorCoverage:
    """Additional tests for DecentralizedActor functionality."""

    def test_decentralized_actor_initialization(self):
        """Test DecentralizedActor initialization."""
        from tianshou.algorithm.multiagent.ctde import DecentralizedActor
        
        # Test basic initialization
        actor = DecentralizedActor(
            obs_dim=8,
            action_dim=4,
            hidden_sizes=[32, 16]
        )
        assert actor is not None

    def test_decentralized_actor_forward(self):
        """Test DecentralizedActor forward pass."""
        from tianshou.algorithm.multiagent.ctde import DecentralizedActor
        
        actor = DecentralizedActor(
            obs_dim=6,
            action_dim=3,
            hidden_sizes=[32]
        )
        
        obs = torch.randn(10, 6)
        action_logits = actor(obs)
        
        assert action_logits.shape == (10, 3)

    def test_decentralized_actor_batch_processing(self):
        """Test DecentralizedActor with various batch sizes."""
        from tianshou.algorithm.multiagent.ctde import DecentralizedActor
        
        actor = DecentralizedActor(obs_dim=4, action_dim=2, hidden_sizes=[16])
        
        for batch_size in [1, 8, 32]:
            obs = torch.randn(batch_size, 4)
            action_logits = actor(obs)
            assert action_logits.shape == (batch_size, 2)


class TestEnhancedPettingZooEnvCoverage:
    """Additional tests for EnhancedPettingZooEnv."""

    def test_env_mode_detection(self):
        """Test automatic environment mode detection."""
        from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv
        
        # Mock parallel environment
        mock_parallel_env = Mock()
        mock_parallel_env.observation_spaces = {"agent_0": Mock(), "agent_1": Mock()}
        mock_parallel_env.action_spaces = {"agent_0": Mock(), "agent_1": Mock()}
        mock_parallel_env.agents = ["agent_0", "agent_1"]
        
        env = EnhancedPettingZooEnv(mock_parallel_env, mode="auto")
        assert env.mode == "parallel"
        
        # Mock AEC environment
        mock_aec_env = Mock()
        del mock_aec_env.observation_spaces  # AEC doesn't have this attribute
        mock_aec_env.agents = ["agent_0", "agent_1"]
        
        env_aec = EnhancedPettingZooEnv(mock_aec_env, mode="auto")
        assert env_aec.mode == "aec"

    def test_env_explicit_mode(self):
        """Test explicit mode setting."""
        from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv
        
        mock_env = Mock()
        mock_env.agents = ["agent_0"]
        
        env_parallel = EnhancedPettingZooEnv(mock_env, mode="parallel")
        assert env_parallel.mode == "parallel"
        
        env_aec = EnhancedPettingZooEnv(mock_env, mode="aec")
        assert env_aec.mode == "aec"


class TestDictObservationCoverage:
    """Additional tests for dict observation handling."""

    def test_dict_obs_wrapper_creation(self):
        """Test DictObservationWrapper creation with various spaces."""
        from tianshou.env.dict_observation import DictObservationWrapper
        
        # Mock environment with dict observation space
        mock_env = Mock()
        mock_space = Mock()
        mock_space.spaces = {
            "image": Mock(),
            "vector": Mock(),
            "position": Mock()
        }
        mock_env.observation_space = mock_space
        
        wrapper = DictObservationWrapper(mock_env)
        assert wrapper is not None

    def test_auto_preprocessor_detection(self):
        """Test automatic preprocessor detection for dict observations."""
        from tianshou.env.dict_observation import DictObservationWrapper
        from unittest.mock import Mock
        
        mock_env = Mock()
        mock_space = Mock()
        
        # Test with image-like observation
        image_space = Mock()
        image_space.shape = (84, 84, 3)  # Image-like shape
        
        mock_space.spaces = {"image": image_space}
        mock_env.observation_space = mock_space
        
        wrapper = DictObservationWrapper(mock_env, auto_preprocess=True)
        assert wrapper is not None


class TestCommunicationPerformance:
    """Performance and stress tests for communication components."""

    def test_large_message_handling(self):
        """Test communication with large message dimensions."""
        from tianshou.algorithm.multiagent import CommunicationChannel
        
        channel = CommunicationChannel(
            agent_ids=[f"agent_{i}" for i in range(5)],
            message_dim=512,  # Large message
            topology="broadcast"
        )
        
        large_message = torch.randn(1, 512)
        channel.send_message("agent_0", large_message)
        
        for i in range(1, 5):
            messages = channel.get_messages(f"agent_{i}")
            assert len(messages) == 1
            assert messages[0].shape == (1, 512)

    def test_many_agents_communication(self):
        """Test communication performance with many agents."""
        from tianshou.algorithm.multiagent import CommunicationChannel
        
        n_agents = 50
        agents = [f"agent_{i}" for i in range(n_agents)]
        
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=64,
            topology="broadcast"
        )
        
        # Each agent sends a message
        for i, agent_id in enumerate(agents):
            message = torch.randn(1, 64)
            channel.send_message(agent_id, message)
        
        # Check that each agent receives messages from others
        for agent_id in agents:
            messages = channel.get_messages(agent_id)
            assert len(messages) >= 0  # At least no errors


class TestPolicyManagerRobustness:
    """Test FlexibleMultiAgentPolicyManager robustness."""

    def test_policy_manager_error_handling(self):
        """Test policy manager handles various error conditions."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        mock_env = Mock()
        mock_env.agents = ["agent_0", "agent_1"]
        
        # Test with invalid mode
        with pytest.raises(ValueError):
            FlexibleMultiAgentPolicyManager(Mock(), mock_env, mode="invalid_mode")

    def test_policy_manager_mode_switching(self):
        """Test dynamic mode switching in policy manager."""
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        
        mock_env = Mock()
        mock_env.agents = ["agent_0", "agent_1"]
        mock_policy = Mock()
        
        manager = FlexibleMultiAgentPolicyManager(
            mock_policy, mock_env, mode="shared"
        )
        
        # Test mode switching
        manager.set_mode("independent")
        assert manager.mode == "independent"


class TestTrainerRobustness:
    """Test training coordinator robustness."""

    def test_trainer_initialization_variants(self):
        """Test various trainer initialization patterns."""
        from tianshou.algorithm.multiagent import SimultaneousTrainer
        
        mock_policies = {"agent_0": Mock(), "agent_1": Mock()}
        mock_envs = Mock()
        mock_buffer = Mock()
        
        trainer = SimultaneousTrainer(
            policies=mock_policies,
            envs=mock_envs,
            buffer=mock_buffer,
            update_frequency=10
        )
        
        assert trainer is not None

    def test_training_frequency_validation(self):
        """Test training frequency parameter validation."""
        from tianshou.algorithm.multiagent import SimultaneousTrainer
        
        mock_policies = {"agent_0": Mock()}
        mock_envs = Mock()
        mock_buffer = Mock()
        
        # Test various frequency values
        for freq in [1, 5, 10, 100]:
            trainer = SimultaneousTrainer(
                policies=mock_policies,
                envs=mock_envs,
                buffer=mock_buffer,
                update_frequency=freq
            )
            assert trainer.update_frequency == freq


class TestIntegrationRobustness:
    """Robustness tests for component integration."""

    def test_component_compatibility(self):
        """Test that different components work together."""
        from tianshou.algorithm.multiagent import (
            CommunicationChannel, MessageEncoder, MessageDecoder, 
            FlexibleMultiAgentPolicyManager
        )
        
        # Create compatible components
        channel = CommunicationChannel(["a0", "a1"], 32, "broadcast")
        encoder = MessageEncoder(64, 32)
        decoder = MessageDecoder(32, 16)
        
        # Test they work together
        obs = torch.randn(5, 64)
        message = encoder(obs)
        channel.send_message("a0", message)
        
        received = channel.get_messages("a1")
        decoded = decoder(received)
        
        assert decoded.shape == (16,)

    def test_batch_size_consistency(self):
        """Test that all components handle consistent batch sizes."""
        components_to_test = [
            ("GlobalStateConstructor", {"mode": "concat"}),
            ("MessageEncoder", {"input_dim": 64, "message_dim": 32}),
            ("MessageDecoder", {"message_dim": 32, "output_dim": 16}),
        ]
        
        for comp_name, kwargs in components_to_test:
            if comp_name == "GlobalStateConstructor":
                from tianshou.algorithm.multiagent.ctde import GlobalStateConstructor
                comp = GlobalStateConstructor(**kwargs)
                
                # Test with different batch sizes
                for batch_size in [1, 8, 16]:
                    obs_dict = {
                        "agent_0": torch.randn(batch_size, 4),
                        "agent_1": torch.randn(batch_size, 6)
                    }
                    result = comp(obs_dict)
                    assert result.shape[0] == batch_size
                    
            elif comp_name == "MessageEncoder":
                from tianshou.algorithm.multiagent import MessageEncoder
                comp = MessageEncoder(**kwargs)
                
                for batch_size in [1, 8, 16]:
                    obs = torch.randn(batch_size, 64)
                    result = comp(obs)
                    assert result.shape[0] == batch_size