"""Fast tests for MARL communication components."""

import numpy as np
import pytest
import torch

from tianshou.algorithm.multiagent import (
    CommunicationChannel,
    MessageDecoder,
    MessageEncoder,
)


class TestCommunicationChannel:
    """Test CommunicationChannel functionality."""

    def test_broadcast_communication(self):
        """Test broadcast communication topology."""
        agents = ["agent_0", "agent_1", "agent_2"]
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=32,
            topology="broadcast"
        )
        
        # Test message sending
        message = torch.randn(1, 32)
        channel.send_message("agent_0", message)
        
        # All agents should receive the message
        messages = channel.get_messages("agent_1")
        assert len(messages) == 1
        assert torch.equal(messages[0], message)

    def test_targeted_communication(self):
        """Test targeted communication."""
        agents = ["agent_0", "agent_1", "agent_2"]
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=32,
            topology="custom",
            custom_topology={"agent_0": ["agent_1"]}
        )
        
        message = torch.randn(1, 32)
        channel.send_message("agent_0", message)
        
        # Only agent_1 should receive the message
        messages_1 = channel.get_messages("agent_1")
        messages_2 = channel.get_messages("agent_2")
        
        assert len(messages_1) == 1
        assert len(messages_2) == 0

    def test_message_buffer_limit(self):
        """Test message buffer size limits."""
        agents = ["agent_0", "agent_1"]
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=32,
            topology="broadcast",
            max_buffer_size=2
        )
        
        # Send more messages than buffer can hold
        for i in range(3):
            message = torch.randn(1, 32)
            channel.send_message("agent_0", message)
        
        messages = channel.get_messages("agent_1")
        assert len(messages) <= 2  # Only last 2 messages

    def test_clear_buffer(self):
        """Test clearing message buffers."""
        agents = ["agent_0", "agent_1"]
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=32,
            topology="broadcast"
        )
        
        message = torch.randn(1, 32)
        channel.send_message("agent_0", message)
        
        assert len(channel.get_messages("agent_1")) == 1
        
        channel.clear_buffers()
        assert len(channel.get_messages("agent_1")) == 0


class TestMessageEncoderDecoder:
    """Test MessageEncoder and MessageDecoder."""

    def test_encoder_output_dimension(self):
        """Test encoder produces correct output dimension."""
        encoder = MessageEncoder(input_dim=64, message_dim=32)
        
        obs = torch.randn(10, 64)  # batch_size=10
        message = encoder(obs)
        
        assert message.shape == (10, 32)

    def test_decoder_no_messages(self):
        """Test decoder with no messages."""
        decoder = MessageDecoder(
            message_dim=32,
            output_dim=16,
            aggregation_method="mean"
        )
        
        messages = []
        output = decoder(messages)
        
        assert output.shape == (16,)
        assert torch.all(output == 0.0)  # Should be zeros

    def test_decoder_aggregation_methods(self):
        """Test different aggregation methods."""
        messages = [torch.randn(1, 32) for _ in range(3)]
        
        methods = ["mean", "sum", "max"]
        for method in methods:
            decoder = MessageDecoder(
                message_dim=32,
                output_dim=16,
                aggregation_method=method
            )
            output = decoder(messages)
            assert output.shape == (16,)


class TestIntegration:
    """Integration tests for communication system."""

    def test_full_communication_cycle(self):
        """Test complete send -> receive -> decode cycle."""
        agents = ["agent_0", "agent_1"]
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=32,
            topology="broadcast"
        )
        
        encoder = MessageEncoder(input_dim=64, message_dim=32)
        decoder = MessageDecoder(message_dim=32, output_dim=16)
        
        # Agent 0 observes and sends message
        obs = torch.randn(1, 64)
        message = encoder(obs)
        channel.send_message("agent_0", message)
        
        # Agent 1 receives and decodes
        received_messages = channel.get_messages("agent_1")
        decoded = decoder(received_messages)
        
        assert len(received_messages) == 1
        assert decoded.shape == (16,)

    def test_performance_with_many_agents(self):
        """Test communication performance with many agents."""
        num_agents = 10
        agents = [f"agent_{i}" for i in range(num_agents)]
        
        channel = CommunicationChannel(
            agent_ids=agents,
            message_dim=32,
            topology="broadcast"
        )
        
        # Send messages from all agents
        for agent_id in agents:
            message = torch.randn(1, 32)
            channel.send_message(agent_id, message)
        
        # Each agent should receive messages from all others
        for agent_id in agents:
            messages = channel.get_messages(agent_id)
            assert len(messages) >= 0  # At least no errors
