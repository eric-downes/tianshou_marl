"""Tests for inter-agent communication module."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Batch
from tianshou.algorithm.multiagent.communication import (
    CommunicationChannel,
    MessageEncoder,
    MessageDecoder,
    CommunicatingPolicy,
    MultiAgentCommunicationWrapper,
)


class DummyActor(nn.Module):
    """Simple actor network for testing."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, obs: torch.Tensor, state=None):
        return self.net(obs), state


class TestCommunicationChannel:
    """Test suite for CommunicationChannel class."""

    def test_broadcast_communication(self):
        """Test broadcast communication mode."""
        agents = ["agent_0", "agent_1", "agent_2"]
        channel = CommunicationChannel(comm_type="broadcast", message_dim=8, agent_ids=agents)

        # Agent 0 sends a message
        msg = torch.randn(8)
        channel.send("agent_0", msg)

        # Agent 1 and 2 should receive it, but not agent 0
        assert len(channel.receive("agent_0")) == 0
        assert len(channel.receive("agent_1")) == 1
        assert len(channel.receive("agent_2")) == 1

        # Check message content
        received = channel.receive("agent_1")[0]
        assert torch.allclose(received, msg)

    def test_targeted_communication(self):
        """Test targeted communication mode."""
        agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        channel = CommunicationChannel(comm_type="targeted", message_dim=8, agent_ids=agents)

        # Agent 0 sends to agent 2
        msg1 = torch.randn(8)
        channel.send("agent_0", msg1, target_id="agent_2")

        # Agent 1 sends to multiple agents
        msg2 = torch.randn(8)
        channel.send("agent_1", msg2, target_id=["agent_0", "agent_3"])

        # Check who receives what
        agent_0_msgs = channel.receive("agent_0")
        agent_1_msgs = channel.receive("agent_1")
        agent_2_msgs = channel.receive("agent_2")
        agent_3_msgs = channel.receive("agent_3")

        assert len(agent_0_msgs) == 1  # From agent_1
        assert len(agent_1_msgs) == 0  # No messages
        assert len(agent_2_msgs) == 1  # From agent_0
        assert len(agent_3_msgs) == 1  # From agent_1

        assert torch.allclose(agent_2_msgs[0], msg1)
        assert torch.allclose(agent_0_msgs[0], msg2)

    def test_graph_communication(self):
        """Test graph-based communication mode."""
        agents = ["agent_0", "agent_1", "agent_2", "agent_3"]

        # Define a simple graph topology (adjacency matrix)
        # 0 -> 1, 0 -> 2
        # 1 -> 2
        # 2 -> 3
        # 3 -> 0
        topology = np.array(
            [
                [0, 1, 1, 0],  # agent_0 can send to 1 and 2
                [0, 0, 1, 0],  # agent_1 can send to 2
                [0, 0, 0, 1],  # agent_2 can send to 3
                [1, 0, 0, 0],  # agent_3 can send to 0
            ]
        )

        channel = CommunicationChannel(
            comm_type="graph", message_dim=8, topology=topology, agent_ids=agents
        )

        # Each agent sends a message
        msgs = []
        for i, agent in enumerate(agents):
            msg = torch.ones(8) * i
            msgs.append(msg)
            channel.send(agent, msg)

        # Check received messages based on topology
        agent_0_msgs = channel.receive("agent_0")
        agent_1_msgs = channel.receive("agent_1")
        agent_2_msgs = channel.receive("agent_2")
        agent_3_msgs = channel.receive("agent_3")

        assert len(agent_0_msgs) == 1  # From agent_3
        assert torch.allclose(agent_0_msgs[0], msgs[3])

        assert len(agent_1_msgs) == 1  # From agent_0
        assert torch.allclose(agent_1_msgs[0], msgs[0])

        assert len(agent_2_msgs) == 2  # From agent_0 and agent_1
        received_values = {msg[0].item() for msg in agent_2_msgs}
        assert received_values == {0.0, 1.0}

        assert len(agent_3_msgs) == 1  # From agent_2
        assert torch.allclose(agent_3_msgs[0], msgs[2])

    def test_message_buffer_limit(self):
        """Test that message buffer respects max_messages limit."""
        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=4, max_messages=3, agent_ids=["sender", "receiver"]
        )

        # Send 5 messages
        for i in range(5):
            msg = torch.ones(4) * i
            channel.send("sender", msg)

        # Receiver should only get the last 3 messages
        received = channel.receive("receiver")
        assert len(received) == 3

        # Should be messages 2, 3, 4 (most recent)
        for i, msg in enumerate(received):
            expected_value = float(i + 2)
            assert torch.allclose(msg, torch.ones(4) * expected_value)

    def test_clear_buffer(self):
        """Test clearing the message buffer."""
        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=4, agent_ids=["agent_0", "agent_1"]
        )

        # Send a message
        channel.send("agent_0", torch.randn(4))
        assert channel.get_buffer_size() == 1

        # Clear buffer
        channel.clear()
        assert channel.get_buffer_size() == 0
        assert len(channel.receive("agent_1")) == 0

    def test_topology_update(self):
        """Test updating communication topology."""
        agents = ["agent_0", "agent_1"]
        initial_topology = np.array([[0, 1], [0, 0]])
        channel = CommunicationChannel(
            comm_type="graph", message_dim=4, topology=initial_topology, agent_ids=agents
        )

        # Initially agent_0 can send to agent_1
        msg = torch.randn(4)
        channel.send("agent_0", msg)
        assert len(channel.receive("agent_1")) == 1

        # Update topology so agent_0 cannot send to agent_1
        channel.clear()
        new_topology = np.array([[0, 0], [1, 0]])
        channel.update_topology(new_topology)

        channel.send("agent_0", msg)
        assert len(channel.receive("agent_1")) == 0

        # Now agent_1 can send to agent_0
        channel.send("agent_1", msg)
        assert len(channel.receive("agent_0")) == 1


class TestMessageEncoderDecoder:
    """Test suite for MessageEncoder and MessageDecoder."""

    def test_encoder_output_dimension(self):
        """Test that encoder produces correct output dimension."""
        encoder = MessageEncoder(input_dim=10, message_dim=8, hidden_dim=16)

        x = torch.randn(10)
        msg = encoder(x)
        assert msg.shape == (8,)

        # Batch input
        x_batch = torch.randn(5, 10)
        msg_batch = encoder(x_batch)
        assert msg_batch.shape == (5, 8)

    def test_encoder_output_range(self):
        """Test that encoder output is normalized."""
        encoder = MessageEncoder(input_dim=10, message_dim=8)

        x = torch.randn(100, 10) * 10  # Large inputs
        msgs = encoder(x)

        # Due to tanh, outputs should be in [-1, 1]
        assert msgs.min() >= -1.0
        assert msgs.max() <= 1.0

    def test_decoder_no_messages(self):
        """Test decoder behavior with no messages."""
        decoder = MessageDecoder(message_dim=8, output_dim=6)

        result = decoder([])
        assert result.shape == (6,)
        assert torch.allclose(result, torch.zeros(6))

    def test_decoder_single_message(self):
        """Test decoder with single message."""
        decoder = MessageDecoder(message_dim=8, output_dim=6)

        msg = torch.randn(8)
        result = decoder([msg])
        assert result.shape == (6,)

    def test_decoder_aggregation_methods(self):
        """Test different aggregation methods in decoder."""
        message_dim = 8
        output_dim = 6

        # Create test messages
        msgs = [torch.ones(message_dim) * i for i in range(3)]

        # Test mean aggregation
        decoder_mean = MessageDecoder(
            message_dim=message_dim, output_dim=output_dim, aggregation="mean"
        )
        result_mean = decoder_mean(msgs)
        assert result_mean.shape == (output_dim,)

        # Test sum aggregation
        decoder_sum = MessageDecoder(
            message_dim=message_dim, output_dim=output_dim, aggregation="sum"
        )
        result_sum = decoder_sum(msgs)
        assert result_sum.shape == (output_dim,)

        # Test max aggregation
        decoder_max = MessageDecoder(
            message_dim=message_dim, output_dim=output_dim, aggregation="max"
        )
        result_max = decoder_max(msgs)
        assert result_max.shape == (output_dim,)

        # Test attention aggregation
        decoder_attn = MessageDecoder(
            message_dim=message_dim, output_dim=output_dim, aggregation="attention"
        )
        result_attn = decoder_attn(msgs)
        assert result_attn.shape == (output_dim,)

    def test_decoder_attention_weights(self):
        """Test that attention aggregation produces valid weights."""
        decoder = MessageDecoder(message_dim=8, output_dim=6, aggregation="attention")

        # Create messages with different patterns
        msg1 = torch.ones(8)
        msg2 = torch.zeros(8)
        msg3 = torch.randn(8)

        result = decoder([msg1, msg2, msg3])
        assert result.shape == (6,)
        # Result should be a weighted combination, not extreme values


class TestCommunicatingPolicy:
    """Test suite for CommunicatingPolicy."""

    def test_policy_without_communication(self):
        """Test policy behavior when communication is disabled."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8

        # Create components
        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=message_dim, agent_ids=["agent_0"]
        )
        encoder = MessageEncoder(obs_dim, message_dim)
        decoder = MessageDecoder(message_dim, message_dim)
        actor = DummyActor(obs_dim, act_dim)  # Without communication enhancement

        policy = CommunicatingPolicy(
            actor=actor,
            comm_channel=channel,
            comm_encoder=encoder,
            comm_decoder=decoder,
            obs_dim=obs_dim,
            act_dim=act_dim,
            agent_id="agent_0",
            communication_enabled=False,
        )

        # Test forward pass
        batch = Batch(obs=torch.randn(obs_dim))
        result = policy.forward(batch)

        assert "act" in result
        assert result.act.shape == (act_dim,)

        # No messages should be sent
        assert channel.get_buffer_size() == 0

    def test_policy_with_communication(self):
        """Test policy behavior with communication enabled."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8
        comm_output_dim = 6

        # Create components
        agents = ["agent_0", "agent_1"]
        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=message_dim, agent_ids=agents
        )
        encoder = MessageEncoder(obs_dim, message_dim)
        decoder = MessageDecoder(message_dim, comm_output_dim)

        # Actor takes enhanced observation (obs + comm_info)
        enhanced_dim = obs_dim + comm_output_dim
        actor = DummyActor(enhanced_dim, act_dim)

        policy = CommunicatingPolicy(
            actor=actor,
            comm_channel=channel,
            comm_encoder=encoder,
            comm_decoder=decoder,
            obs_dim=obs_dim,
            act_dim=act_dim,
            agent_id="agent_0",
            communication_enabled=True,
        )

        # Test forward pass
        batch = Batch(obs=torch.randn(obs_dim))
        result = policy.forward(batch)

        assert "act" in result
        assert result.act.shape == (act_dim,)

        # A message should be sent
        assert channel.get_buffer_size() == 1

    def test_policy_communication_toggle(self):
        """Test enabling/disabling communication."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8

        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=message_dim, agent_ids=["agent_0"]
        )
        encoder = MessageEncoder(obs_dim, message_dim)
        decoder = MessageDecoder(message_dim, message_dim)

        # Create actor that can handle both obs dimensions
        actor_with_comm = DummyActor(obs_dim + message_dim, act_dim)
        actor_without_comm = DummyActor(obs_dim, act_dim)

        policy = CommunicatingPolicy(
            actor=actor_without_comm,  # Start with no-comm actor
            comm_channel=channel,
            comm_encoder=encoder,
            comm_decoder=decoder,
            obs_dim=obs_dim,
            act_dim=act_dim,
            agent_id="agent_0",
            communication_enabled=False,
        )

        batch = Batch(obs=torch.randn(obs_dim))

        # Test with communication disabled
        policy.set_communication_enabled(False)
        result1 = policy.forward(batch)
        assert channel.get_buffer_size() == 0

        # Test with communication enabled
        policy.actor = actor_with_comm  # Switch to comm-aware actor
        policy.set_communication_enabled(True)
        result2 = policy.forward(batch)
        assert channel.get_buffer_size() == 1

    def test_policy_batch_processing(self):
        """Test policy with batched observations."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8
        batch_size = 5

        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=message_dim, agent_ids=["agent_0", "agent_1"]
        )
        encoder = MessageEncoder(obs_dim, message_dim)
        decoder = MessageDecoder(message_dim, message_dim)
        actor = DummyActor(obs_dim + message_dim, act_dim)

        policy = CommunicatingPolicy(
            actor=actor,
            comm_channel=channel,
            comm_encoder=encoder,
            comm_decoder=decoder,
            obs_dim=obs_dim,
            act_dim=act_dim,
            agent_id="agent_0",
            communication_enabled=True,
        )

        # Send a message from another agent first
        channel.send("agent_1", torch.randn(message_dim))

        # Test with batched observations
        batch = Batch(obs=torch.randn(batch_size, obs_dim))
        result = policy.forward(batch)

        assert result.act.shape == (batch_size, act_dim)


class TestMultiAgentCommunicationWrapper:
    """Test suite for MultiAgentCommunicationWrapper."""

    def test_wrapper_coordination(self):
        """Test that wrapper coordinates multiple agents."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8
        agents = ["agent_0", "agent_1", "agent_2"]

        # Create shared channel
        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=message_dim, agent_ids=agents
        )

        # Create policies for each agent
        policies = {}
        for agent_id in agents:
            encoder = MessageEncoder(obs_dim, message_dim)
            decoder = MessageDecoder(message_dim, message_dim)
            actor = DummyActor(obs_dim + message_dim, act_dim)

            policy = CommunicatingPolicy(
                actor=actor,
                comm_channel=channel,
                comm_encoder=encoder,
                comm_decoder=decoder,
                obs_dim=obs_dim,
                act_dim=act_dim,
                agent_id=agent_id,
                communication_enabled=True,
            )
            policies[agent_id] = policy

        # Create wrapper
        wrapper = MultiAgentCommunicationWrapper(policies, channel)

        # Test step with all agents
        observations = {agent_id: torch.randn(obs_dim) for agent_id in agents}
        actions = wrapper.step(observations)

        assert len(actions) == len(agents)
        for agent_id in agents:
            assert agent_id in actions
            assert actions[agent_id].shape == (act_dim,)

    def test_wrapper_reset(self):
        """Test wrapper reset functionality."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8
        agents = ["agent_0", "agent_1"]

        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=message_dim, agent_ids=agents
        )

        policies = {}
        for agent_id in agents:
            encoder = MessageEncoder(obs_dim, message_dim)
            decoder = MessageDecoder(message_dim, message_dim)
            actor = DummyActor(obs_dim + message_dim, act_dim)

            policy = CommunicatingPolicy(
                actor=actor,
                comm_channel=channel,
                comm_encoder=encoder,
                comm_decoder=decoder,
                obs_dim=obs_dim,
                act_dim=act_dim,
                agent_id=agent_id,
            )
            policies[agent_id] = policy

        wrapper = MultiAgentCommunicationWrapper(policies, channel)

        # Send some messages
        channel.send("agent_0", torch.randn(message_dim))
        assert channel.get_buffer_size() == 1

        # Reset should clear messages
        wrapper.reset()
        assert channel.get_buffer_size() == 0

    def test_wrapper_communication_toggle(self):
        """Test enabling/disabling communication for all agents."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8
        agents = ["agent_0", "agent_1"]

        channel = CommunicationChannel(
            comm_type="broadcast", message_dim=message_dim, agent_ids=agents
        )

        policies = {}
        for agent_id in agents:
            encoder = MessageEncoder(obs_dim, message_dim)
            decoder = MessageDecoder(message_dim, message_dim)

            # Create actors for both modes
            actor_no_comm = DummyActor(obs_dim, act_dim)

            policy = CommunicatingPolicy(
                actor=actor_no_comm,
                comm_channel=channel,
                comm_encoder=encoder,
                comm_decoder=decoder,
                obs_dim=obs_dim,
                act_dim=act_dim,
                agent_id=agent_id,
                communication_enabled=True,
            )
            policies[agent_id] = policy

        wrapper = MultiAgentCommunicationWrapper(policies, channel)

        # Disable communication for all
        wrapper.set_communication_enabled(False)
        for policy in policies.values():
            assert not policy.communication_enabled

        # Enable communication for all
        wrapper.set_communication_enabled(True)
        for policy in policies.values():
            assert policy.communication_enabled


class TestIntegration:
    """Integration tests for communication system."""

    def test_full_communication_cycle(self):
        """Test complete communication cycle between multiple agents."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8
        comm_output_dim = 6
        agents = ["agent_0", "agent_1", "agent_2"]

        # Create communication topology (ring topology)
        topology = np.array(
            [
                [0, 1, 0],  # 0 -> 1
                [0, 0, 1],  # 1 -> 2
                [1, 0, 0],  # 2 -> 0
            ]
        )

        channel = CommunicationChannel(
            comm_type="graph", message_dim=message_dim, topology=topology, agent_ids=agents
        )

        policies = {}
        for agent_id in agents:
            encoder = MessageEncoder(obs_dim, message_dim)
            decoder = MessageDecoder(message_dim, comm_output_dim)
            actor = DummyActor(obs_dim + comm_output_dim, act_dim)

            policy = CommunicatingPolicy(
                actor=actor,
                comm_channel=channel,
                comm_encoder=encoder,
                comm_decoder=decoder,
                obs_dim=obs_dim,
                act_dim=act_dim,
                agent_id=agent_id,
                communication_enabled=True,
            )
            policies[agent_id] = policy

        wrapper = MultiAgentCommunicationWrapper(policies, channel)

        # Run multiple steps to test message propagation
        for step in range(3):
            observations = {agent_id: torch.randn(obs_dim) for agent_id in agents}
            actions = wrapper.step(observations)

            # All agents should produce actions
            assert len(actions) == len(agents)
            for agent_id in agents:
                assert actions[agent_id].shape == (act_dim,)

    def test_performance_with_many_agents(self):
        """Test system performance with many agents."""
        obs_dim = 10
        act_dim = 4
        message_dim = 8
        num_agents = 20
        agents = [f"agent_{i}" for i in range(num_agents)]

        channel = CommunicationChannel(
            comm_type="broadcast",
            message_dim=message_dim,
            max_messages=10,  # Limit messages to prevent memory issues
            agent_ids=agents,
        )

        policies = {}
        for agent_id in agents:
            encoder = MessageEncoder(obs_dim, message_dim)
            decoder = MessageDecoder(message_dim, message_dim)
            actor = DummyActor(obs_dim + message_dim, act_dim)

            policy = CommunicatingPolicy(
                actor=actor,
                comm_channel=channel,
                comm_encoder=encoder,
                comm_decoder=decoder,
                obs_dim=obs_dim,
                act_dim=act_dim,
                agent_id=agent_id,
                communication_enabled=True,
            )
            policies[agent_id] = policy

        wrapper = MultiAgentCommunicationWrapper(policies, channel)

        # Test that system handles many agents efficiently
        observations = {agent_id: torch.randn(obs_dim) for agent_id in agents}

        import time

        start_time = time.time()
        actions = wrapper.step(observations)
        elapsed = time.time() - start_time

        assert len(actions) == num_agents
        # Should complete reasonably quickly (< 1 second for 20 agents)
        assert elapsed < 1.0
