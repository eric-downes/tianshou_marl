"""Inter-agent communication module for multi-agent reinforcement learning."""

import time
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tianshou.data import Batch


class CommunicationChannel:
    """Inter-agent communication module.

    This class manages message passing between agents in multi-agent environments.
    It supports various communication patterns including broadcast, targeted, and
    graph-based communication.

    Args:
        comm_type: Communication pattern
            - "broadcast": All agents receive all messages
            - "targeted": Agents send messages to specific recipients
            - "graph": Communication follows graph topology
        message_dim: Dimension of message vectors
        max_messages: Maximum messages per step
        topology: Adjacency matrix for graph communication
        agent_ids: List of agent identifiers

    """

    def __init__(
        self,
        comm_type: Literal["broadcast", "targeted", "graph"] = "broadcast",
        message_dim: int = 32,
        max_messages: int = 10,
        topology: np.ndarray | None = None,
        agent_ids: list[str] | None = None,
    ):
        self.comm_type = comm_type
        self.message_dim = message_dim
        self.max_messages = max_messages
        self.topology = topology
        self.message_buffer: list[dict[str, Any]] = []

        # Set up agent indexing for graph-based communication
        if agent_ids is not None:
            self.agent_to_idx = {agent: idx for idx, agent in enumerate(agent_ids)}
            self.idx_to_agent = {idx: agent for agent, idx in self.agent_to_idx.items()}
        else:
            self.agent_to_idx = {}
            self.idx_to_agent = {}

        # Validate topology for graph-based communication
        if self.comm_type == "graph":
            if topology is None:
                raise ValueError("Graph communication requires a topology matrix")
            if agent_ids is not None and topology.shape[0] != len(agent_ids):
                raise ValueError(
                    f"Topology shape {topology.shape} doesn't match number of agents {len(agent_ids)}"
                )

    def send(
        self,
        sender_id: str,
        message: torch.Tensor,
        target_id: str | list[str] | None = None,
    ):
        """Send message from agent.

        Args:
            sender_id: ID of the sending agent
            message: Message tensor to send
            target_id: Target agent ID(s) for targeted communication

        """
        if message.shape[-1] != self.message_dim:
            raise ValueError(
                f"Message dimension {message.shape[-1]} doesn't match expected {self.message_dim}"
            )

        msg_packet = {
            "sender": sender_id,
            "message": message.detach().clone(),  # Detach to avoid gradient issues
            "target": target_id,
            "timestamp": time.time(),
        }
        self.message_buffer.append(msg_packet)

    def receive(self, receiver_id: str) -> list[torch.Tensor]:
        """Receive messages for agent.

        Args:
            receiver_id: ID of the receiving agent

        Returns:
            List of message tensors received by the agent

        """
        messages = []

        for packet in self.message_buffer:
            if self._should_receive(receiver_id, packet):
                messages.append(packet["message"])

        # Limit number of messages
        if len(messages) > self.max_messages:
            # Take most recent messages
            messages = messages[-self.max_messages :]

        return messages

    def _should_receive(self, receiver_id: str, packet: dict[str, Any]) -> bool:
        """Determine if receiver should get message.

        Args:
            receiver_id: ID of potential receiver
            packet: Message packet

        Returns:
            True if receiver should get the message

        """
        if self.comm_type == "broadcast":
            # Broadcast: everyone except sender receives
            return packet["sender"] != receiver_id

        elif self.comm_type == "targeted":
            # Targeted: only specified recipients
            target = packet["target"]
            if target is None:
                return False
            if isinstance(target, list):
                return receiver_id in target
            return packet["target"] == receiver_id

        elif self.comm_type == "graph":
            # Graph: based on topology matrix
            if receiver_id not in self.agent_to_idx or packet["sender"] not in self.agent_to_idx:
                return False
            sender_idx = self.agent_to_idx[packet["sender"]]
            receiver_idx = self.agent_to_idx[receiver_id]
            return self.topology[sender_idx, receiver_idx] > 0

        return False

    def clear(self):
        """Clear message buffer for next step."""
        self.message_buffer = []

    def update_topology(self, new_topology: np.ndarray):
        """Update communication topology for graph-based communication.

        Args:
            new_topology: New adjacency matrix

        """
        if self.comm_type != "graph":
            raise ValueError("Topology updates only apply to graph communication")
        self.topology = new_topology

    def get_buffer_size(self) -> int:
        """Get current number of messages in buffer."""
        return len(self.message_buffer)


class MessageEncoder(nn.Module):
    """Neural network module for encoding messages.

    Args:
        input_dim: Dimension of input features
        message_dim: Dimension of output message
        hidden_dim: Hidden layer dimension

    """

    def __init__(self, input_dim: int, message_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh(),  # Normalize messages to [-1, 1]
        )
        self.message_dim = message_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into message."""
        return self.encoder(x)


class MessageDecoder(nn.Module):
    """Neural network module for decoding received messages.

    Args:
        message_dim: Dimension of input messages
        output_dim: Dimension of decoded output
        hidden_dim: Hidden layer dimension
        aggregation: How to aggregate multiple messages

    """

    def __init__(
        self,
        message_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        aggregation: Literal["mean", "sum", "max", "attention"] = "mean",
    ):
        super().__init__()
        self.aggregation = aggregation
        self.output_dim = output_dim

        # Message processing network
        self.processor = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Attention mechanism for aggregation
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(message_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )

    def forward(self, messages: list[torch.Tensor]) -> torch.Tensor:
        """Decode and aggregate received messages.

        Args:
            messages: List of message tensors

        Returns:
            Aggregated message information

        """
        if not messages:
            # Return zeros if no messages
            return torch.zeros(self.output_dim)

        # Stack messages for batch processing
        if len(messages) == 1:
            stacked = messages[0].unsqueeze(0) if messages[0].dim() == 1 else messages[0]
        else:
            stacked = torch.stack(messages)  # Shape: (num_messages, message_dim)

        # Process messages
        processed = self.processor(stacked)  # Shape: (num_messages, output_dim)

        # Aggregate messages
        if self.aggregation == "mean":
            return processed.mean(dim=0)
        elif self.aggregation == "sum":
            return processed.sum(dim=0)
        elif self.aggregation == "max":
            return processed.max(dim=0)[0]
        elif self.aggregation == "attention":
            # Compute attention weights
            attention_logits = self.attention(stacked)  # Shape: (num_messages, 1)
            attention_weights = F.softmax(attention_logits, dim=0)
            # Weighted sum
            return (processed * attention_weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")


class CommunicatingPolicy(nn.Module):
    """Policy with communication capabilities.

    This policy wraps an existing policy and adds communication functionality.
    Messages are exchanged between agents and incorporated into the decision-making
    process.

    Args:
        actor: Actor network that takes enhanced observations and outputs actions
        comm_channel: Communication channel for message passing
        comm_encoder: Module to encode messages
        comm_decoder: Module to decode received messages
        obs_dim: Dimension of observations
        act_dim: Dimension of actions
        agent_id: ID of this agent
        communication_enabled: Whether to use communication

    """

    def __init__(
        self,
        actor: nn.Module,
        comm_channel: CommunicationChannel,
        comm_encoder: MessageEncoder,
        comm_decoder: MessageDecoder,
        obs_dim: int,
        act_dim: int,
        agent_id: str,
        communication_enabled: bool = True,
    ):
        super().__init__()
        self.actor = actor
        self.comm_channel = comm_channel
        self.comm_encoder = comm_encoder
        self.comm_decoder = comm_decoder
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_id = agent_id
        self.communication_enabled = communication_enabled

        # Enhanced observation dimension (obs + comm_info)
        self.enhanced_obs_dim = obs_dim + comm_decoder.output_dim

    def forward(self, batch: Batch, state: Any | None = None, **kwargs) -> Batch:
        """Forward pass with communication.

        Args:
            batch: Input batch with observations
            state: Optional state for recurrent policies
            **kwargs: Additional keyword arguments (for compatibility)

        Returns:
            Batch with actions and updated state

        """
        obs = batch.obs

        # Ensure obs is a tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        # Handle communication
        if self.communication_enabled:
            # Receive messages from other agents
            messages = self.comm_channel.receive(self.agent_id)

            # Process messages
            if messages:
                comm_info = self.comm_decoder(messages)
            else:
                comm_info = torch.zeros(self.comm_decoder.output_dim)

            # Ensure comm_info has same batch dimension as obs
            if obs.dim() > 1 and comm_info.dim() == 1:
                comm_info = comm_info.unsqueeze(0).expand(obs.shape[0], -1)

            # Combine observation with communication
            enhanced_obs = torch.cat([obs, comm_info], dim=-1)

            # Generate action and message using enhanced observation
            actor_output = self.actor(enhanced_obs, state)

            # Extract action and state
            if isinstance(actor_output, tuple):
                action, new_state = actor_output
            else:
                action = actor_output
                new_state = state

            # Generate and send message based on current observation
            message = self.comm_encoder(obs)
            self.comm_channel.send(self.agent_id, message)
        else:
            # No communication, just use regular observation
            actor_output = self.actor(obs, state)

            if isinstance(actor_output, tuple):
                action, new_state = actor_output
            else:
                action = actor_output
                new_state = state

        return Batch(act=action, state=new_state)

    def learn(self, batch: Batch, **kwargs) -> dict[str, float]:
        """Learning update for the policy.

        This is a placeholder that should be overridden by specific algorithms.

        Args:
            batch: Training batch
            **kwargs: Additional keyword arguments (algorithm-specific)

        Returns:
            Dictionary of training statistics

        """
        # This should be implemented by specific algorithm classes
        raise NotImplementedError("Learn method must be implemented by specific algorithm")

    def set_communication_enabled(self, enabled: bool):
        """Enable or disable communication.

        Args:
            enabled: Whether to enable communication

        """
        self.communication_enabled = enabled

    def reset_communication(self):
        """Reset communication channel (clear message buffer)."""
        self.comm_channel.clear()


class MultiAgentCommunicationWrapper:
    """Wrapper to manage communication for multiple agents.

    This class coordinates communication between multiple CommunicatingPolicy instances.

    Args:
        policies: Dictionary mapping agent IDs to CommunicatingPolicy instances
        comm_channel: Shared communication channel

    """

    def __init__(
        self, policies: dict[str, CommunicatingPolicy], comm_channel: CommunicationChannel
    ):
        self.policies = policies
        self.comm_channel = comm_channel

        # Ensure all policies use the same channel
        for policy in policies.values():
            policy.comm_channel = comm_channel

    def step(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Execute one step for all agents with communication.

        Args:
            observations: Dictionary mapping agent IDs to observations

        Returns:
            Dictionary mapping agent IDs to actions

        """
        # Clear previous messages
        self.comm_channel.clear()

        actions = {}

        # First pass: all agents generate messages
        for agent_id, obs in observations.items():
            if agent_id in self.policies:
                batch = Batch(obs=obs)
                # This will generate both action and message
                result = self.policies[agent_id].forward(batch)
                actions[agent_id] = result.act

        return actions

    def reset(self):
        """Reset communication for all agents."""
        self.comm_channel.clear()
        for policy in self.policies.values():
            policy.reset_communication()

    def set_communication_enabled(self, enabled: bool):
        """Enable or disable communication for all agents."""
        for policy in self.policies.values():
            policy.set_communication_enabled(enabled)
