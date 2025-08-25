#!/usr/bin/env python3
"""
Targeted Communication Demo for MARL.

This example demonstrates targeted (selective) communication between specific agents.
Agents can send messages to specific teammates based on the situation.
"""

import argparse
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tianshou.algorithm.multiagent import (
    CommunicationChannel,
    MessageDecoder,
    MessageEncoder,
)
from tianshou.data import Batch


class TargetedCommunicationAgent:
    """Agent that can send targeted messages to specific teammates."""
    
    def __init__(
        self,
        agent_id: str,
        obs_dim: int,
        act_dim: int,
        message_dim: int,
        all_agent_ids: list[str],
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.message_dim = message_dim
        self.all_agent_ids = all_agent_ids
        self.device = device
        
        # Message encoder for creating messages
        self.encoder = MessageEncoder(
            input_dim=obs_dim,
            message_dim=message_dim,
            hidden_dim=32,
        ).to(device)
        
        # Message decoder for processing received messages
        self.decoder = MessageDecoder(
            message_dim=message_dim,
            output_dim=message_dim,
            hidden_dim=32,
            aggregation="attention",  # Use attention to weight messages
        ).to(device)
        
        # Decision network (takes obs + decoded messages)
        input_dim = obs_dim + message_dim
        self.decision_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        ).to(device)
        
        # Target selection network (decides who to send messages to)
        self.target_net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(all_agent_ids)),
            nn.Softmax(dim=-1),
        ).to(device)
    
    def select_target(self, obs: torch.Tensor) -> str | None:
        """Select which agent to send a message to based on observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        # Get target probabilities
        target_probs = self.target_net(obs)
        
        # Sample a target (excluding self)
        valid_indices = [i for i, aid in enumerate(self.all_agent_ids) if aid != self.agent_id]
        if not valid_indices:
            return None
        
        # Mask out self and renormalize
        masked_probs = target_probs.clone()
        self_idx = self.all_agent_ids.index(self.agent_id)
        masked_probs[self_idx] = 0
        masked_probs = masked_probs / masked_probs.sum()
        
        # Sample target based on probabilities
        target_idx = torch.multinomial(masked_probs, 1).item()
        
        # Sometimes don't send any message (20% chance)
        if torch.rand(1).item() < 0.2:
            return None
        
        return self.all_agent_ids[target_idx]
    
    def act(
        self,
        obs: torch.Tensor,
        received_messages: list[torch.Tensor],
    ) -> int:
        """Choose action based on observation and received messages."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        # Decode received messages
        if received_messages:
            comm_info = self.decoder(received_messages)
        else:
            comm_info = torch.zeros(self.message_dim, device=self.device)
        
        # Combine observation with communication
        enhanced_obs = torch.cat([obs, comm_info], dim=-1)
        
        # Get action logits
        logits = self.decision_net(enhanced_obs)
        
        # Sample action (add some exploration noise)
        if torch.rand(1).item() < 0.1:  # 10% random actions
            action = torch.randint(0, self.act_dim, (1,)).item()
        else:
            action = torch.argmax(logits).item()
        
        return action
    
    def generate_message(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate a message based on observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        return self.encoder(obs)


def demonstrate_targeted_communication():
    """Demonstrate targeted communication between agents."""
    
    # Setup
    num_agents = 4
    obs_dim = 10
    act_dim = 5
    message_dim = 8
    device = "cpu"
    
    # Create agent IDs
    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    
    # Create communication channel with targeted mode
    comm_channel = CommunicationChannel(
        comm_type="targeted",
        agent_ids=agent_ids,
        message_dim=message_dim,
        max_messages=5,
    )
    
    # Create agents
    agents = {}
    for agent_id in agent_ids:
        agents[agent_id] = TargetedCommunicationAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            act_dim=act_dim,
            message_dim=message_dim,
            all_agent_ids=agent_ids,
            device=device,
        )
    
    print("=" * 60)
    print("Targeted Communication Demo")
    print("=" * 60)
    print(f"Number of agents: {num_agents}")
    print(f"Message dimension: {message_dim}")
    print("Communication type: Targeted")
    print("=" * 60)
    
    # Simulate several timesteps
    num_timesteps = 5
    
    for t in range(num_timesteps):
        print(f"\n--- Timestep {t + 1} ---")
        
        # Clear previous messages
        comm_channel.clear()
        
        # Generate random observations for each agent
        observations = {}
        for agent_id in agent_ids:
            observations[agent_id] = torch.randn(obs_dim)
        
        # Phase 1: Agents decide on targets and send messages
        print("\nPhase 1: Sending Messages")
        for agent_id, agent in agents.items():
            obs = observations[agent_id]
            
            # Select target for message
            target = agent.select_target(obs)
            
            if target is not None:
                # Generate and send message
                message = agent.generate_message(obs)
                comm_channel.send(agent_id, message, target_id=target)
                print(f"  {agent_id} -> {target}")
            else:
                print(f"  {agent_id} -> (no message)")
        
        # Phase 2: Agents receive messages and take actions
        print("\nPhase 2: Receiving Messages and Acting")
        actions = {}
        for agent_id, agent in agents.items():
            # Receive messages
            messages = comm_channel.receive(agent_id)
            num_messages = len(messages)
            
            # Take action
            obs = observations[agent_id]
            action = agent.act(obs, messages)
            actions[agent_id] = action
            
            print(f"  {agent_id}: received {num_messages} message(s), action={action}")
        
        # Show message buffer statistics
        print(f"\nBuffer size: {comm_channel.get_buffer_size()} messages")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def demonstrate_graph_communication():
    """Demonstrate graph-based communication topology."""
    
    print("\n" + "=" * 60)
    print("Graph-Based Communication Demo")
    print("=" * 60)
    
    # Setup
    num_agents = 4
    obs_dim = 10
    message_dim = 8
    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    
    # Define communication topology (adjacency matrix)
    # Agent 0 can send to agents 1 and 2
    # Agent 1 can send to agent 2
    # Agent 2 can send to agent 3
    # Agent 3 can send to agent 0
    topology = np.array([
        [0, 1, 1, 0],  # Agent 0's connections
        [0, 0, 1, 0],  # Agent 1's connections
        [0, 0, 0, 1],  # Agent 2's connections
        [1, 0, 0, 0],  # Agent 3's connections
    ])
    
    print("Communication topology (adjacency matrix):")
    print(topology)
    print("\nConnections:")
    for i, sender in enumerate(agent_ids):
        receivers = [agent_ids[j] for j in range(num_agents) if topology[i, j] > 0]
        if receivers:
            print(f"  {sender} can send to: {', '.join(receivers)}")
        else:
            print(f"  {sender} cannot send messages")
    
    # Create communication channel with graph topology
    comm_channel = CommunicationChannel(
        comm_type="graph",
        agent_ids=agent_ids,
        message_dim=message_dim,
        topology=topology,
    )
    
    # Create simple encoders for demonstration
    encoders = {}
    for agent_id in agent_ids:
        encoders[agent_id] = MessageEncoder(
            input_dim=obs_dim,
            message_dim=message_dim,
            hidden_dim=32,
        )
    
    print("\n--- Simulating Communication ---")
    
    # Each agent sends a message
    for i, sender_id in enumerate(agent_ids):
        obs = torch.randn(obs_dim)
        message = encoders[sender_id](obs)
        comm_channel.send(sender_id, message)
        print(f"  {sender_id} sent a message")
    
    # Check who receives messages
    print("\n--- Message Reception ---")
    for receiver_id in agent_ids:
        messages = comm_channel.receive(receiver_id)
        print(f"  {receiver_id} received {len(messages)} message(s)")
        
        # Show which agents sent messages (based on topology)
        senders = []
        for j, sender_id in enumerate(agent_ids):
            if topology[j, agent_ids.index(receiver_id)] > 0:
                senders.append(sender_id)
        if senders:
            print(f"    from: {', '.join(senders)}")
    
    print("\n" + "=" * 60)
    print("Graph communication demo completed!")
    print("=" * 60)


def main():
    """Run communication demonstrations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        type=str,
        default="targeted",
        choices=["targeted", "graph", "both"],
        help="Which demo to run",
    )
    args = parser.parse_args()
    
    if args.demo in ["targeted", "both"]:
        demonstrate_targeted_communication()
    
    if args.demo in ["graph", "both"]:
        demonstrate_graph_communication()
    
    print("\nKey Insights:")
    print("- Targeted communication allows selective message passing")
    print("- Graph topology enforces structured communication patterns")
    print("- Attention-based aggregation helps weight important messages")
    print("- These patterns are useful for hierarchical or team-based scenarios")


if __name__ == "__main__":
    main()