#!/usr/bin/env python3
"""
Simple MARL Communication Demo.

A simplified demonstration of agent communication that works with the current architecture.
This example shows basic message passing between agents in a multi-agent environment.
"""

import argparse
import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from pettingzoo.mpe import simple_spread_v3

from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.multiagent import (
    CommunicationChannel,
    FlexibleMultiAgentPolicyManager,
    MessageDecoder,
    MessageEncoder,
)
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv
from tianshou.trainer import OffPolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


class CommunicatingQNetwork(nn.Module):
    """Q-network that incorporates communication messages."""
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        message_dim: int,
        hidden_sizes: list[int],
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.message_dim = message_dim
        self.device = device
        
        # Message encoder: encodes observations into messages to send
        self.message_encoder = MessageEncoder(
            input_dim=obs_dim,
            message_dim=message_dim,
            hidden_dim=64,
        )
        
        # Message decoder: processes received messages
        self.message_decoder = MessageDecoder(
            message_dim=message_dim,
            output_dim=message_dim,
            hidden_dim=64,
            aggregation="mean",  # Use mean aggregation for simplicity
        )
        
        # Q-network that takes observation + decoded messages
        input_dim = obs_dim + message_dim
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, act_dim))
        
        self.q_network = nn.Sequential(*layers)
        
        # Store for communication
        self.last_obs = None
        self.received_messages = []
        
    def forward(self, obs: torch.Tensor, state: Any = None, info: dict = {}) -> torch.Tensor:
        """Forward pass with communication."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        # Store observation for message generation
        self.last_obs = obs
        
        # Process received messages
        if self.received_messages:
            comm_info = self.message_decoder(self.received_messages)
        else:
            comm_info = torch.zeros(self.message_dim, device=self.device)
        
        # Ensure comm_info has same batch dimension as obs
        if obs.dim() > 1 and comm_info.dim() == 1:
            comm_info = comm_info.unsqueeze(0).expand(obs.shape[0], -1)
        elif obs.dim() == 1 and comm_info.dim() == 1:
            # Both are 1D, no expansion needed
            pass
        
        # Combine observation with communication info
        enhanced_obs = torch.cat([obs, comm_info], dim=-1)
        
        # Get Q-values
        q_values = self.q_network(enhanced_obs)
        
        return q_values
    
    def generate_message(self) -> torch.Tensor | None:
        """Generate a message based on the last observation."""
        if self.last_obs is not None:
            return self.message_encoder(self.last_obs)
        return None
    
    def receive_messages(self, messages: list[torch.Tensor]) -> None:
        """Receive messages from other agents."""
        self.received_messages = messages
    
    def clear_messages(self) -> None:
        """Clear received messages."""
        self.received_messages = []


class CommunicatingPolicyWrapper:
    """Wrapper that adds communication to a policy."""
    
    def __init__(
        self,
        policy: DiscreteQLearningPolicy,
        comm_network: CommunicatingQNetwork,
        comm_channel: CommunicationChannel,
        agent_id: str,
    ):
        self.policy = policy
        self.comm_network = comm_network
        self.comm_channel = comm_channel
        self.agent_id = agent_id
        
    def __call__(self, batch: Batch, state: Any = None, **kwargs) -> Batch:
        """Forward with communication."""
        # Receive messages
        messages = self.comm_channel.receive(self.agent_id)
        self.comm_network.receive_messages(messages)
        
        # Get action from policy
        result = self.policy(batch, state, **kwargs)
        
        # Send message
        message = self.comm_network.generate_message()
        if message is not None:
            self.comm_channel.send(self.agent_id, message)
        
        return result
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped policy."""
        return getattr(self.policy, name)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=2)
    parser.add_argument("--test-num", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--with-communication", action="store_true", default=False)
    return parser.parse_args()


def create_agents(
    env: Any,
    args: argparse.Namespace,
) -> tuple[FlexibleMultiAgentPolicyManager, list[torch.optim.Adam], CommunicationChannel | None]:
    """Create agents with optional communication."""
    agents = env.agents
    
    # Create communication channel if needed
    comm_channel = None
    if args.with_communication:
        comm_channel = CommunicationChannel(
            comm_type="broadcast",
            agent_ids=agents,
            message_dim=args.message_dim,
        )
    
    policies = {}
    optimizers = []
    
    # Get environment spaces
    obs_shape = env.observation_space.shape
    action_space = env.action_space
    obs_dim = obs_shape[0] if len(obs_shape) > 0 else 1
    act_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
    
    for agent_id in agents:
        if args.with_communication:
            # Create communicating Q-network
            model = CommunicatingQNetwork(
                obs_dim=obs_dim,
                act_dim=act_dim,
                message_dim=args.message_dim,
                hidden_sizes=[128, 128],
                device=args.device,
            ).to(args.device)
        else:
            # Standard Q-network
            model = Net(
                state_shape=(obs_dim,),
                action_shape=act_dim,
                hidden_sizes=[128, 128],
            ).to(args.device)
        
        # Create optimizer
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Create Q-learning policy
        policy = DiscreteQLearningPolicy(
            model=model,
            action_space=action_space,
            observation_space=env.observation_space,
            eps_training=args.eps_train,
            eps_inference=args.eps_test,
        ).to(args.device)
        
        # Wrap with communication if needed
        if args.with_communication and comm_channel is not None:
            policy = CommunicatingPolicyWrapper(
                policy=policy,
                comm_network=model,
                comm_channel=comm_channel,
                agent_id=agent_id,
            )
        
        policies[agent_id] = policy
        optimizers.append(optim)
    
    # Create policy manager
    policy_manager = FlexibleMultiAgentPolicyManager(policies, env, mode="independent")
    
    return policy_manager, optimizers, comm_channel


def train_agents(args: argparse.Namespace = get_args()) -> dict:
    """Train agents with optional communication."""
    
    # Environment setup - simple_spread is a cooperative environment
    def env_fn():
        return simple_spread_v3.env(N=3, max_cycles=25, continuous_actions=False)
    
    # Create vectorized environments
    train_envs = DummyVectorEnv(
        [lambda: EnhancedPettingZooEnv(env_fn()) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: EnhancedPettingZooEnv(env_fn()) for _ in range(args.test_num)]
    )
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # Create sample environment for agent setup
    sample_env = EnhancedPettingZooEnv(env_fn())
    
    # Create agents
    policy_manager, optimizers, comm_channel = create_agents(sample_env, args)
    
    # Create combined optimizer
    from itertools import chain
    combined_params = chain(*[opt.param_groups[0]["params"] for opt in optimizers])
    combined_optim = torch.optim.Adam(combined_params, lr=args.lr)
    
    # Create replay buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs),
        ignore_obs_next=True,
    )
    
    # Create collectors
    train_collector = Collector(policy_manager, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy_manager, test_envs, exploration_noise=True)
    
    # Reset collectors
    train_collector.reset()
    test_collector.reset()
    
    # Collect initial samples
    train_collector.collect(n_step=args.batch_size * args.training_num)
    
    # Create logger
    comm_str = "with_comm" if args.with_communication else "no_comm"
    log_path = f"{args.logdir}/simple_{comm_str}"
    os.makedirs(log_path, exist_ok=True)
    logger = TensorboardLogger(log_path)
    
    def save_best_fn(policy) -> None:
        torch.save(policy.state_dict(), f"{log_path}/policy.pth")
    
    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards > 10.0
    
    def train_fn(epoch: int, env_step: int) -> None:
        # Clear communication channel at the start of each epoch
        if comm_channel is not None:
            comm_channel.clear()
        
        # Decay exploration
        eps = max(args.eps_train * (0.95 ** (epoch // 5)), 0.05)
        if hasattr(policy_manager, "policies"):
            for agent_policy in policy_manager.policies.values():
                if hasattr(agent_policy, "policy"):
                    # It's a wrapper
                    agent_policy.policy.set_eps(eps)
                elif hasattr(agent_policy, "set_eps"):
                    agent_policy.set_eps(eps)
    
    def test_fn(epoch: int, env_step: int) -> None:
        # Clear communication channel for testing
        if comm_channel is not None:
            comm_channel.clear()
            
        if hasattr(policy_manager, "policies"):
            for agent_policy in policy_manager.policies.values():
                if hasattr(agent_policy, "policy"):
                    # It's a wrapper
                    agent_policy.policy.set_eps(args.eps_test)
                elif hasattr(agent_policy, "set_eps"):
                    agent_policy.set_eps(args.eps_test)
    
    # Start training
    result = OffPolicyTrainer(
        policy=policy_manager,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        update_per_step=args.update_per_step,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()
    
    return result


if __name__ == "__main__":
    args = get_args()
    
    print("=" * 60)
    print("Simple Communication Demo")
    print("=" * 60)
    print(f"Communication: {'ENABLED' if args.with_communication else 'DISABLED'}")
    print(f"Device: {args.device}")
    print(f"Message dimension: {args.message_dim if args.with_communication else 'N/A'}")
    print(f"Training epochs: {args.epoch}")
    print(f"Buffer size: {args.buffer_size}")
    print("=" * 60)
    
    # Train agents
    result = train_agents(args)
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    comm_status = "WITH communication" if args.with_communication else "WITHOUT communication"
    print(f"Training completed {comm_status}")
    print(f"Best reward: {result.get('best_reward', 'N/A')}")
    print(f"Training duration: {result.get('duration', 'N/A')} seconds")
    print("=" * 60)
    
    # Suggestion for comparison
    if args.with_communication:
        print("\nTo compare performance without communication:")
        print("  python simple_communication_demo.py")
    else:
        print("\nTo enable communication and compare:")
        print("  python simple_communication_demo.py --with-communication")
    print("=" * 60)