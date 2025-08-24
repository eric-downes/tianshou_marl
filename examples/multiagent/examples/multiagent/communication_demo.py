#!/usr/bin/env python3
"""
MARL Communication Demo

Demonstrates agent-to-agent communication using Tianshou's communication framework.
This example shows how agents can share information to coordinate their actions.

Requirements:
- pip install pettingzoo[classic]
"""

import argparse
import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.butterfly.pistonball_v6 import env as pistonball_env

from tianshou.algorithm import DQN
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.multiagent import (
    CommunicatingPolicy,
    CommunicationChannel,
    FlexibleMultiAgentPolicyManager,
    MessageDecoder,
    MessageEncoder,
    MultiAgentCommunicationWrapper,
)
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv
from tianshou.trainer import OffPolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


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
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=4)
    parser.add_argument("--test-num", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--message-dim", type=int, default=32)
    parser.add_argument("--with-communication", action="store_true", default=False)
    return parser.parse_args()


def create_communicating_agents(
    env: Any,
    args: argparse.Namespace,
) -> tuple[FlexibleMultiAgentPolicyManager, torch.optim.Adam]:
    """Create agents with communication capabilities."""
    observation_space = env.observation_space
    action_space = env.action_space
    agents = env.agents
    
    # Create communication channel
    comm_channel = CommunicationChannel(
        agent_ids=agents,
        message_dim=args.message_dim,
        topology="broadcast"  # All agents can communicate with all others
    )
    
    policies = {}
    optimizers = []
    
    for agent_id in agents:
        obs_dim = observation_space[agent_id].shape[0]
        
        if args.with_communication:
            # Add message dimension to observation space for communication
            # The decoder will add message features to observations
            decoder = MessageDecoder(
                message_dim=args.message_dim,
                output_dim=args.message_dim,
                aggregation_method="attention"
            )
            
            # Adjust network input size to account for decoded messages
            net_input_dim = obs_dim + args.message_dim
        else:
            net_input_dim = obs_dim
            decoder = None
        
        # Create base network model
        model = Net(
            state_shape=(net_input_dim,), 
            action_shape=action_space[agent_id].n, 
            hidden_sizes=[128, 128], 
            device=args.device
        ).to(args.device)
        
        # Create base policy
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        base_policy = DiscreteQLearningPolicy(
            model=model,
            action_space=action_space[agent_id],
            observation_space=observation_space[agent_id] if not args.with_communication else gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(net_input_dim,)
            ),
            eps_training=args.eps_train,
            eps_inference=args.eps_test,
        ).to(args.device)
        
        if args.with_communication:
            # Create message encoder
            encoder = MessageEncoder(
                input_dim=obs_dim,  # Encode from raw observations
                message_dim=args.message_dim
            )
            
            # Wrap policy with communication capabilities
            comm_policy = CommunicatingPolicy(
                base_policy=base_policy,
                encoder=encoder,
                decoder=decoder,
                communication_channel=comm_channel,
                agent_id=agent_id
            )
            policies[agent_id] = comm_policy
        else:
            policies[agent_id] = base_policy
            
        optimizers.append(optim)
    
    # Create policy manager
    policy_manager = FlexibleMultiAgentPolicyManager(
        policies, env, mode="independent"
    )
    
    # If using communication, wrap with communication wrapper
    if args.with_communication:
        policy_manager = MultiAgentCommunicationWrapper(
            policy_manager, comm_channel
        )
    
    # Create combined optimizer for logging
    from itertools import chain
    combined_params = chain(*[opt.param_groups[0]["params"] for opt in optimizers])
    combined_optim = torch.optim.Adam(combined_params, lr=args.lr)
    
    return policy_manager, combined_optim


def train_communicating_agents(args: argparse.Namespace = get_args()) -> dict:
    """Train agents with optional communication."""
    # Environment setup - use a cooperative environment where communication helps
    def env_fn():
        # PistonBall is a cooperative environment where agents need to coordinate
        return pistonball_env(n_pistons=5, continuous=False)
    
    # Create vectorized environments  
    train_envs = DummyVectorEnv([
        lambda: EnhancedPettingZooEnv(env_fn()) 
        for _ in range(args.training_num)
    ])
    test_envs = DummyVectorEnv([
        lambda: EnhancedPettingZooEnv(env_fn()) 
        for _ in range(args.test_num)
    ])
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # Create sample environment for agent setup
    sample_env = EnhancedPettingZooEnv(env_fn())
    
    # Create communicating agents
    policy, optim = create_communicating_agents(sample_env, args)
    
    # Create replay buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs),
        ignore_obs_next=True,
    )
    
    # Create collectors
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    
    # Collect initial samples
    train_collector.collect(n_step=args.batch_size * args.training_num)
    
    # Create logger
    log_path = f"{args.logdir}/comm_{args.with_communication}"
    os.makedirs(log_path, exist_ok=True)
    logger = TensorboardLogger(log_path)
    
    def save_best_fn(policy_to_save) -> None:
        torch.save(policy_to_save.state_dict(), f"{log_path}/policy.pth")
    
    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards > 15.0  # Environment-specific threshold
    
    def train_fn(epoch: int, env_step: int) -> None:
        # Decay exploration
        eps = max(args.eps_train * (0.95 ** (epoch // 5)), 0.05)
        if hasattr(policy, 'policies'):
            for agent_policy in policy.policies.values():
                if hasattr(agent_policy, 'base_policy'):
                    agent_policy.base_policy.set_eps(eps)
                else:
                    agent_policy.set_eps(eps)
        else:
            policy.set_eps(eps)
        
        print(f"Epoch {epoch}: eps = {eps:.3f}")
    
    def test_fn(epoch: int, env_step: int) -> None:
        if hasattr(policy, 'policies'):
            for agent_policy in policy.policies.values():
                if hasattr(agent_policy, 'base_policy'):
                    agent_policy.base_policy.set_eps(args.eps_test)
                else:
                    agent_policy.set_eps(args.eps_test)
        else:
            policy.set_eps(args.eps_test)
    
    # Start training
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        update_per_step=args.update_per_step,
        episode_per_test=args.test_num * 2,
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
    
    print(f"Communication Demo")
    print(f"Communication: {'Enabled' if args.with_communication else 'Disabled'}")
    print(f"Device: {args.device}")
    print(f"Message dimension: {args.message_dim}")
    
    # Train agents
    result = train_communicating_agents(args)
    
    comm_status = "with" if args.with_communication else "without"
    print(f"\nTraining completed {comm_status} communication:")
    print(f"Best reward: {result['best_reward']:.2f}")
    print(f"Training took {result.get('duration', 'N/A')} seconds")
    
    # Suggestion for comparison
    if args.with_communication:
        print("\nTo compare performance, run again with --with-communication flag removed")
    else:
        print("\nTo see communication benefits, run again with --with-communication flag")