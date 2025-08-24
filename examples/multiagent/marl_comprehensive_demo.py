#!/usr/bin/env python3
"""
Comprehensive MARL Demo showcasing Tianshou's multi-agent capabilities.

This example demonstrates:
1. Enhanced PettingZoo environment integration
2. Flexible policy management with parameter sharing
3. CTDE (Centralized Training, Decentralized Execution) algorithms
4. Communication between agents
5. Different training coordination modes

Requirements:
- pip install pettingzoo[classic]
"""

import argparse
import os
from typing import Any

import numpy as np
import torch
from pettingzoo.classic.connect_four_v3 import env as connect_four_env
from pettingzoo.classic.tictactoe_v3 import env as tic_tac_toe_env

from tianshou.algorithm import DQN
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.multiagent import (
    CTDEPolicy,
    FlexibleMultiAgentPolicyManager,
    QMIXPolicy,
    SimultaneousTrainer,
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
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--logger", type=str, default="tensorboard")
    parser.add_argument("--wandb-project", type=str, default="tianshou")
    parser.add_argument("--watch", action="store_true", default=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--algorithm", type=str, default="independent", 
                       choices=["independent", "shared", "qmix"])
    parser.add_argument("--environment", type=str, default="tic_tac_toe", 
                       choices=["tic_tac_toe", "connect_four"])
    return parser.parse_args()


def get_agents(
    env: Any,
    args: argparse.Namespace,
) -> tuple[FlexibleMultiAgentPolicyManager, torch.optim.Adam, list]:
    """
    Create multi-agent policies based on the chosen algorithm.
    
    Args:
        env: The environment
        args: Command line arguments
        
    Returns:
        Tuple of (policy_manager, optimizer, agents_list)
    """
    # Get environment info
    observation_space = env.observation_space
    action_space = env.action_space
    agents = env.agents
    
    # Create networks and policies for each agent
    if args.algorithm == "independent":
        # Independent learning - each agent has its own policy
        policies = {}
        optimizers = []
        
        for agent_id in agents:
            # Create network
            model = Net(
                state_shape=observation_space[agent_id].shape,
                action_shape=action_space[agent_id].n, 
                hidden_sizes=[128, 128, 128], 
                device=args.device
            ).to(args.device)
            
            # Create DQN policy
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
            policy = DiscreteQLearningPolicy(
                model=model,
                action_space=action_space[agent_id],
                observation_space=observation_space[agent_id],
                eps_training=args.eps_train,
                eps_inference=args.eps_test,
            ).to(args.device)
            
            policies[agent_id] = policy
            optimizers.append(optim)
        
        # Create flexible policy manager
        policy_manager = FlexibleMultiAgentPolicyManager(
            policies, env, mode="independent"
        )
        
        # Combine optimizers (for logging purposes)
        from itertools import chain
        combined_params = chain(*[opt.param_groups[0]["params"] for opt in optimizers])
        combined_optim = torch.optim.Adam(combined_params, lr=args.lr)
        
    elif args.algorithm == "shared":
        # Parameter sharing - all agents share the same policy
        model = Net(
            state_shape=list(observation_space.values())[0].shape, 
            action_shape=list(action_space.values())[0].n, 
            hidden_sizes=[128, 128, 128], 
            device=args.device
        ).to(args.device)
        
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        shared_policy = DiscreteQLearningPolicy(
            model=model,
            action_space=list(action_space.values())[0],
            observation_space=list(observation_space.values())[0],
            eps_training=args.eps_train,
            eps_inference=args.eps_test,
        ).to(args.device)
        
        policy_manager = FlexibleMultiAgentPolicyManager(
            shared_policy, env, mode="shared"
        )
        combined_optim = optim
        
    elif args.algorithm == "qmix":
        # QMIX - CTDE with value mixing
        policies = {}
        
        for agent_id in agents:
            model = Net(
                state_shape=observation_space[agent_id].shape, 
                action_shape=action_space[agent_id].n, 
                hidden_sizes=[64, 64], 
                device=args.device
            ).to(args.device)
            
            policy = DiscreteQLearningPolicy(
                model=model,
                action_space=action_space[agent_id],
                observation_space=observation_space[agent_id],
                eps_training=args.eps_train,
                eps_inference=args.eps_test,
            ).to(args.device)
            
            policies[agent_id] = policy
        
        # Create QMIX policy with centralized mixing network
        qmix_policy = QMIXPolicy(
            policies=policies,
            env=env,
            lr=args.lr,
            device=args.device
        )
        
        policy_manager = FlexibleMultiAgentPolicyManager(
            qmix_policy, env, mode="custom"
        )
        combined_optim = qmix_policy.optim
    
    return policy_manager, combined_optim, agents


def train_agent(
    args: argparse.Namespace = get_args(),
) -> tuple[dict, FlexibleMultiAgentPolicyManager]:
    """Train multi-agent policies."""
    # Environment setup
    if args.environment == "tic_tac_toe":
        env_fn = tic_tac_toe_env
    elif args.environment == "connect_four":
        env_fn = connect_four_env
    else:
        raise ValueError(f"Unknown environment: {args.environment}")
    
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
    
    # Get sample environment for agent creation
    sample_env = EnhancedPettingZooEnv(env_fn())
    
    # Create agents
    policy, optim, agents = get_agents(sample_env, args)
    
    # Create replay buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs),
        ignore_obs_next=True,
    )
    
    # Create collectors
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    
    # Set up training parameters
    train_collector.collect(n_step=args.batch_size * args.training_num)
    
    # Create logger
    if args.logger == "wandb":
        try:
            import wandb
            logger = wandb.WandbLogger(
                save_interval=1,
                name=f"marl_{args.algorithm}_{args.environment}",
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        except ImportError:
            print("wandb not installed, falling back to tensorboard")
            logger = TensorboardLogger(args.logdir)
    else:
        logger = TensorboardLogger(args.logdir)
    
    def save_best_fn(policy: FlexibleMultiAgentPolicyManager) -> None:
        if hasattr(policy, 'policies'):
            for agent_id, agent_policy in policy.policies.items():
                torch.save(agent_policy.state_dict(), f"{args.logdir}/policy_{agent_id}.pth")
        else:
            torch.save(policy.state_dict(), f"{args.logdir}/policy.pth")
    
    def stop_fn(mean_rewards: float) -> bool:
        # Stop training if average reward > 0 (winning more than losing)
        return mean_rewards > 0.1
    
    def train_fn(epoch: int, env_step: int) -> None:
        # Decay exploration
        eps = max(args.eps_train * (0.99 ** (epoch // 10)), 0.01)
        if hasattr(policy, 'policies'):
            for agent_policy in policy.policies.values():
                agent_policy.set_eps(eps)
        else:
            policy.set_eps(eps)
    
    def test_fn(epoch: int, env_step: int) -> None:
        if hasattr(policy, 'policies'):
            for agent_policy in policy.policies.values():
                agent_policy.set_eps(args.eps_test)
        else:
            policy.set_eps(args.eps_test)
    
    # Start training
    result = OffPolicyTrainer(
        policy=policy,
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
    
    return result, policy


def watch_agent(
    policy: FlexibleMultiAgentPolicyManager,
    args: argparse.Namespace = get_args(),
) -> None:
    """Watch trained agent play."""
    # Set up test environment  
    if args.environment == "tic_tac_toe":
        env = EnhancedPettingZooEnv(tic_tac_toe_env(render_mode="human"))
    elif args.environment == "connect_four":
        env = EnhancedPettingZooEnv(connect_four_env(render_mode="human"))
    
    # Set policies to evaluation mode
    if hasattr(policy, 'policies'):
        for agent_policy in policy.policies.values():
            agent_policy.eval()
            agent_policy.set_eps(args.eps_test)
    else:
        policy.eval()
        policy.set_eps(args.eps_test)
    
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    
    print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")


if __name__ == "__main__":
    args = get_args()
    
    # Create log directory
    os.makedirs(args.logdir, exist_ok=True)
    
    print(f"Training {args.algorithm} on {args.environment}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Environment: {args.environment}")
    print(f"Device: {args.device}")
    
    # Train or watch
    if args.watch:
        # Load trained policy (implement loading logic here)
        print("Watching requires a trained policy - train first!")
    else:
        result, policy = train_agent(args)
        print(f"Training completed. Best reward: {result['best_reward']}")
        
        # Optionally watch the trained agent
        if args.render > 0:
            watch_agent(policy, args)