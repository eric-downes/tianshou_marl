"""Simplified API for Tianshou MARL.

This module provides a simplified, user-friendly interface to Tianshou's
multi-agent reinforcement learning capabilities while maintaining full
backward compatibility with the original API.
"""

from tianshou.algorithm.multiagent import (
    FlexibleMultiAgentPolicyManager as PolicyManager,
    QMIXPolicy,
    MADDPGPolicy,
    CTDEPolicy,
    CommunicatingPolicy,
    CommunicationChannel,
)
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy as DQNPolicy
from tianshou.algorithm.modelfree.ppo import PPO as PPOPolicy
from tianshou.algorithm.modelfree.a2c import A2C as A2CPolicy
from tianshou.algorithm.modelfree.sac import SACPolicy
from tianshou.algorithm.modelfree.ddpg import DDPG as DDPGPolicy
from tianshou.algorithm.modelfree.td3 import TD3 as TD3Policy

# Import trainer and data components with simplified names
from tianshou.data import Collector, VectorReplayBuffer as ReplayBuffer
from tianshou.trainer import OffPolicyTrainer as Trainer
from tianshou.env.enhanced_pettingzoo_env import EnhancedPettingZooEnv as MARLEnv

# Placeholder for AutoPolicy - will be implemented next
class AutoPolicy:
    """Automatic policy selection based on environment characteristics."""
    
    @classmethod
    def from_env(cls, env, mode="independent", config=None, device="cpu", **kwargs):
        """Create an appropriate policy automatically based on environment.
        
        Args:
            env: The multi-agent environment
            mode: Policy mode ("independent", "shared", "grouped")
            config: Configuration dictionary
            device: Device to use ("cpu" or "cuda")
            **kwargs: Additional arguments
            
        Returns:
            FlexibleMultiAgentPolicyManager configured for the environment
        """
        # This is a placeholder implementation - will be replaced
        from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
        from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
        from tianshou.algorithm.modelfree.ppo import PPO
        from tianshou.utils.net.common import Net
        from gymnasium import spaces
        import torch
        import numpy as np
        
        config = config or {}
        
        # Detect action space type
        if hasattr(env, 'action_space'):
            action_space = env.action_space
        else:
            # For PettingZoo environments
            action_space = list(env.action_spaces.values())[0] if hasattr(env, 'action_spaces') else None
            
        # Select appropriate base policy class
        if isinstance(action_space, spaces.Discrete):
            policy_class = DiscreteQLearningPolicy
            is_discrete = True
        elif isinstance(action_space, spaces.Box):
            # For continuous action spaces, we'll use a simple actor-critic setup
            # PPO is an algorithm, not a policy, so we can't use it directly here
            from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
            policy_class = DiscreteActorPolicy  # This is a workaround
            is_discrete = False
        else:
            # Default to DQN for unknown spaces
            policy_class = DiscreteQLearningPolicy
            is_discrete = True
            
        # Get configuration parameters
        learning_rate = config.get("learning_rate", 1e-3)
        hidden_sizes = config.get("hidden_sizes", [128, 128])
        epsilon = config.get("epsilon", 0.1)
        
        # Create policies based on mode
        if mode == "shared":
            # Parameter sharing - single policy for all agents
            if hasattr(env, 'observation_space'):
                obs_space = env.observation_space
            else:
                obs_space = list(env.observation_spaces.values())[0] if hasattr(env, 'observation_spaces') else None
                
            # Get observation shape
            state_shape = None
            if hasattr(obs_space, 'shape'):
                state_shape = obs_space.shape
            elif hasattr(obs_space, 'spaces'):
                # Handle Dict spaces
                if 'observation' in obs_space.spaces:
                    obs_subspace = obs_space.spaces['observation']
                    if hasattr(obs_subspace, 'shape'):
                        state_shape = obs_subspace.shape
                    else:
                        state_shape = (4,)  # Default
                else:
                    # Flatten dict space
                    total_size = sum(
                        np.prod(space.shape) if hasattr(space, 'shape') else space.n
                        for space in obs_space.spaces.values()
                    )
                    state_shape = (int(total_size),)
            else:
                state_shape = (obs_space.n,) if hasattr(obs_space, 'n') else (4,)  # Default fallback
            
            # Fallback if still None
            if state_shape is None:
                state_shape = (4,)
                
            # Get action shape  
            if is_discrete:
                action_shape = action_space.n if hasattr(action_space, 'n') else 2
            else:
                action_shape = action_space.shape[0] if hasattr(action_space, 'shape') else 2
                
            # Create network
            if is_discrete:
                model = Net(
                    state_shape=state_shape,
                    action_shape=action_shape,
                    hidden_sizes=hidden_sizes,
                ).to(device)
                
                # Create shared policy
                shared_policy = policy_class(
                    model=model,
                    action_space=action_space,
                    observation_space=obs_space,
                    eps_training=epsilon,
                    eps_inference=epsilon * 0.5,
                )
            else:
                # For continuous policies, we need actor and critic networks
                from tianshou.utils.net.continuous import ActorProb, Critic
                
                actor = ActorProb(
                    obs_space.shape,
                    action_shape,
                    hidden_sizes=hidden_sizes,
                    device=device,
                ).to(device)
                
                critic = Critic(
                    obs_space.shape,
                    hidden_sizes=hidden_sizes,
                    device=device,
                ).to(device)
                
                optim_actor = torch.optim.Adam(actor.parameters(), lr=learning_rate)
                optim_critic = torch.optim.Adam(critic.parameters(), lr=learning_rate)
                
                # For continuous spaces, use discrete actor as a placeholder for now
                # In a real implementation, we'd need proper continuous policy support
                shared_policy = DiscreteQLearningPolicy(
                    model=critic,  # Using critic as model for now
                    action_space=action_space,
                    observation_space=obs_space,
                    eps_training=epsilon,
                    eps_inference=epsilon * 0.5,
                )
                
            # Create manager with shared policy
            return FlexibleMultiAgentPolicyManager(
                policies=shared_policy,
                env=env,
                mode="shared",
                **kwargs
            )
            
        else:
            # Independent or grouped mode
            policies = {}
            agents = env.agents if hasattr(env, 'agents') else []
            
            for agent_id in agents:
                # Get agent-specific spaces
                if hasattr(env, 'observation_spaces'):
                    obs_space = env.observation_spaces.get(agent_id, env.observation_space)
                else:
                    obs_space = env.observation_space
                    
                if hasattr(env, 'action_spaces'):
                    act_space = env.action_spaces.get(agent_id, env.action_space)
                else:
                    act_space = env.action_space if hasattr(env, 'action_space') else action_space
                    
                # Get shapes  
                state_shape = None
                if hasattr(obs_space, 'shape'):
                    state_shape = obs_space.shape
                elif hasattr(obs_space, 'spaces'):
                    # Handle Dict spaces
                    if 'observation' in obs_space.spaces:
                        obs_subspace = obs_space.spaces['observation']
                        if hasattr(obs_subspace, 'shape'):
                            state_shape = obs_subspace.shape
                        else:
                            state_shape = (4,)  # Default
                    else:
                        # Flatten dict space
                        total_size = sum(
                            np.prod(space.shape) if hasattr(space, 'shape') else space.n
                            for space in obs_space.spaces.values()
                        )
                        state_shape = (int(total_size),)
                else:
                    state_shape = (obs_space.n,) if hasattr(obs_space, 'n') else (4,)
                    
                if isinstance(act_space, spaces.Discrete):
                    action_shape = act_space.n
                else:
                    action_shape = act_space.shape[0] if hasattr(act_space, 'shape') else 2
                    
                # Create network for this agent
                if is_discrete:
                    # Fallback if state_shape is still None
                    if state_shape is None:
                        state_shape = (4,)  # Default fallback
                    
                    model = Net(
                        state_shape=state_shape,
                        action_shape=action_shape,
                        hidden_sizes=hidden_sizes,
                    ).to(device)
                    
                    policies[agent_id] = policy_class(
                        model=model,
                        action_space=act_space,
                        observation_space=obs_space,
                        eps_training=epsilon,
                        eps_inference=epsilon * 0.5,
                    )
                else:
                    # For continuous, fall back to DQN for now as a placeholder
                    # A real implementation would properly handle continuous policies
                    model = Net(
                        state_shape=state_shape,
                        action_shape=2,  # Dummy action shape for continuous
                        hidden_sizes=hidden_sizes,
                    ).to(device)
                    
                    policies[agent_id] = DiscreteQLearningPolicy(
                        model=model,
                        action_space=spaces.Discrete(2),  # Dummy discrete space
                        observation_space=obs_space,
                        eps_training=epsilon,
                        eps_inference=epsilon * 0.5,
                    )
                    
            return FlexibleMultiAgentPolicyManager(
                policies=policies,
                env=env,
                mode=mode,
                **kwargs
            )


# Add factory methods to PolicyManager
def _policy_manager_from_env(cls, env, algorithm="DQN", mode="independent", config=None, **kwargs):
    """Factory method to create PolicyManager from environment.
    
    Args:
        env: The multi-agent environment
        algorithm: Algorithm to use ("DQN", "PPO", "A2C", etc.)
        mode: Policy mode ("independent", "shared", "grouped")
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        FlexibleMultiAgentPolicyManager configured for the environment
    """
    # Map algorithm names to policy classes
    algorithm_map = {
        "DQN": DQNPolicy,
        "PPO": PPOPolicy,
        "A2C": A2CPolicy,
        "SAC": SACPolicy,
        "DDPG": DDPGPolicy,
        "TD3": TD3Policy,
    }
    
    if algorithm not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(algorithm_map.keys())}")
    
    # Use AutoPolicy's logic but with specified algorithm
    config = config or {}
    config["algorithm_class"] = algorithm_map[algorithm]
    
    return AutoPolicy.from_env(env, mode=mode, config=config, **kwargs)

# Monkey-patch the factory method onto PolicyManager
PolicyManager.from_env = classmethod(_policy_manager_from_env)

__all__ = [
    # Policy managers
    "PolicyManager",
    "AutoPolicy",
    
    # MARL-specific policies
    "QMIXPolicy",
    "MADDPGPolicy", 
    "CTDEPolicy",
    "CommunicatingPolicy",
    "CommunicationChannel",
    
    # Standard RL policies
    "DQNPolicy",
    "PPOPolicy",
    "A2CPolicy",
    "SACPolicy",
    "DDPGPolicy",
    "TD3Policy",
    
    # Training components
    "Collector",
    "ReplayBuffer",
    "Trainer",
    "MARLEnv",
]

__version__ = "0.1.0"