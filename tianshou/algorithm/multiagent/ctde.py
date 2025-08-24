"""Centralized Training with Decentralized Execution (CTDE) support.

This module provides CTDE architectures for multi-agent reinforcement learning,
enabling agents to use global information during training while maintaining
decentralized execution capabilities.
"""

import copy
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tianshou.data import Batch, to_torch
from tianshou.algorithm.algorithm_base import Policy
from gymnasium import spaces


class CTDEPolicy(Policy):
    """Base class for CTDE (Centralized Training, Decentralized Execution) policies.
    
    This policy enables agents to use global information during training while
    maintaining the ability to act based only on local observations during execution.
    
    Example usage:
    ::
        actor = DecentralizedActor(obs_dim=4, action_dim=2)
        critic = CentralizedCritic(global_obs_dim=12)
        
        policy = CTDEPolicy(
            actor=actor,
            critic=critic,
            optim_actor=optim.Adam(actor.parameters()),
            optim_critic=optim.Adam(critic.parameters())
        )
        
        # Execution: uses only local observations
        action = policy.forward(local_obs_batch)
        
        # Training: uses global information
        loss = policy.learn(batch_with_global_state)
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optim_actor: optim.Optimizer,
        optim_critic: optim.Optimizer,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        enable_global_info: bool = True,
        discount_factor: float = 0.99,
        **kwargs
    ):
        """Initialize CTDE policy.
        
        Args:
            actor: Decentralized actor network (uses local observations)
            critic: Centralized critic network (uses global state)
            optim_actor: Optimizer for actor network
            optim_critic: Optimizer for critic network
            observation_space: Local observation space
            action_space: Action space
            enable_global_info: Whether to use global information during training
            discount_factor: Discount factor for future rewards
            **kwargs: Additional policy parameters
        """
        # Extract tau before passing kwargs to parent
        self.tau = kwargs.pop('tau', 0.005)  # Soft update parameter
        
        super().__init__(observation_space=observation_space, action_space=action_space, **kwargs)
        
        self.actor = actor
        self.critic = critic
        self.optim_actor = optim_actor
        self.optim_critic = optim_critic
        self.enable_global_info = enable_global_info
        self.discount_factor = discount_factor
        self.device = "cpu"  # Default device
        
    def forward(self, batch: Batch, state: Optional[Any] = None, **kwargs) -> Batch:
        """Forward pass for action selection (decentralized execution).
        
        During execution, only local observations are used.
        
        Args:
            batch: Batch containing local observations
            state: Optional RNN state
            **kwargs: Additional arguments
            
        Returns:
            Batch containing actions and optional state
        """
        # Ensure observations are tensors
        obs = to_torch(batch.obs, device=self.device)
        
        # Decentralized execution - use only local observations
        # Handle different actor signatures
        import inspect
        sig = inspect.signature(self.actor.forward if hasattr(self.actor, 'forward') else self.actor.__call__)
        
        # Check if actor expects state parameter
        if len(sig.parameters) > 1 and 'state' in sig.parameters:
            result = self.actor(obs, state)
        else:
            result = self.actor(obs)
        
        # Handle different return types
        if isinstance(result, tuple):
            actions, state = result
        else:
            actions = result
            # state remains as passed in
            
        return Batch(act=actions, state=state)
    
    def learn(self, batch: Batch, **kwargs) -> Dict[str, Any]:
        """Learning with centralized critic (centralized training).
        
        During training, global information is used if available.
        
        Args:
            batch: Training batch potentially containing global state
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        obs = to_torch(batch.obs, device=self.device)
        act = to_torch(batch.act, device=self.device)
        rew = to_torch(batch.rew, device=self.device)
        obs_next = to_torch(batch.obs_next, device=self.device)
        terminated = to_torch(batch.terminated, device=self.device)
        
        # Use global state if available and enabled
        if self.enable_global_info and hasattr(batch, 'global_obs'):
            critic_input = to_torch(batch.global_obs, device=self.device)
            critic_input_next = to_torch(batch.global_obs_next, device=self.device)
        else:
            # Fall back to local observations
            critic_input = obs
            critic_input_next = obs_next
        
        # Compute values with centralized critic
        values = self.critic(critic_input)
        values_next = self.critic(critic_input_next)
        
        # Ensure proper dimensions
        if values.dim() > 1 and values.shape[1] > 1:
            # If critic outputs multiple values, take mean or first
            values = values.mean(dim=1, keepdim=True)
            values_next = values_next.mean(dim=1, keepdim=True)
        elif values.dim() == 1:
            values = values.unsqueeze(-1)
            values_next = values_next.unsqueeze(-1)
            
        # Ensure rewards and terminated have proper shape
        if rew.dim() == 1:
            rew = rew.unsqueeze(-1)
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)
        
        # Compute TD target
        td_target = rew + self.discount_factor * values_next * (~terminated).float()
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(values, td_target.detach())
        
        # Actor loss (policy gradient)
        if hasattr(self.actor, 'forward'):
            actions_pred, _ = self.actor(obs, None)
        else:
            actions_pred = self.actor(obs)
            
        # Advantage for policy gradient
        advantage = (td_target - values).detach()
        
        # Simple policy gradient loss (negative for gradient ascent)
        log_probs = -F.cross_entropy(actions_pred, act, reduction='none')
        actor_loss = -(log_probs * advantage).mean()
        
        # Update networks
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }
    
    def soft_update_targets(self):
        """Soft update target networks using tau parameter."""
        if hasattr(self, 'actor_target'):
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        if hasattr(self, 'critic_target'):
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _build_global_state(self, batch: Batch) -> torch.Tensor:
        """Construct global state from all agents' observations.
        
        This is a placeholder that should be overridden by specific implementations.
        
        Args:
            batch: Batch containing all agents' observations
            
        Returns:
            Global state tensor
        """
        # Default: concatenate all observations
        if hasattr(batch, 'global_obs'):
            return batch.global_obs
        else:
            # This would need proper implementation based on environment
            return batch.obs


class GlobalStateConstructor(nn.Module):
    """Constructs global state from individual agent observations.
    
    Supports multiple construction modes:
    - concatenate: Simple concatenation of all observations
    - mean: Mean aggregation of all observations
    - attention: Attention-based aggregation
    - graph: Graph neural network aggregation
    - custom: User-defined function
    """
    
    def __init__(
        self,
        mode: Literal["concatenate", "mean", "attention", "graph", "custom"] = "concatenate",
        obs_dim: Optional[int] = None,
        n_agents: Optional[int] = None,
        hidden_dim: int = 64,
        adjacency_matrix: Optional[torch.Tensor] = None,
        custom_fn: Optional[Callable] = None,
    ):
        """Initialize global state constructor.
        
        Args:
            mode: Construction mode
            obs_dim: Dimension of individual observations
            n_agents: Number of agents
            hidden_dim: Hidden dimension for neural aggregation
            adjacency_matrix: Adjacency matrix for graph mode
            custom_fn: Custom construction function
        """
        super().__init__()
        
        self.mode = mode
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.adjacency_matrix = adjacency_matrix
        self.custom_fn = custom_fn
        
        if mode == "attention" and obs_dim and n_agents:
            # Attention-based aggregation
            self.attention = nn.MultiheadAttention(
                embed_dim=obs_dim,
                num_heads=4,
                batch_first=True
            )
            self.output_proj = nn.Linear(obs_dim, hidden_dim)
            
        elif mode == "graph" and obs_dim:
            # Graph neural network aggregation
            self.gnn_layer1 = nn.Linear(obs_dim, hidden_dim)
            self.gnn_layer2 = nn.Linear(hidden_dim, hidden_dim)
            
    def build(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build global state from observations.
        
        Args:
            observations: Dictionary mapping agent_id to observation tensor
            
        Returns:
            Global state tensor
        """
        if self.mode == "concatenate":
            # Simple concatenation
            obs_list = [obs for obs in observations.values()]
            return torch.cat(obs_list, dim=-1)
            
        elif self.mode == "mean":
            # Mean aggregation
            obs_list = [obs for obs in observations.values()]
            obs_stack = torch.stack(obs_list, dim=0)  # [n_agents, batch, obs_dim]
            return obs_stack.mean(dim=0)  # [batch, obs_dim]
            
        elif self.mode == "attention":
            # Stack observations for attention
            obs_list = list(observations.values())
            obs_stack = torch.stack(obs_list, dim=1)  # [batch, n_agents, obs_dim]
            
            # Self-attention
            attended, _ = self.attention(obs_stack, obs_stack, obs_stack)
            
            # Average pooling and projection
            pooled = attended.mean(dim=1)  # [batch, obs_dim]
            return self.output_proj(pooled)
            
        elif self.mode == "graph":
            # Graph-based aggregation
            obs_list = list(observations.values())
            obs_stack = torch.stack(obs_list, dim=1)  # [batch, n_agents, obs_dim]
            
            # GNN layer 1
            h = F.relu(self.gnn_layer1(obs_stack))
            
            # Aggregate neighbors (using adjacency if provided)
            if self.adjacency_matrix is not None:
                # Weighted aggregation based on adjacency
                adj_expanded = self.adjacency_matrix.unsqueeze(0).expand(h.shape[0], -1, -1)
                h_aggregated = torch.bmm(adj_expanded, h) / adj_expanded.sum(dim=-1, keepdim=True)
            else:
                # Mean aggregation
                h_aggregated = h.mean(dim=1, keepdim=True).expand_as(h)
            
            # GNN layer 2
            h_final = self.gnn_layer2(h_aggregated)
            
            # Global pooling
            return h_final.mean(dim=1)
            
        elif self.mode == "custom" and self.custom_fn:
            return self.custom_fn(observations)
            
        else:
            # Default to concatenation
            obs_list = [obs for obs in observations.values()]
            return torch.cat(obs_list, dim=-1)


class DecentralizedActor(nn.Module):
    """Decentralized actor network for CTDE.
    
    Uses only local observations to select actions.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize decentralized actor.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor, state=None) -> Tuple[torch.Tensor, Any]:
        """Forward pass.
        
        Args:
            obs: Local observations
            state: Optional RNN state
            
        Returns:
            Tuple of (action logits, state)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions, state


class CentralizedCritic(nn.Module):
    """Centralized critic network for CTDE.
    
    Uses global state/information during training.
    """
    
    def __init__(self, global_obs_dim: int, n_agents: int, hidden_dim: int = 128):
        """Initialize centralized critic.
        
        Args:
            global_obs_dim: Global observation/state dimension
            n_agents: Number of agents (for multi-agent Q-values)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.fc1 = nn.Linear(global_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_agents)  # Q-value per agent
        
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            global_obs: Global state/observations
            
        Returns:
            Q-values for all agents
        """
        x = F.relu(self.fc1(global_obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class QMIXMixer(nn.Module):
    """QMIX mixing network.
    
    Combines individual Q-values into a global Q-value while maintaining
    monotonicity constraints.
    """
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
        enforce_monotonic: bool = True
    ):
        """Initialize QMIX mixer.
        
        Args:
            n_agents: Number of agents
            state_dim: Global state dimension
            mixing_embed_dim: Mixing network embedding dimension
            hypernet_embed_dim: Hypernetwork embedding dimension
            enforce_monotonic: Whether to enforce monotonicity constraint
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.enforce_monotonic = enforce_monotonic
        
        # Hypernetwork for weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, n_agents * mixing_embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim)
        )
        
        # Hypernetwork for biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Mix individual Q-values.
        
        Args:
            q_values: Individual Q-values [batch, n_agents]
            state: Global state [batch, state_dim]
            
        Returns:
            Mixed Q-value [batch, 1]
        """
        batch_size = q_values.shape[0]
        
        # Generate weights and biases from global state
        w1 = self.hyper_w1(state).view(batch_size, self.n_agents, -1)
        b1 = self.hyper_b1(state).view(batch_size, 1, -1)
        
        w2 = self.hyper_w2(state).view(batch_size, -1, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        # Enforce positive weights for monotonicity
        if self.enforce_monotonic:
            w1 = torch.abs(w1)
            w2 = torch.abs(w2)
        
        # First layer
        q_values = q_values.view(batch_size, 1, self.n_agents)
        h = F.elu(torch.bmm(q_values, w1) + b1)
        
        # Second layer
        q_total = torch.bmm(h, w2) + b2
        
        return q_total.view(batch_size, 1)


class QMIXPolicy(Policy):
    """QMIX policy for multi-agent Q-learning.
    
    Uses individual Q-networks for decentralized execution and a mixing
    network for centralized training.
    """
    
    def __init__(
        self,
        actors: List[nn.Module],
        mixer: QMIXMixer,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_agents: int,
        optimizer: Optional[optim.Optimizer] = None,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        **kwargs
    ):
        """Initialize QMIX policy.
        
        Args:
            actors: List of Q-networks (one per agent)
            mixer: QMIX mixing network
            observation_space: Observation space
            action_space: Action space (must be Discrete)
            n_agents: Number of agents
            optimizer: Optimizer for all networks
            discount_factor: Discount factor
            epsilon: Epsilon for epsilon-greedy exploration
            **kwargs: Additional parameters
        """
        super().__init__(observation_space=observation_space, action_space=action_space, **kwargs)
        
        self.actors = actors
        self.mixer = mixer
        self.n_agents = n_agents
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.device = "cpu"  # Default device
        
        # Create optimizer if not provided
        if optimizer is None:
            params = []
            for actor in actors:
                params.extend(actor.parameters())
            params.extend(mixer.parameters())
            self.optimizer = optim.Adam(params)
        else:
            self.optimizer = optimizer
            
        # Target networks for double Q-learning
        self.target_actors = [copy.deepcopy(actor) for actor in actors]
        self.target_mixer = copy.deepcopy(mixer)
        
    def forward(self, batch: Batch, state=None, **kwargs) -> Batch:
        """Forward pass for decentralized execution.
        
        Args:
            batch: Multi-agent batch with observations or single agent batch
            state: Optional RNN state
            **kwargs: Additional arguments
            
        Returns:
            Batch with actions for each agent
        """
        # Check if this is a multi-agent batch or single observation
        if hasattr(batch, 'obs') and not any(key.startswith("agent_") for key in batch.keys()):
            # Single observation batch - use first actor for simplicity
            obs = to_torch(batch.obs, device=self.device)
            
            # Get Q-values from first actor
            actor_out = self.actors[0](obs)
            # Handle tuple return (q_values, state) or just q_values
            if isinstance(actor_out, tuple):
                q_values = actor_out[0]
            else:
                q_values = actor_out
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Random action
                actions = torch.randint(0, self.action_space.n, (obs.shape[0],))
            else:
                # Greedy action
                actions = q_values.argmax(dim=-1)
            
            return Batch(act=actions, state=state)
        
        # Multi-agent batch
        result = Batch()
        
        for i, agent_id in enumerate(batch.keys()):
            if agent_id.startswith("agent_"):
                obs = to_torch(batch[agent_id].obs, device=self.device)
                
                # Get Q-values from actor
                actor_out = self.actors[i](obs)
                # Handle tuple return (q_values, state) or just q_values
                if isinstance(actor_out, tuple):
                    q_values = actor_out[0]
                else:
                    q_values = actor_out
                
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    # Random action
                    actions = torch.randint(0, self.action_space.n, (obs.shape[0],))
                else:
                    # Greedy action
                    actions = q_values.argmax(dim=-1)
                
                result[agent_id] = Batch(act=actions)
                
        return result
    
    def learn(self, batch: Batch, **kwargs) -> Dict[str, Any]:
        """QMIX learning step.
        
        Args:
            batch: Multi-agent batch with global state
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of training metrics
        """
        # Collect individual Q-values
        q_values_list = []
        q_values_next_list = []
        rewards_list = []
        
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            if agent_id in batch:
                agent_batch = batch[agent_id]
                obs = to_torch(agent_batch.obs, device=self.device)
                obs_next = to_torch(agent_batch.obs_next, device=self.device)
                act = to_torch(agent_batch.act, device=self.device).long()
                rew = to_torch(agent_batch.rew, device=self.device)
                
                # Current Q-values
                actor_out = self.actors[i](obs)
                if isinstance(actor_out, tuple):
                    q_values = actor_out[0]
                else:
                    q_values = actor_out
                
                # Ensure q_values has the right shape for gathering
                if q_values.dim() == 1:
                    # If q_values is 1D, it's likely a single Q-value per sample
                    # For discrete actions, we need Q-values for each action
                    q_values = q_values.unsqueeze(-1)
                
                
                # For discrete actions, gather the Q-value for the taken action
                q_values_selected = q_values.gather(1, act.unsqueeze(-1))
                q_values_list.append(q_values_selected)
                
                # Target Q-values
                with torch.no_grad():
                    target_out = self.target_actors[i](obs_next)
                    if isinstance(target_out, tuple):
                        q_values_next = target_out[0]
                    else:
                        q_values_next = target_out
                    q_values_next_max = q_values_next.max(dim=-1, keepdim=True)[0]
                    q_values_next_list.append(q_values_next_max)
                
                rewards_list.append(rew)
        
        # Stack Q-values
        q_values_all = torch.cat(q_values_list, dim=-1)
        q_values_next_all = torch.cat(q_values_next_list, dim=-1)
        rewards = torch.stack(rewards_list, dim=-1).mean(dim=-1, keepdim=True)
        
        # Get global state
        global_state = to_torch(batch.global_obs, device=self.device)
        global_state_next = to_torch(batch.global_obs_next, device=self.device)
        
        # Mix Q-values
        q_total = self.mixer(q_values_all, global_state)
        
        with torch.no_grad():
            q_total_next = self.target_mixer(q_values_next_all, global_state_next)
            
        # TD target
        terminated = to_torch(batch[f"agent_0"].terminated, device=self.device)
        td_target = rewards + self.discount_factor * q_total_next * (~terminated).float().unsqueeze(-1)
        
        # Loss
        loss = F.mse_loss(q_total, td_target)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "q_values": q_total.mean().item(),
        }
    
    def update_target_networks(self, tau: float = 0.005):
        """Soft update target networks.
        
        Args:
            tau: Soft update parameter
        """
        for i in range(self.n_agents):
            for param, target_param in zip(
                self.actors[i].parameters(),
                self.target_actors[i].parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
                
        for param, target_param in zip(
            self.mixer.parameters(),
            self.target_mixer.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )


class MADDPGPolicy(Policy):
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
    
    Each agent has its own actor and critic, but critics use information
    from all agents during training.
    """
    
    def __init__(
        self,
        actors: List[nn.Module],
        critics: List[nn.Module],
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_agents: int,
        optimizer_actors: Optional[List[optim.Optimizer]] = None,
        optimizer_critics: Optional[List[optim.Optimizer]] = None,
        discount_factor: float = 0.99,
        tau: float = 0.01,
        **kwargs
    ):
        """Initialize MADDPG policy.
        
        Args:
            actors: List of actor networks (one per agent)
            critics: List of critic networks (one per agent)
            observation_space: Observation space
            action_space: Action space (continuous)
            n_agents: Number of agents
            optimizer_actors: Optimizers for actors
            optimizer_critics: Optimizers for critics
            discount_factor: Discount factor
            tau: Soft update parameter for target networks
            **kwargs: Additional parameters
        """
        super().__init__(observation_space=observation_space, action_space=action_space, **kwargs)
        
        self.actors = actors
        self.critics = critics
        self.n_agents = n_agents
        self.discount_factor = discount_factor
        self.tau = tau
        self.device = "cpu"  # Default device
        
        # Create optimizers if not provided
        if optimizer_actors is None:
            self.optimizer_actors = [
                optim.Adam(actor.parameters()) for actor in actors
            ]
        else:
            self.optimizer_actors = optimizer_actors
            
        if optimizer_critics is None:
            self.optimizer_critics = [
                optim.Adam(critic.parameters()) for critic in critics
            ]
        else:
            self.optimizer_critics = optimizer_critics
        
        # Target networks
        self.target_actors = [copy.deepcopy(actor) for actor in actors]
        self.target_critics = [copy.deepcopy(critic) for critic in critics]
        
    def forward(self, batch: Batch, state=None, **kwargs) -> Batch:
        """Forward pass for decentralized execution.
        
        Args:
            batch: Multi-agent batch
            state: Optional RNN state
            **kwargs: Additional arguments
            
        Returns:
            Batch with actions for each agent
        """
        result = Batch()
        
        for i, agent_id in enumerate(batch.keys()):
            if agent_id.startswith("agent_"):
                obs = to_torch(batch[agent_id].obs, device=self.device)
                
                # Get action from actor
                if hasattr(self.actors[i], 'forward'):
                    actions, _ = self.actors[i](obs, state)
                else:
                    actions = self.actors[i](obs)
                
                result[agent_id] = Batch(act=actions)
                
        return result
    
    def learn(self, batch: Batch, **kwargs) -> Dict[str, Any]:
        """MADDPG learning step.
        
        Args:
            batch: Multi-agent batch
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of training metrics
        """
        losses = {}
        
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            
            # Get all agents' observations and actions
            all_obs = []
            all_actions = []
            all_obs_next = []
            all_actions_next = []
            
            for j in range(self.n_agents):
                other_id = f"agent_{j}"
                if other_id in batch:
                    other_batch = batch[other_id]
                    all_obs.append(to_torch(other_batch.obs, device=self.device))
                    
                    # Get actions and ensure proper dimensions
                    act_tensor = to_torch(other_batch.act, device=self.device)
                    # For continuous actions, ensure 2D shape [batch, action_dim]
                    if isinstance(self.action_space, spaces.Box):
                        if act_tensor.dim() == 1:
                            # Check if it's a batch of scalar actions or needs reshaping
                            batch_size = all_obs[-1].shape[0]
                            action_dim = self.action_space.shape[0]
                            if act_tensor.shape[0] == batch_size * action_dim:
                                # Reshape flattened actions back to [batch, action_dim]
                                act_tensor = act_tensor.view(batch_size, action_dim)
                            elif act_tensor.shape[0] == batch_size:
                                # Single action dimension, add dimension
                                act_tensor = act_tensor.unsqueeze(-1)
                        # Already 2D, keep as is
                    else:
                        # Discrete actions
                        if act_tensor.dim() == 1:
                            act_tensor = act_tensor.unsqueeze(-1)
                    all_actions.append(act_tensor)
                    all_obs_next.append(to_torch(other_batch.obs_next, device=self.device))
                    
                    # Get next actions from target actor
                    with torch.no_grad():
                        if hasattr(self.target_actors[j], 'forward'):
                            next_actions, _ = self.target_actors[j](all_obs_next[-1], None)
                        else:
                            next_actions = self.target_actors[j](all_obs_next[-1])
                        all_actions_next.append(next_actions)
            
            # Concatenate all observations and actions for centralized critic
            all_obs_concat = torch.cat(all_obs, dim=-1)
            # Actions are already properly formatted when added to all_actions
            all_actions_concat = torch.cat(all_actions, dim=-1)
            all_obs_next_concat = torch.cat(all_obs_next, dim=-1)
            # Next actions should already have proper shape from actors
            all_actions_next_concat = torch.cat(all_actions_next, dim=-1)
            
            # Critic update
            agent_batch = batch[agent_id]
            rewards = to_torch(agent_batch.rew, device=self.device)
            terminated = to_torch(agent_batch.terminated, device=self.device)
            
            # Current Q-value
            q_value = self.critics[i](torch.cat([all_obs_concat, all_actions_concat], dim=-1))
            
            # Target Q-value
            with torch.no_grad():
                q_value_next = self.target_critics[i](
                    torch.cat([all_obs_next_concat, all_actions_next_concat], dim=-1)
                )
                td_target = rewards + self.discount_factor * q_value_next.squeeze() * (~terminated).float()
            
            # Critic loss
            critic_loss = F.mse_loss(q_value.squeeze(), td_target)
            
            # Update critic
            self.optimizer_critics[i].zero_grad()
            critic_loss.backward()
            self.optimizer_critics[i].step()
            
            # Actor update
            obs_i = to_torch(agent_batch.obs, device=self.device)
            if hasattr(self.actors[i], 'forward'):
                actions_i, _ = self.actors[i](obs_i, None)
            else:
                actions_i = self.actors[i](obs_i)
            
            # Replace agent's action in the concatenated actions
            all_actions_for_actor = all_actions.copy()
            all_actions_for_actor[i] = actions_i
            all_actions_for_actor_concat = torch.cat(all_actions_for_actor, dim=-1)
            
            # Actor loss (negative Q-value for gradient ascent)
            actor_loss = -self.critics[i](
                torch.cat([all_obs_concat, all_actions_for_actor_concat], dim=-1)
            ).mean()
            
            # Update actor
            self.optimizer_actors[i].zero_grad()
            actor_loss.backward()
            self.optimizer_actors[i].step()
            
            losses[f"{agent_id}_actor_loss"] = actor_loss.item()
            losses[f"{agent_id}_critic_loss"] = critic_loss.item()
        
        # Aggregate losses
        losses["actor_loss"] = np.mean([v for k, v in losses.items() if "actor_loss" in k])
        losses["critic_loss"] = np.mean([v for k, v in losses.items() if "critic_loss" in k])
        
        return losses
    
    def update_target_networks(self):
        """Soft update target networks."""
        for i in range(self.n_agents):
            # Update target actor
            for param, target_param in zip(
                self.actors[i].parameters(),
                self.target_actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            # Update target critic
            for param, target_param in zip(
                self.critics[i].parameters(),
                self.target_critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )