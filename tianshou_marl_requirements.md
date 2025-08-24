# Tianshou MARL Enhancement Requirements

## Document Purpose
This document specifies required Multi-Agent Reinforcement Learning (MARL) enhancements for the Tianshou library. It is written for Tianshou maintainers or contributors to understand and implement necessary features for production MARL applications.

## Executive Summary
Tianshou currently provides experimental MARL support through `MultiAgentPolicyManager` and `PettingZooEnv`. However, production MARL applications require additional capabilities for efficient training, flexible agent configurations, and robust parallel environment handling.

## Core Requirements

### 1. Enhanced Parallel Environment Support

#### Current State
- Tianshou's `PettingZooEnv` primarily supports AEC (Agent Environment Cycle) environments
- Limited support for ParallelEnv where all agents act simultaneously

#### Required Enhancement
```python
class EnhancedPettingZooEnv(PettingZooEnv):
    """Enhanced wrapper supporting both AEC and Parallel PettingZoo environments."""
    
    def __init__(self, env, mode: Literal["aec", "parallel"] = "auto"):
        """
        Args:
            env: PettingZoo environment (AEC or Parallel)
            mode: Environment mode - "aec", "parallel", or "auto" (detect from env)
        """
        # Auto-detect environment type
        if mode == "auto":
            self.mode = "parallel" if hasattr(env, "possible_agents") else "aec"
        else:
            self.mode = mode
        
        super().__init__(env)
        
    def step(self, action: Union[int, np.ndarray, Dict[str, Any]]):
        """Handle both single-agent and multi-agent actions."""
        if self.mode == "parallel":
            # All agents act simultaneously
            return self._parallel_step(action)
        else:
            # Sequential agent actions
            return self._aec_step(action)
```

**Rationale**: Many real-world scenarios require simultaneous decision-making (e.g., auctions, markets, traffic systems).

### 2. Flexible Policy Configuration

#### Current State
- `MultiAgentPolicyManager` requires one policy per agent
- No built-in support for parameter sharing or heterogeneous agent groups

#### Required Enhancement
```python
class FlexibleMultiAgentPolicyManager(MultiAgentPolicyManager):
    """Enhanced policy manager with flexible agent-policy mapping."""
    
    def __init__(
        self,
        policies: Union[BasePolicy, List[BasePolicy], Dict[str, BasePolicy]],
        env: PettingZooEnv,
        mode: Literal["independent", "shared", "grouped", "custom"] = "independent",
        agent_groups: Optional[Dict[str, List[str]]] = None,
        policy_mapping_fn: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            policies: Single policy (shared), list (one per agent), or dict (named policies)
            env: PettingZoo environment
            mode: Policy assignment mode
                - "independent": Each agent has its own policy
                - "shared": All agents share the same policy
                - "grouped": Agents in same group share policy
                - "custom": Use policy_mapping_fn
            agent_groups: Dict mapping group names to agent IDs (for "grouped" mode)
            policy_mapping_fn: Custom function mapping agent_id to policy_id
        """
        self.mode = mode
        self.agent_groups = agent_groups or {}
        self.policy_mapping_fn = policy_mapping_fn
        
        # Build policy mapping based on mode
        self.policy_map = self._build_policy_map(policies, env.agents)
        
    def _build_policy_map(self, policies, agents):
        """Build agent-to-policy mapping based on configuration mode."""
        if self.mode == "shared":
            # All agents use the same policy
            return {agent: policies if isinstance(policies, BasePolicy) else policies[0] 
                   for agent in agents}
        
        elif self.mode == "grouped":
            # Agents in same group share policy
            policy_map = {}
            for group_name, group_agents in self.agent_groups.items():
                group_policy = policies[group_name] if isinstance(policies, dict) else policies[0]
                for agent in group_agents:
                    policy_map[agent] = group_policy
            return policy_map
        
        elif self.mode == "custom":
            # Use custom mapping function
            return {agent: policies[self.policy_mapping_fn(agent)] for agent in agents}
        
        else:  # "independent"
            # Each agent has its own policy
            if isinstance(policies, list):
                return dict(zip(agents, policies))
            elif isinstance(policies, dict):
                return policies
            else:
                raise ValueError("Independent mode requires list or dict of policies")
```

**Rationale**: Different MARL scenarios require different policy configurations:
- **Homogeneous agents** (e.g., identical robots) benefit from parameter sharing
- **Heterogeneous agents** (e.g., different player types) need separate policies
- **Grouped agents** (e.g., teams) share policies within groups

### 3. Dict Observation Space Support

#### Current State
- Limited support for Dict/nested observation spaces
- Manual flattening required for complex observations

#### Required Enhancement
```python
class DictObservationWrapper:
    """Automatic handling of Dict observation spaces."""
    
    def __init__(self, policy: BasePolicy, observation_space: gym.spaces.Dict):
        """
        Args:
            policy: Base policy to wrap
            observation_space: Dict observation space
        """
        self.policy = policy
        self.observation_space = observation_space
        self.preprocessor = self._build_preprocessor()
        
    def _build_preprocessor(self):
        """Build preprocessor for Dict observations."""
        if self.policy.preprocessing:
            # User-defined preprocessing
            return self.policy.preprocessing
        else:
            # Auto-build preprocessor
            return DictToTensorPreprocessor(self.observation_space)
    
    def forward(self, batch: Batch, state: Optional[Any] = None):
        """Process Dict observations before policy forward."""
        # Preprocess observations
        if isinstance(batch.obs, dict):
            batch.obs = self.preprocessor(batch.obs)
        return self.policy.forward(batch, state)

class DictToTensorPreprocessor(nn.Module):
    """Automatic Dict-to-Tensor converter for neural network policies."""
    
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__()
        self.keys = sorted(observation_space.spaces.keys())
        self.spaces = observation_space.spaces
        
        # Build feature extractors for each component
        self.extractors = nn.ModuleDict()
        for key, space in self.spaces.items():
            self.extractors[key] = self._build_extractor(key, space)
        
    def _build_extractor(self, key: str, space: gym.Space) -> nn.Module:
        """Build feature extractor for space component."""
        if isinstance(space, gym.spaces.Box):
            return nn.Linear(np.prod(space.shape), 64)
        elif isinstance(space, gym.spaces.Discrete):
            return nn.Embedding(space.n, 32)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return nn.ModuleList([nn.Embedding(n, 16) for n in space.nvec])
        else:
            raise NotImplementedError(f"Space type {type(space)} not supported")
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert Dict observation to flat tensor."""
        features = []
        for key in self.keys:
            feature = self.extractors[key](obs[key])
            features.append(feature)
        return torch.cat(features, dim=-1)
```

**Rationale**: Complex environments often have structured observations (e.g., different sensor modalities, game state components) best represented as dictionaries.

### 4. Training Coordination Modes

#### Current State
- Basic simultaneous training of all agents
- No built-in support for different training schemes

#### Required Enhancement
```python
class MATrainer:
    """Multi-agent trainer with flexible training coordination."""
    
    def __init__(
        self,
        policy_manager: FlexibleMultiAgentPolicyManager,
        training_mode: Literal["simultaneous", "sequential", "self_play", "league"] = "simultaneous",
        **kwargs
    ):
        """
        Args:
            policy_manager: Multi-agent policy manager
            training_mode: How agents are trained
                - "simultaneous": All agents train together
                - "sequential": Agents train one at a time
                - "self_play": Agent trains against past versions
                - "league": Population-based training
        """
        self.policy_manager = policy_manager
        self.training_mode = training_mode
        
    def train_step(self, batch: Batch) -> Dict[str, Any]:
        """Execute one training step based on mode."""
        if self.training_mode == "simultaneous":
            # Standard: all agents learn from their experiences
            return self._simultaneous_train(batch)
        
        elif self.training_mode == "sequential":
            # Round-robin: one agent learns while others are fixed
            return self._sequential_train(batch)
        
        elif self.training_mode == "self_play":
            # Agent learns against historical versions
            return self._self_play_train(batch)
        
        elif self.training_mode == "league":
            # Population-based with matchmaking
            return self._league_train(batch)
    
    def _self_play_train(self, batch: Batch) -> Dict[str, Any]:
        """Self-play training implementation."""
        # Get learning agent and opponent policies
        learning_agent = self.get_current_learner()
        opponent_pool = self.get_opponent_pool()
        
        # Sample opponent from pool
        opponent = np.random.choice(opponent_pool, p=self.opponent_weights)
        
        # Train learning agent against opponent
        losses = {}
        for agent_id, agent_batch in batch.split_by_agent():
            if agent_id == learning_agent:
                # Update learning agent
                losses[agent_id] = self.policy_manager.policies[agent_id].learn(agent_batch)
            # Opponents don't learn
        
        # Periodically add current policy to opponent pool
        if self.step % self.snapshot_interval == 0:
            self.add_to_opponent_pool(learning_agent)
        
        return losses
```

**Rationale**: Different training schemes are optimal for different scenarios:
- **Competitive games**: Self-play prevents overfitting to specific opponents
- **Cooperative tasks**: Simultaneous training ensures coordination
- **Asymmetric games**: Sequential training can stabilize learning

### 5. Centralized Training with Decentralized Execution (CTDE)

#### Current State
- No explicit support for CTDE architectures
- Manual implementation required for centralized critics

#### Required Enhancement
```python
class CTDEPolicy(BasePolicy):
    """Base class for CTDE (Centralized Training, Decentralized Execution) policies."""
    
    def __init__(
        self,
        actor: nn.Module,  # Decentralized actor
        critic: nn.Module,  # Centralized critic
        optim: torch.optim.Optimizer,
        enable_global_info: bool = True,
        **kwargs
    ):
        """
        Args:
            actor: Decentralized actor network (uses local observations)
            critic: Centralized critic network (uses global state)
            optim: Optimizer for both networks
            enable_global_info: Whether to use global information during training
        """
        super().__init__(**kwargs)
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self.enable_global_info = enable_global_info
        
    def forward(self, batch: Batch, state: Optional[Any] = None, **kwargs):
        """Forward pass for action selection (decentralized execution)."""
        # During execution, only use local observations
        local_obs = batch.obs
        actions, state = self.actor(local_obs, state)
        return Batch(act=actions, state=state)
    
    def learn(self, batch: Batch, **kwargs):
        """Learning with centralized critic (centralized training)."""
        if self.enable_global_info:
            # Concatenate all agents' observations for centralized critic
            global_state = self._build_global_state(batch)
            values = self.critic(global_state)
        else:
            # Fall back to local observations
            values = self.critic(batch.obs)
        
        # Compute actor and critic losses
        actor_loss = self._compute_actor_loss(batch, values)
        critic_loss = self._compute_critic_loss(batch, values)
        
        # Update networks
        self.optim.zero_grad()
        (actor_loss + critic_loss).backward()
        self.optim.step()
        
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    
    def _build_global_state(self, batch: Batch) -> torch.Tensor:
        """Construct global state from all agents' observations."""
        # Implementation depends on environment
        # Could concatenate all observations, or use environment's global state
        pass
```

**Rationale**: CTDE is crucial for multi-agent coordination while maintaining deployment feasibility (agents can't access global information during execution).

### 6. Communication Between Agents

#### Current State
- No built-in support for agent communication
- Must be implemented at environment level

#### Required Enhancement
```python
class CommunicationChannel:
    """Inter-agent communication module."""
    
    def __init__(
        self,
        comm_type: Literal["broadcast", "targeted", "graph"] = "broadcast",
        message_dim: int = 32,
        max_messages: int = 10,
        topology: Optional[np.ndarray] = None  # For graph-based communication
    ):
        """
        Args:
            comm_type: Communication pattern
                - "broadcast": All agents receive all messages
                - "targeted": Agents send messages to specific recipients
                - "graph": Communication follows graph topology
            message_dim: Dimension of message vectors
            max_messages: Maximum messages per step
            topology: Adjacency matrix for graph communication
        """
        self.comm_type = comm_type
        self.message_dim = message_dim
        self.max_messages = max_messages
        self.topology = topology
        self.message_buffer = []
        
    def send(self, sender_id: str, message: torch.Tensor, target_id: Optional[str] = None):
        """Send message from agent."""
        msg_packet = {
            "sender": sender_id,
            "message": message,
            "target": target_id,
            "timestamp": time.time()
        }
        self.message_buffer.append(msg_packet)
        
    def receive(self, receiver_id: str) -> List[torch.Tensor]:
        """Receive messages for agent."""
        messages = []
        
        for packet in self.message_buffer:
            if self._should_receive(receiver_id, packet):
                messages.append(packet["message"])
        
        # Limit number of messages
        if len(messages) > self.max_messages:
            messages = messages[:self.max_messages]
        
        return messages
    
    def _should_receive(self, receiver_id: str, packet: Dict) -> bool:
        """Determine if receiver should get message."""
        if self.comm_type == "broadcast":
            return packet["sender"] != receiver_id
        elif self.comm_type == "targeted":
            return packet["target"] == receiver_id
        elif self.comm_type == "graph":
            sender_idx = self.agent_to_idx[packet["sender"]]
            receiver_idx = self.agent_to_idx[receiver_id]
            return self.topology[sender_idx, receiver_idx] > 0
        return False
    
    def clear(self):
        """Clear message buffer for next step."""
        self.message_buffer = []

class CommunicatingPolicy(BasePolicy):
    """Policy with communication capabilities."""
    
    def __init__(
        self,
        actor: nn.Module,
        comm_channel: CommunicationChannel,
        comm_encoder: nn.Module,  # Encode messages
        comm_decoder: nn.Module,  # Decode received messages
        **kwargs
    ):
        super().__init__(**kwargs)
        self.actor = actor
        self.comm_channel = comm_channel
        self.comm_encoder = comm_encoder
        self.comm_decoder = comm_decoder
        
    def forward(self, batch: Batch, state: Optional[Any] = None):
        """Forward with communication."""
        # Receive messages from other agents
        messages = self.comm_channel.receive(batch.agent_id)
        
        # Process messages
        if messages:
            comm_info = self.comm_decoder(torch.stack(messages))
        else:
            comm_info = torch.zeros(self.comm_decoder.output_dim)
        
        # Combine observation with communication
        enhanced_obs = torch.cat([batch.obs, comm_info], dim=-1)
        
        # Generate action and message
        action, message_raw = self.actor(enhanced_obs, state)
        
        # Send message
        message = self.comm_encoder(message_raw)
        self.comm_channel.send(batch.agent_id, message)
        
        return Batch(act=action, state=state)
```

**Rationale**: Communication is essential for coordination in partially observable environments and team-based scenarios.

### 7. Opponent Modeling

#### Current State
- No built-in support for modeling other agents
- Required for competitive and mixed-motive scenarios

#### Required Enhancement
```python
class OpponentModel(nn.Module):
    """Model to predict other agents' behaviors."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        model_type: Literal["deterministic", "stochastic"] = "stochastic"
    ):
        super().__init__()
        self.model_type = model_type
        
        # Encoder for opponent's observable features
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if model_type == "deterministic":
            # Predict single action
            self.predictor = nn.Linear(hidden_dim, action_dim)
        else:
            # Predict action distribution
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.logstd_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, opponent_obs: torch.Tensor) -> Union[torch.Tensor, torch.distributions.Distribution]:
        """Predict opponent's action/distribution."""
        features = self.encoder(opponent_obs)
        
        if self.model_type == "deterministic":
            return self.predictor(features)
        else:
            mean = self.mean_head(features)
            logstd = self.logstd_head(features)
            return torch.distributions.Normal(mean, logstd.exp())

class OpponentAwarePolicy(BasePolicy):
    """Policy that models and adapts to opponents."""
    
    def __init__(
        self,
        actor: nn.Module,
        opponent_models: Dict[str, OpponentModel],
        use_opponent_models: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.actor = actor
        self.opponent_models = nn.ModuleDict(opponent_models)
        self.use_opponent_models = use_opponent_models
        
    def forward(self, batch: Batch, state: Optional[Any] = None):
        """Act considering opponent predictions."""
        if self.use_opponent_models and hasattr(batch, "opponent_obs"):
            # Predict opponent actions
            opponent_predictions = {}
            for opp_id, opp_obs in batch.opponent_obs.items():
                if opp_id in self.opponent_models:
                    opponent_predictions[opp_id] = self.opponent_models[opp_id](opp_obs)
            
            # Augment observation with predictions
            aug_obs = self._augment_with_predictions(batch.obs, opponent_predictions)
        else:
            aug_obs = batch.obs
        
        # Generate action
        action, state = self.actor(aug_obs, state)
        return Batch(act=action, state=state)
    
    def update_opponent_models(self, batch: Batch):
        """Update opponent models with observed behavior."""
        for opp_id in batch.opponent_ids:
            if opp_id in self.opponent_models:
                opp_batch = batch.get_opponent_data(opp_id)
                # Supervised learning on opponent's action history
                pred_actions = self.opponent_models[opp_id](opp_batch.obs)
                loss = F.mse_loss(pred_actions, opp_batch.act)
                loss.backward()
```

**Rationale**: Understanding and predicting opponent behavior is crucial for strategic decision-making in competitive and mixed-motive scenarios.

## Implementation Priority

### Phase 1: Core Infrastructure (Essential)
1. **Enhanced Parallel Environment Support** - Required for simultaneous action scenarios
2. **Flexible Policy Configuration** - Enables efficient parameter sharing
3. **Dict Observation Space Support** - Handles complex, structured observations

### Phase 2: Training Enhancement (Important)
4. **Training Coordination Modes** - Supports diverse training strategies
5. **CTDE Support** - Critical for coordinated multi-agent systems

### Phase 3: Advanced Features (Nice-to-have)
6. **Communication Between Agents** - For cooperative scenarios
7. **Opponent Modeling** - For competitive scenarios

## Testing Requirements

Each enhancement should include:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test with standard PettingZoo environments
3. **Performance Benchmarks**: Compare against baseline implementations
4. **Documentation**: Clear examples and API documentation

### Example Test Case
```python
def test_parameter_sharing():
    """Test that parameter sharing reduces memory and improves training."""
    # Create environment with 10 identical agents
    env = make_pettingzoo_env("simple_spread_v3", n_agents=10)
    
    # Test independent policies
    independent_manager = FlexibleMultiAgentPolicyManager(
        policies=[PPOPolicy(...) for _ in range(10)],
        env=env,
        mode="independent"
    )
    
    # Test shared policy
    shared_manager = FlexibleMultiAgentPolicyManager(
        policies=PPOPolicy(...),
        env=env,
        mode="shared"
    )
    
    # Compare memory usage
    independent_memory = get_memory_usage(independent_manager)
    shared_memory = get_memory_usage(shared_manager)
    assert shared_memory < independent_memory * 0.2  # Should use <20% memory
    
    # Train and compare convergence
    independent_rewards = train(independent_manager, steps=10000)
    shared_rewards = train(shared_manager, steps=10000)
    
    # Shared should converge faster for identical agents
    assert shared_rewards[-1] > independent_rewards[-1] * 0.9
```

## Backward Compatibility

All enhancements should maintain backward compatibility with existing Tianshou code:

1. New features should be optional (default to current behavior)
2. Existing APIs should continue to work
3. Deprecation warnings for any changed interfaces

## Performance Considerations

1. **Memory Efficiency**: Parameter sharing should significantly reduce memory for homogeneous agents
2. **Computational Efficiency**: Parallel environments should vectorize operations
3. **Scalability**: Support 100+ agents for swarm scenarios
4. **GPU Utilization**: Batch operations across agents when possible

## Conclusion

These enhancements would transform Tianshou from experimental MARL support to a production-ready MARL framework. The modular design allows incremental implementation while maintaining backward compatibility. Priority should be given to Phase 1 features as they address fundamental limitations in current MARL support.