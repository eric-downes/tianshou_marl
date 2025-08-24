"""Tests for multi-agent training coordination modes."""

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

from tianshou.algorithm.algorithm_base import Policy
from tianshou.algorithm.multiagent.flexible_policy import (
    FlexibleMultiAgentPolicyManager,
)
from tianshou.algorithm.multiagent.training_coordinator import (
    LeaguePlayTrainer,
    MATrainer,
    SelfPlayTrainer,
    SequentialTrainer,
    SimultaneousTrainer,
)
from tianshou.data import Batch, ReplayBuffer


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, agent_id: str = "agent"):
        # Provide default spaces
        obs_space = spaces.Box(low=-1, high=1, shape=(4,))
        act_space = spaces.Discrete(2)
        super().__init__(observation_space=obs_space, action_space=act_space)
        self.agent_id = agent_id
        self.learn_count = 0
        self.last_batch = None
        self.training = True

    def forward(self, batch: Batch, state=None, **kwargs):
        """Simple forward that returns random actions."""
        batch_size = batch.obs.shape[0] if hasattr(batch.obs, "shape") else 1
        return Batch(act=np.random.randint(0, 2, size=batch_size), state=state)

    def learn(self, batch: Batch, **kwargs):
        """Mock learn method that tracks calls."""
        self.learn_count += 1
        self.last_batch = batch
        return {"loss": np.random.random(), "agent": self.agent_id}

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set eval mode."""
        return self.train(False)


class MockEnvironment:
    """Mock multi-agent environment."""

    def __init__(self, n_agents: int = 3):
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = spaces.Discrete(2)

    def reset(self):
        """Reset environment."""
        obs = {agent: np.random.randn(4) for agent in self.agents}
        return obs, {}

    def step(self, actions):
        """Step environment."""
        obs = {agent: np.random.randn(4) for agent in self.agents}
        rewards = {agent: np.random.random() for agent in self.agents}
        terms = dict.fromkeys(self.agents, False)
        truncs = dict.fromkeys(self.agents, False)
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, terms, truncs, infos


def create_mock_batch(n_agents: int = 3, batch_size: int = 32) -> Batch:
    """Create a mock batch for multi-agent training."""
    batch = Batch()
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        agent_batch = Batch(
            obs=np.random.randn(batch_size, 4),
            act=np.random.randint(0, 2, size=batch_size),
            rew=np.random.randn(batch_size),
            terminated=np.zeros(batch_size, dtype=bool),
            truncated=np.zeros(batch_size, dtype=bool),
            obs_next=np.random.randn(batch_size, 4),
            info={},
        )
        batch[agent_id] = agent_batch
    return batch


@pytest.mark.slow
class TestMATrainer:
    """Test base MATrainer class."""

    def test_trainer_initialization(self):
        """Test MATrainer can be initialized."""
        env = MockEnvironment()
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        trainer = MATrainer(policy_manager=manager, training_mode="simultaneous")

        assert trainer.policy_manager == manager
        assert trainer.training_mode == "simultaneous"
        assert trainer.step_count == 0

    def test_invalid_training_mode(self):
        """Test error on invalid training mode."""
        env = MockEnvironment()
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        with pytest.raises(ValueError, match="Invalid training mode"):
            MATrainer(policy_manager=manager, training_mode="invalid_mode")

    def test_train_step_dispatches_correctly(self):
        """Test train_step calls appropriate training method."""
        env = MockEnvironment()
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        trainer = MATrainer(policy_manager=manager, training_mode="simultaneous")
        batch = create_mock_batch()

        # Mock the specific training method
        trainer._simultaneous_train = MagicMock(return_value={"loss": 0.1})

        result = trainer.train_step(batch)

        trainer._simultaneous_train.assert_called_once_with(batch)
        assert "loss" in result


@pytest.mark.slow
class TestSimultaneousTrainer:
    """Test simultaneous training mode."""

    def test_simultaneous_training(self):
        """Test all agents train simultaneously."""
        env = MockEnvironment(n_agents=3)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        trainer = SimultaneousTrainer(policy_manager=manager)
        batch = create_mock_batch(n_agents=3)

        # Train step
        losses = trainer.train_step(batch)

        # All policies should have learned
        for agent, policy in policies.items():
            assert policy.learn_count == 1
            assert agent in losses
            assert "loss" in losses[agent]

    def test_simultaneous_with_shared_policy(self):
        """Test simultaneous training with parameter sharing."""
        env = MockEnvironment(n_agents=3)
        shared_policy = MockPolicy("shared")
        manager = FlexibleMultiAgentPolicyManager(policies=shared_policy, env=env, mode="shared")

        trainer = SimultaneousTrainer(policy_manager=manager)
        batch = create_mock_batch(n_agents=3)

        # Train step
        losses = trainer.train_step(batch)

        # Shared policy should have learned once per agent
        assert shared_policy.learn_count == 3
        assert len(losses) == 3

    def test_simultaneous_respects_training_frequency(self):
        """Test that training frequency is respected."""
        env = MockEnvironment(n_agents=3)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        # Set different training frequencies
        trainer = SimultaneousTrainer(
            policy_manager=manager, agent_train_freq={"agent_0": 1, "agent_1": 2, "agent_2": 3}
        )

        batch = create_mock_batch(n_agents=3)

        # First step - only agent_0 should train
        trainer.train_step(batch)
        assert policies["agent_0"].learn_count == 1
        assert policies["agent_1"].learn_count == 0
        assert policies["agent_2"].learn_count == 0

        # Second step - agent_0 and agent_1 should train
        trainer.train_step(batch)
        assert policies["agent_0"].learn_count == 2
        assert policies["agent_1"].learn_count == 1
        assert policies["agent_2"].learn_count == 0

        # Third step - all agents should have trained
        trainer.train_step(batch)
        assert policies["agent_0"].learn_count == 3
        assert policies["agent_1"].learn_count == 1
        assert policies["agent_2"].learn_count == 1


@pytest.mark.slow
class TestSequentialTrainer:
    """Test sequential training mode."""

    def test_sequential_training(self):
        """Test agents train one at a time."""
        env = MockEnvironment(n_agents=3)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        trainer = SequentialTrainer(policy_manager=manager)
        batch = create_mock_batch(n_agents=3)

        # First step - only agent_0 trains
        losses = trainer.train_step(batch)
        assert policies["agent_0"].learn_count == 1
        assert policies["agent_1"].learn_count == 0
        assert policies["agent_2"].learn_count == 0

        # Second step - only agent_1 trains
        losses = trainer.train_step(batch)
        assert policies["agent_0"].learn_count == 1
        assert policies["agent_1"].learn_count == 1
        assert policies["agent_2"].learn_count == 0

        # Third step - only agent_2 trains
        losses = trainer.train_step(batch)
        assert policies["agent_0"].learn_count == 1
        assert policies["agent_1"].learn_count == 1
        assert policies["agent_2"].learn_count == 1

        # Fourth step - cycles back to agent_0
        losses = trainer.train_step(batch)
        assert policies["agent_0"].learn_count == 2
        assert policies["agent_1"].learn_count == 1
        assert policies["agent_2"].learn_count == 1

    def test_sequential_with_custom_order(self):
        """Test sequential training with custom agent order."""
        env = MockEnvironment(n_agents=3)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        # Custom order
        custom_order = ["agent_2", "agent_0", "agent_1"]
        trainer = SequentialTrainer(policy_manager=manager, agent_order=custom_order)
        batch = create_mock_batch(n_agents=3)

        # First step - agent_2 trains
        trainer.train_step(batch)
        assert policies["agent_2"].learn_count == 1
        assert policies["agent_0"].learn_count == 0
        assert policies["agent_1"].learn_count == 0

        # Second step - agent_0 trains
        trainer.train_step(batch)
        assert policies["agent_2"].learn_count == 1
        assert policies["agent_0"].learn_count == 1
        assert policies["agent_1"].learn_count == 0


@pytest.mark.slow
class TestSelfPlayTrainer:
    """Test self-play training mode."""

    def test_self_play_initialization(self):
        """Test self-play trainer initialization."""
        env = MockEnvironment(n_agents=2)
        main_policy = MockPolicy("main")
        manager = FlexibleMultiAgentPolicyManager(
            policies={"agent_0": main_policy, "agent_1": main_policy}, env=env, mode="independent"
        )

        trainer = SelfPlayTrainer(
            policy_manager=manager,
            main_agent_id="agent_0",
            snapshot_interval=100,
            opponent_pool_size=10,
        )

        assert trainer.main_agent_id == "agent_0"
        assert trainer.snapshot_interval == 100
        assert trainer.opponent_pool_size == 10
        assert len(trainer.opponent_pool) == 0

    def test_self_play_training(self):
        """Test self-play training updates only main agent."""
        env = MockEnvironment(n_agents=2)
        main_policy = MockPolicy("main")
        opponent_policy = MockPolicy("opponent")

        manager = FlexibleMultiAgentPolicyManager(
            policies={"agent_0": main_policy, "agent_1": opponent_policy},
            env=env,
            mode="independent",
        )

        trainer = SelfPlayTrainer(policy_manager=manager, main_agent_id="agent_0")

        batch = create_mock_batch(n_agents=2)

        # Train step - only main agent should learn
        losses = trainer.train_step(batch)

        assert main_policy.learn_count == 1
        assert opponent_policy.learn_count == 0
        assert "agent_0" in losses
        assert "agent_1" not in losses

    def test_self_play_snapshot_creation(self):
        """Test that snapshots are created at intervals."""
        env = MockEnvironment(n_agents=2)
        main_policy = MockPolicy("main")

        manager = FlexibleMultiAgentPolicyManager(
            policies={"agent_0": main_policy, "agent_1": main_policy}, env=env, mode="independent"
        )

        trainer = SelfPlayTrainer(
            policy_manager=manager, main_agent_id="agent_0", snapshot_interval=3  # Every 3 steps
        )

        batch = create_mock_batch(n_agents=2)

        # Train for several steps
        for i in range(10):
            trainer.train_step(batch)

        # Should have created snapshots at steps 3, 6, 9
        assert len(trainer.opponent_pool) == 3

    def test_self_play_opponent_sampling(self):
        """Test opponent sampling from pool."""
        env = MockEnvironment(n_agents=2)
        main_policy = MockPolicy("main")

        manager = FlexibleMultiAgentPolicyManager(
            policies={"agent_0": main_policy, "agent_1": main_policy}, env=env, mode="independent"
        )

        trainer = SelfPlayTrainer(
            policy_manager=manager,
            main_agent_id="agent_0",
            snapshot_interval=1,  # Snapshot every step
            opponent_sampling="uniform",
        )

        batch = create_mock_batch(n_agents=2)

        # Create opponent pool
        for i in range(5):
            trainer.train_step(batch)

        assert len(trainer.opponent_pool) == 5

        # Sample opponent
        opponent = trainer._sample_opponent()
        assert opponent is not None
        assert opponent in trainer.opponent_pool

    def test_self_play_with_prioritized_sampling(self):
        """Test prioritized opponent sampling based on win rate."""
        env = MockEnvironment(n_agents=2)
        main_policy = MockPolicy("main")

        manager = FlexibleMultiAgentPolicyManager(
            policies={"agent_0": main_policy, "agent_1": main_policy}, env=env, mode="independent"
        )

        trainer = SelfPlayTrainer(
            policy_manager=manager,
            main_agent_id="agent_0",
            opponent_sampling="prioritized",
            win_rate_window=10,
        )

        # Manually add opponents with different win rates
        for i in range(3):
            snapshot = copy.deepcopy(main_policy)
            trainer.opponent_pool.append(snapshot)
            # Simulate different win rates
            trainer.opponent_win_rates[id(snapshot)] = 0.2 * i

        # Sample many times and check distribution
        samples = []
        for _ in range(100):
            opponent = trainer._sample_opponent()
            samples.append(trainer.opponent_pool.index(opponent))

        # Higher win rate opponents should be sampled more often
        assert samples.count(2) > samples.count(0)


@pytest.mark.slow
class TestLeaguePlayTrainer:
    """Test league play training mode."""

    def test_league_play_initialization(self):
        """Test league play trainer initialization."""
        env = MockEnvironment(n_agents=4)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        trainer = LeaguePlayTrainer(
            policy_manager=manager,
            league_size=10,
            promotion_threshold=0.6,
            relegation_threshold=0.4,
        )

        assert trainer.league_size == 10
        assert trainer.promotion_threshold == 0.6
        assert trainer.relegation_threshold == 0.4
        assert len(trainer.league) == 4  # Initial policies

    def test_league_play_matchmaking(self):
        """Test matchmaking in league play."""
        env = MockEnvironment(n_agents=4)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        trainer = LeaguePlayTrainer(
            policy_manager=manager, matchmaking="elo"  # Elo-based matchmaking
        )

        # Assign different Elo ratings
        for i, agent in enumerate(env.agents):
            trainer.elo_ratings[agent] = 1000 + i * 100

        # Get match
        match = trainer._make_match()

        assert len(match) == 2  # Pair of agents
        assert all(agent in env.agents for agent in match)

    def test_league_play_promotion_relegation(self):
        """Test promotion and relegation in league."""
        env = MockEnvironment(n_agents=4)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        trainer = LeaguePlayTrainer(
            policy_manager=manager, promotion_threshold=0.7, relegation_threshold=0.3
        )

        # Simulate performance
        trainer.agent_performance = {
            "agent_0": 0.8,  # Should be promoted
            "agent_1": 0.5,  # Stays
            "agent_2": 0.2,  # Should be relegated
            "agent_3": 0.6,  # Stays
        }

        promoted, relegated = trainer._update_league()

        assert "agent_0" in promoted
        assert "agent_2" in relegated
        assert len(promoted) == 1
        assert len(relegated) == 1


@pytest.mark.slow
class TestIntegration:
    """Integration tests for training coordination."""

    def test_training_with_replay_buffer(self):
        """Test training with replay buffer integration."""
        env = MockEnvironment(n_agents=2)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        # Create replay buffers for each agent
        buffers = {agent: ReplayBuffer(size=1000) for agent in env.agents}

        trainer = SimultaneousTrainer(policy_manager=manager, replay_buffers=buffers)

        # Collect some data
        for _ in range(100):
            batch = create_mock_batch(n_agents=2, batch_size=1)
            for agent in env.agents:
                buffers[agent].add(batch[agent])

        # Sample and train
        sampled_batch = Batch()
        for agent, buffer in buffers.items():
            sampled_batch[agent], _ = buffer.sample(32)

        losses = trainer.train_step(sampled_batch)

        assert all(agent in losses for agent in env.agents)
        assert all(policies[agent].learn_count == 1 for agent in env.agents)

    def test_mode_switching(self):
        """Test switching between training modes."""
        env = MockEnvironment(n_agents=3)
        policies = {agent: MockPolicy(agent) for agent in env.agents}
        manager = FlexibleMultiAgentPolicyManager(policies=policies, env=env, mode="independent")

        # Start with simultaneous
        trainer = MATrainer(policy_manager=manager, training_mode="simultaneous")

        batch = create_mock_batch(n_agents=3)

        # Train with simultaneous
        trainer.train_step(batch)
        assert all(policies[agent].learn_count == 1 for agent in env.agents)

        # Switch to sequential
        trainer.set_training_mode("sequential")

        # Reset learn counts
        for policy in policies.values():
            policy.learn_count = 0

        # Train with sequential
        trainer.train_step(batch)
        assert sum(policy.learn_count for policy in policies.values()) == 1

    def test_checkpoint_save_load(self):
        """Test saving and loading trainer state."""
        env = MockEnvironment(n_agents=2)
        main_policy = MockPolicy("main")

        manager = FlexibleMultiAgentPolicyManager(
            policies={"agent_0": main_policy, "agent_1": main_policy}, env=env, mode="independent"
        )

        trainer = SelfPlayTrainer(
            policy_manager=manager, main_agent_id="agent_0", snapshot_interval=1
        )

        batch = create_mock_batch(n_agents=2)

        # Train and create snapshots
        for _ in range(3):
            trainer.train_step(batch)

        # Save state
        state = trainer.state_dict()

        assert "step_count" in state
        assert "opponent_pool_size" in state
        assert state["opponent_pool_size"] == 3

        # Create new trainer and load state
        new_trainer = SelfPlayTrainer(policy_manager=manager, main_agent_id="agent_0")

        new_trainer.load_state_dict(state)

        assert new_trainer.step_count == trainer.step_count
        # Note: opponent pool itself is not saved, only the size
        # The actual pool would need to be reconstructed from saved policies
