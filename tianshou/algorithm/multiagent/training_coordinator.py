"""Multi-agent training coordination modes.

This module provides flexible training coordination strategies for multi-agent
reinforcement learning, including simultaneous, sequential, self-play, and
league play training modes.
"""

import copy
from collections import deque
from typing import Any, Literal

import numpy as np

from tianshou.algorithm.algorithm_base import Policy
from tianshou.algorithm.multiagent.flexible_policy import (
    FlexibleMultiAgentPolicyManager,
)
from tianshou.data import Batch, ReplayBuffer


class MATrainer:
    """Multi-agent trainer with flexible training coordination.

    This trainer coordinates how multiple agents learn from their experiences,
    supporting various training schemes optimized for different multi-agent scenarios.

    Example usage:
    ::
        # Simultaneous training (all agents learn together)
        trainer = MATrainer(
            policy_manager=policy_manager,
            training_mode="simultaneous"
        )

        # Sequential training (agents take turns learning)
        trainer = MATrainer(
            policy_manager=policy_manager,
            training_mode="sequential"
        )

        # Self-play training (agent learns against past versions)
        trainer = MATrainer(
            policy_manager=policy_manager,
            training_mode="self_play"
        )
    """

    VALID_MODES = ["simultaneous", "sequential", "self_play", "league"]

    def __init__(
        self,
        policy_manager: FlexibleMultiAgentPolicyManager,
        training_mode: Literal[
            "simultaneous", "sequential", "self_play", "league"
        ] = "simultaneous",
        **kwargs: Any,
    ) -> None:
        """Initialize multi-agent trainer.

        Args:
            policy_manager: Multi-agent policy manager
            training_mode: How agents are trained
                - "simultaneous": All agents train together
                - "sequential": Agents train one at a time
                - "self_play": Agent trains against past versions
                - "league": Population-based training
            **kwargs: Mode-specific parameters

        """
        if training_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid training mode: {training_mode}. Must be one of {self.VALID_MODES}"
            )

        self.policy_manager = policy_manager
        self.training_mode = training_mode
        self.step_count = 0
        self.kwargs = kwargs

    def train_step(self, batch: Batch) -> dict[str, Any]:
        """Execute one training step based on mode.

        Args:
            batch: Training batch containing agent experiences

        Returns:
            Dictionary of training metrics

        """
        self.step_count += 1

        if self.training_mode == "simultaneous":
            return self._simultaneous_train(batch)
        elif self.training_mode == "sequential":
            return self._sequential_train(batch)
        elif self.training_mode == "self_play":
            return self._self_play_train(batch)
        elif self.training_mode == "league":
            return self._league_train(batch)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

    def _simultaneous_train(self, batch: Batch) -> dict[str, Any]:
        """Simultaneous training - all agents learn together.

        Args:
            batch: Multi-agent batch

        Returns:
            Dictionary mapping agent_id to loss metrics

        """
        losses: dict[str, Any] = {}

        # Train all agents
        for agent_id in self.policy_manager.policies.keys():
            if agent_id in batch:  # type: ignore[operator]
                agent_batch = batch[agent_id]

                # Add global state if present
                if hasattr(batch, "global_obs"):
                    agent_batch.global_obs = batch.global_obs
                if hasattr(batch, "global_obs_next"):
                    agent_batch.global_obs_next = batch.global_obs_next

                policy = self.policy_manager.policies[agent_id]
                losses[str(agent_id)] = policy.learn(agent_batch)

        return losses

    def _sequential_train(self, batch: Batch) -> dict[str, Any]:
        """Sequential training - agents train one at a time.

        Args:
            batch: Multi-agent batch

        Returns:
            Dictionary with single agent's loss metrics

        """
        # Initialize sequential attributes if not present
        if not hasattr(self, "current_agent_idx"):
            self.current_agent_idx = 0
            self.current_agent_steps = 0
            self.agent_order = sorted(self.policy_manager.policies.keys())
            self.steps_per_agent = 1

        losses: dict[str, Any] = {}

        # Get current agent
        current_agent = self.agent_order[self.current_agent_idx]

        # Train only current agent
        if current_agent in batch:  # type: ignore[operator]
            agent_batch = batch[current_agent]

            # Add global state if present
            if hasattr(batch, "global_obs"):
                agent_batch.global_obs = batch.global_obs
            if hasattr(batch, "global_obs_next"):
                agent_batch.global_obs_next = batch.global_obs_next

            policy = self.policy_manager.policies[current_agent]
            losses[str(current_agent)] = policy.learn(agent_batch)

        # Update counters
        self.current_agent_steps += 1
        if self.current_agent_steps >= self.steps_per_agent:
            self.current_agent_steps = 0
            self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agent_order)

        return losses

    def _self_play_train(self, batch: Batch) -> dict[str, Any]:
        """Self-play training - agent trains against past versions.

        Args:
            batch: Multi-agent batch

        Returns:
            Dictionary with main agent's loss metrics

        """
        # This is a placeholder - will be implemented by SelfPlayTrainer
        raise NotImplementedError("Use SelfPlayTrainer for self-play training")

    def _league_train(self, batch: Batch) -> dict[str, Any]:
        """League play training - population-based training.

        Args:
            batch: Multi-agent batch

        Returns:
            Dictionary with training metrics

        """
        # This is a placeholder - will be implemented by LeaguePlayTrainer
        raise NotImplementedError("Use LeaguePlayTrainer for league play training")

    def set_training_mode(self, mode: Literal["simultaneous", "sequential", "self_play", "league"]) -> None:
        """Change training mode.

        Args:
            mode: New training mode

        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid training mode: {mode}")
        self.training_mode = mode

        # Initialize mode-specific attributes if needed
        if mode == "sequential" and not hasattr(self, "current_agent_idx"):
            self.current_agent_idx = 0
            self.current_agent_steps = 0
            self.agent_order = sorted(self.policy_manager.policies.keys())
            self.steps_per_agent = 1

    def state_dict(self) -> dict[str, Any]:
        """Get trainer state for checkpointing.

        Returns:
            Dictionary containing trainer state

        """
        return {
            "step_count": self.step_count,
            "training_mode": self.training_mode,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainer state from checkpoint.

        Args:
            state: Saved trainer state

        """
        self.step_count = state.get("step_count", 0)
        self.training_mode = state.get("training_mode", "simultaneous")

    def save_checkpoint(self, path: str) -> None:
        """Save trainer checkpoint to file.

        Args:
            path: Path to save checkpoint

        """
        import torch

        checkpoint = {"trainer_state": self.state_dict(), "policies": {}}

        # Save each policy's state
        for agent_id, policy in self.policy_manager.policies.items():
            if hasattr(policy, "state_dict"):
                checkpoint["policies"][str(agent_id)] = policy.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load trainer checkpoint from file.

        Args:
            path: Path to checkpoint file

        """
        import torch

        checkpoint = torch.load(path)

        # Load trainer state
        if "trainer_state" in checkpoint:
            self.load_state_dict(checkpoint["trainer_state"])

        # Load policy states
        if "policies" in checkpoint:
            for agent_id, policy_state in checkpoint["policies"].items():
                if agent_id in self.policy_manager.policies:
                    policy = self.policy_manager.policies[agent_id]
                    if hasattr(policy, "load_state_dict"):
                        policy.load_state_dict(policy_state)


class SimultaneousTrainer(MATrainer):
    """Simultaneous training coordinator.

    All agents learn from their experiences at the same time.
    Supports different training frequencies for each agent.
    """

    def __init__(
        self,
        policy_manager: FlexibleMultiAgentPolicyManager,
        agent_train_freq: dict[str, int] | None = None,
        replay_buffers: dict[str, ReplayBuffer] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize simultaneous trainer.

        Args:
            policy_manager: Multi-agent policy manager
            agent_train_freq: Training frequency for each agent (default: 1 for all)
            replay_buffers: Optional replay buffers for each agent
            **kwargs: Additional parameters

        """
        super().__init__(policy_manager, "simultaneous", **kwargs)
        self.agent_train_freq = agent_train_freq or {}
        self.replay_buffers = replay_buffers or {}

    def train_step(self, batch: Batch) -> dict[str, Any]:
        """Execute simultaneous training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary of losses for agents that trained

        """
        self.step_count += 1
        losses = {}

        # Get agents to train
        if self.policy_manager.mode == "shared":
            # For shared mode, train all agents together
            agents_to_train = self.policy_manager.agents
        else:
            # For other modes, use policies keys
            agents_to_train = self.policy_manager.policies.keys()

        for agent_id in agents_to_train:
            # Check training frequency
            freq = self.agent_train_freq.get(agent_id, 1)
            if self.step_count % freq != 0:
                continue

            if agent_id in batch:
                agent_batch = batch[agent_id]

                # Add global state if present in the batch
                if hasattr(batch, "global_obs"):
                    agent_batch.global_obs = batch.global_obs
                if hasattr(batch, "global_obs_next"):
                    agent_batch.global_obs_next = batch.global_obs_next

                # Get policy for this agent
                if self.policy_manager.mode == "shared":
                    policy = self.policy_manager.policies["shared"]
                    # Train the policy
                    policy.train()
                    losses[str(agent_id)] = policy.learn(agent_batch)
                else:
                    policy_maybe = self.policy_manager.policies.get(agent_id)
                    if policy_maybe is not None:
                        # Train the policy
                        policy_maybe.train()
                        losses[str(agent_id)] = policy_maybe.learn(agent_batch)

        return losses


class SequentialTrainer(MATrainer):
    """Sequential training coordinator.

    Agents train one at a time in a round-robin fashion.
    Useful for scenarios where training stability is important.
    """

    def __init__(
        self,
        policy_manager: FlexibleMultiAgentPolicyManager,
        agent_order: list[str] | None = None,
        steps_per_agent: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize sequential trainer.

        Args:
            policy_manager: Multi-agent policy manager
            agent_order: Order in which agents train (default: alphabetical)
            steps_per_agent: Number of training steps per agent before switching
            **kwargs: Additional parameters

        """
        super().__init__(policy_manager, "sequential", **kwargs)

        # Set agent training order
        if agent_order:
            self.agent_order = agent_order  # type: ignore[assignment]
        else:
            self.agent_order = [str(key) for key in sorted(policy_manager.policies.keys())]

        self.steps_per_agent = steps_per_agent
        self.current_agent_idx = 0
        self.current_agent_steps = 0

    def train_step(self, batch: Batch) -> dict[str, Any]:
        """Execute sequential training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary with current agent's loss

        """
        self.step_count += 1
        losses = {}

        # Get current agent
        current_agent = self.agent_order[self.current_agent_idx]

        # Train only current agent
        if current_agent in batch:  # type: ignore[operator]
            agent_batch = batch[current_agent]
            policy = self.policy_manager.policies[current_agent]

            # Set current agent to train mode, others to eval
            for agent_id, p in self.policy_manager.policies.items():
                if agent_id == current_agent:
                    p.train()
                else:
                    p.eval()

            losses[str(current_agent)] = policy.learn(agent_batch)

        # Update counters
        self.current_agent_steps += 1
        if self.current_agent_steps >= self.steps_per_agent:
            self.current_agent_steps = 0
            self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agent_order)

        return losses


class SelfPlayTrainer(MATrainer):
    """Self-play training coordinator.

    Main agent trains against historical versions of itself.
    Prevents overfitting to specific opponent strategies.
    """

    def __init__(
        self,
        policy_manager: FlexibleMultiAgentPolicyManager,
        main_agent_id: str,
        snapshot_interval: int = 100,
        opponent_pool_size: int = 20,
        opponent_sampling: Literal["uniform", "prioritized", "latest"] = "uniform",
        win_rate_window: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize self-play trainer.

        Args:
            policy_manager: Multi-agent policy manager
            main_agent_id: ID of the learning agent
            snapshot_interval: Steps between policy snapshots
            opponent_pool_size: Maximum size of opponent pool
            opponent_sampling: How to sample opponents from pool
            win_rate_window: Window for tracking win rates (for prioritized sampling)
            **kwargs: Additional parameters

        """
        super().__init__(policy_manager, "self_play", **kwargs)

        self.main_agent_id = main_agent_id
        self.snapshot_interval = snapshot_interval
        self.opponent_pool_size = opponent_pool_size
        self.opponent_sampling = opponent_sampling
        self.win_rate_window = win_rate_window

        # Initialize opponent pool
        self.opponent_pool: list[Policy] = []
        self.opponent_win_rates: dict[int, float] = {}
        self.win_history: deque[bool] = deque(maxlen=win_rate_window)

    def train_step(self, batch: Batch) -> dict[str, Any]:
        """Execute self-play training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary with main agent's loss

        """
        self.step_count += 1
        losses = {}

        # Only train main agent
        if self.main_agent_id in batch:
            agent_batch = batch[self.main_agent_id]
            policy = self.policy_manager.policies[self.main_agent_id]

            # Set main agent to train mode
            policy.train()
            losses[self.main_agent_id] = policy.learn(agent_batch)

        # Create snapshot periodically
        if self.step_count % self.snapshot_interval == 0:
            self._create_snapshot()

        return losses

    def _create_snapshot(self) -> Policy:
        """Create a snapshot of current main policy."""
        main_policy = self.policy_manager.policies[self.main_agent_id]
        snapshot = copy.deepcopy(main_policy)
        snapshot.eval()  # Snapshots don't learn

        # Add to pool
        self.opponent_pool.append(snapshot)

        # Remove oldest if pool is full
        if len(self.opponent_pool) > self.opponent_pool_size:
            oldest = self.opponent_pool.pop(0)
            if id(oldest) in self.opponent_win_rates:
                del self.opponent_win_rates[id(oldest)]

        return snapshot

    def _sample_opponent(self) -> Policy | None:
        """Sample an opponent from the pool.

        Returns:
            Sampled opponent policy or None if pool is empty

        """
        if not self.opponent_pool:
            return None

        if self.opponent_sampling == "uniform":
            return np.random.choice(self.opponent_pool)

        elif self.opponent_sampling == "latest":
            return self.opponent_pool[-1]

        elif self.opponent_sampling == "prioritized":
            # Sample based on win rates (harder opponents more likely)
            if not self.opponent_win_rates:
                return np.random.choice(self.opponent_pool)

            # Calculate sampling weights based on win rates
            weights = []
            for opponent in self.opponent_pool:
                win_rate = self.opponent_win_rates.get(id(opponent), 0.5)
                # Higher win rate = higher weight (harder opponent)
                weights.append(win_rate + 0.1)  # Add small constant to avoid zero

            weights = np.array(weights)
            weights = weights / weights.sum()

            return np.random.choice(self.opponent_pool, p=weights)

        # All cases handled above
        raise ValueError(f"Unknown opponent_sampling type: {self.opponent_sampling}")

    def update_win_rate(self, opponent_id: int, won: bool) -> None:
        """Update win rate against an opponent.

        Args:
            opponent_id: ID of opponent policy
            won: Whether main agent won

        """
        if opponent_id not in self.opponent_win_rates:
            self.opponent_win_rates[opponent_id] = 0.5

        # Exponential moving average
        alpha = 0.1
        self.opponent_win_rates[opponent_id] = (
            alpha * (1.0 if won else 0.0) + (1 - alpha) * self.opponent_win_rates[opponent_id]
        )

    def state_dict(self) -> dict[str, Any]:
        """Get trainer state for checkpointing.

        Returns:
            Dictionary containing trainer state

        """
        state = super().state_dict()
        state.update(
            {
                "opponent_pool_size": len(self.opponent_pool),
                "main_agent_id": self.main_agent_id,
                "opponent_win_rates": self.opponent_win_rates,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainer state from checkpoint.

        Args:
            state: Saved trainer state

        """
        super().load_state_dict(state)
        self.main_agent_id = state.get("main_agent_id", self.main_agent_id)
        self.opponent_win_rates = state.get("opponent_win_rates", {})
        # Note: Actual opponent pool needs to be reconstructed separately


class LeaguePlayTrainer(MATrainer):
    """League play training coordinator.

    Population-based training with matchmaking and promotion/relegation.
    Inspired by AlphaStar League.
    """

    def __init__(
        self,
        policy_manager: FlexibleMultiAgentPolicyManager,
        league_size: int = 16,
        promotion_threshold: float = 0.6,
        relegation_threshold: float = 0.4,
        matchmaking: Literal["random", "elo", "win_rate"] = "random",
        games_per_evaluation: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize league play trainer.

        Args:
            policy_manager: Multi-agent policy manager
            league_size: Maximum size of the league
            promotion_threshold: Win rate threshold for promotion
            relegation_threshold: Win rate threshold for relegation
            matchmaking: How to match agents for games
            games_per_evaluation: Games before evaluating promotion/relegation
            **kwargs: Additional parameters

        """
        super().__init__(policy_manager, "league", **kwargs)

        self.league_size = league_size
        self.promotion_threshold = promotion_threshold
        self.relegation_threshold = relegation_threshold
        self.matchmaking = matchmaking
        self.games_per_evaluation = games_per_evaluation

        # Initialize league with current policies
        self.league = list(policy_manager.policies.keys())

        # Performance tracking
        self.agent_performance = dict.fromkeys(self.league, 0.5)
        self.elo_ratings = dict.fromkeys(self.league, 1000)
        self.game_count = 0
        self.match_history: deque[tuple[str, str, str]] = deque(maxlen=100)  # (winner, loser, timestamp)

    def train_step(self, batch: Batch) -> dict[str, Any]:
        """Execute league play training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary with training metrics

        """
        self.step_count += 1
        self.game_count += 1
        losses = {}

        # Make a match
        match = self._make_match()

        # Train matched agents
        for agent_id in match:
            if agent_id in batch:
                agent_batch = batch[agent_id]
                policy = self.policy_manager.policies[agent_id]
                policy.train()
                losses[str(agent_id)] = policy.learn(agent_batch)

        # Update league periodically
        if self.game_count % self.games_per_evaluation == 0:
            self._update_league()

        return losses

    def _make_match(self) -> list[str]:
        """Create a match between agents.

        Returns:
            List of agent IDs for the match

        """
        if self.matchmaking == "random":
            # Random pairing
            if len(self.league) < 2:
                return [str(agent) for agent in self.league]
            return [str(agent) for agent in np.random.choice(self.league, size=2, replace=False)]

        elif self.matchmaking == "elo":
            # Elo-based matchmaking (similar skill levels)
            if len(self.league) < 2:
                return [str(agent) for agent in self.league]

            # Sort by Elo rating
            sorted_agents = sorted(self.league, key=lambda x: self.elo_ratings[x])

            # Pick two agents with similar Elo
            idx = np.random.randint(0, len(sorted_agents) - 1)
            return [str(sorted_agents[idx]), str(sorted_agents[idx + 1])]

        elif self.matchmaking == "win_rate":
            # Match based on win rates
            if len(self.league) < 2:
                return [str(agent) for agent in self.league]

            # Sort by performance
            sorted_agents = sorted(self.league, key=lambda x: self.agent_performance[x])

            # Pick two agents with similar performance
            idx = np.random.randint(0, len(sorted_agents) - 1)
            return [str(sorted_agents[idx]), str(sorted_agents[idx + 1])]

        # All matchmaking types handled above
        raise ValueError(f"Unknown matchmaking type: {self.matchmaking}")

    def _update_league(self) -> tuple[list[str], list[str]]:
        """Update league with promotion and relegation.

        Returns:
            Tuple of (promoted agents, relegated agents)

        """
        promoted: list[str] = []
        relegated: list[str] = []

        for agent_id, performance in self.agent_performance.items():
            if performance >= self.promotion_threshold:
                promoted.append(str(agent_id))
                # Could add logic to move to higher league

            elif performance <= self.relegation_threshold:
                relegated.append(str(agent_id))
                # Could add logic to move to lower league

        return promoted, relegated

    def update_match_result(self, winner: str, loser: str) -> None:
        """Update ratings based on match result.

        Args:
            winner: ID of winning agent
            loser: ID of losing agent

        """
        # Update performance
        alpha = 0.1
        self.agent_performance[winner] = alpha * 1.0 + (1 - alpha) * self.agent_performance[winner]
        self.agent_performance[loser] = alpha * 0.0 + (1 - alpha) * self.agent_performance[loser]

        # Update Elo ratings
        self._update_elo(winner, loser)

        # Record match
        import time
        self.match_history.append((winner, loser, str(time.time())))

    def _update_elo(self, winner: str, loser: str, k: float = 32) -> None:
        """Update Elo ratings.

        Args:
            winner: ID of winning agent
            loser: ID of losing agent
            k: Elo K-factor

        """
        winner_elo = self.elo_ratings[winner]
        loser_elo = self.elo_ratings[loser]

        # Expected scores
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 - expected_winner

        # Update ratings
        self.elo_ratings[winner] = int(winner_elo + k * (1 - expected_winner))
        self.elo_ratings[loser] = int(loser_elo + k * (0 - expected_loser))
