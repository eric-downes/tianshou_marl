"""Multi-agent reinforcement learning algorithms."""

from tianshou.algorithm.multiagent.marl import (
    MultiAgentPolicy,
    MultiAgentOffPolicyAlgorithm,
    MultiAgentOnPolicyAlgorithm,
    MARLDispatcher,
    MapTrainingStats,
)
from tianshou.algorithm.multiagent.flexible_policy import FlexibleMultiAgentPolicyManager
from tianshou.algorithm.multiagent.training_coordinator import (
    MATrainer,
    SimultaneousTrainer,
    SequentialTrainer,
    SelfPlayTrainer,
    LeaguePlayTrainer,
)

__all__ = [
    "MultiAgentPolicy",
    "MultiAgentOffPolicyAlgorithm",
    "MultiAgentOnPolicyAlgorithm",
    "MARLDispatcher",
    "MapTrainingStats",
    "FlexibleMultiAgentPolicyManager",
    "MATrainer",
    "SimultaneousTrainer",
    "SequentialTrainer",
    "SelfPlayTrainer",
    "LeaguePlayTrainer",
]