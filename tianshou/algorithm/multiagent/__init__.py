"""Multi-agent reinforcement learning algorithms."""

from tianshou.algorithm.multiagent.marl import (
    MultiAgentPolicy,
    MultiAgentOffPolicyAlgorithm,
    MultiAgentOnPolicyAlgorithm,
    MARLDispatcher,
    MapTrainingStats,
)
from tianshou.algorithm.multiagent.flexible_policy import FlexibleMultiAgentPolicyManager

__all__ = [
    "MultiAgentPolicy",
    "MultiAgentOffPolicyAlgorithm",
    "MultiAgentOnPolicyAlgorithm",
    "MARLDispatcher",
    "MapTrainingStats",
    "FlexibleMultiAgentPolicyManager",
]