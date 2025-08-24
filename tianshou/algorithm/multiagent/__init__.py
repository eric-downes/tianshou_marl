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
from tianshou.algorithm.multiagent.ctde import (
    CTDEPolicy,
    GlobalStateConstructor,
    DecentralizedActor,
    CentralizedCritic,
    QMIXPolicy,
    QMIXMixer,
    MADDPGPolicy,
)
from tianshou.algorithm.multiagent.communication import (
    CommunicationChannel,
    MessageEncoder,
    MessageDecoder,
    CommunicatingPolicy,
    MultiAgentCommunicationWrapper,
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
    "CTDEPolicy",
    "GlobalStateConstructor",
    "DecentralizedActor",
    "CentralizedCritic",
    "QMIXPolicy",
    "QMIXMixer",
    "MADDPGPolicy",
    "CommunicationChannel",
    "MessageEncoder",
    "MessageDecoder",
    "CommunicatingPolicy",
    "MultiAgentCommunicationWrapper",
]