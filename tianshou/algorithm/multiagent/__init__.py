"""Multi-agent reinforcement learning algorithms."""

from tianshou.algorithm.multiagent.communication import (
    CommunicatingPolicy,
    CommunicationChannel,
    MessageDecoder,
    MessageEncoder,
    MultiAgentCommunicationWrapper,
)
from tianshou.algorithm.multiagent.ctde import (
    CentralizedCritic,
    CTDEPolicy,
    DecentralizedActor,
    GlobalStateConstructor,
    MADDPGPolicy,
    QMIXMixer,
    QMIXPolicy,
)
from tianshou.algorithm.multiagent.flexible_policy import (
    FlexibleMultiAgentPolicyManager,
)
from tianshou.algorithm.multiagent.marl import (
    MapTrainingStats,
    MARLDispatcher,
    MultiAgentOffPolicyAlgorithm,
    MultiAgentOnPolicyAlgorithm,
    MultiAgentPolicy,
)
from tianshou.algorithm.multiagent.training_coordinator import (
    LeaguePlayTrainer,
    MATrainer,
    SelfPlayTrainer,
    SequentialTrainer,
    SimultaneousTrainer,
)

__all__ = [
    "CTDEPolicy",
    "CentralizedCritic",
    "CommunicatingPolicy",
    "CommunicationChannel",
    "DecentralizedActor",
    "FlexibleMultiAgentPolicyManager",
    "GlobalStateConstructor",
    "LeaguePlayTrainer",
    "MADDPGPolicy",
    "MARLDispatcher",
    "MATrainer",
    "MapTrainingStats",
    "MessageDecoder",
    "MessageEncoder",
    "MultiAgentCommunicationWrapper",
    "MultiAgentOffPolicyAlgorithm",
    "MultiAgentOnPolicyAlgorithm",
    "MultiAgentPolicy",
    "QMIXMixer",
    "QMIXPolicy",
    "SelfPlayTrainer",
    "SequentialTrainer",
    "SimultaneousTrainer",
]
