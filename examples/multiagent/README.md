# Multi-Agent Reinforcement Learning (MARL) Examples

This directory contains comprehensive examples demonstrating Tianshou's multi-agent reinforcement learning capabilities.

## Overview

Tianshou provides a rich set of MARL features including:

- **Enhanced PettingZoo Integration**: Seamless support for both AEC and Parallel PettingZoo environments
- **Flexible Policy Management**: Support for independent learning, parameter sharing, and custom policy configurations
- **CTDE Algorithms**: Centralized Training, Decentralized Execution with QMIX and MADDPG implementations
- **Agent Communication**: Built-in communication framework for agent coordination
- **Training Coordination**: Multiple training modes (simultaneous, sequential, self-play, league play)

## Examples

### 1. Comprehensive MARL Demo (`marl_comprehensive_demo.py`)

A complete example showcasing the main MARL algorithms and policy management modes.

**Features:**
- Independent learning vs parameter sharing vs QMIX
- Support for classic PettingZoo environments (Tic-Tac-Toe, Connect Four)
- Comprehensive training setup with logging and checkpointing

**Usage:**
```bash
# Independent learning on Tic-Tac-Toe
python marl_comprehensive_demo.py --algorithm independent --environment tic_tac_toe

# Parameter sharing on Connect Four
python marl_comprehensive_demo.py --algorithm shared --environment connect_four

# QMIX algorithm
python marl_comprehensive_demo.py --algorithm qmix --environment tic_tac_toe --epoch 100
```

### 2. Communication Demo (`communication_demo.py`)

Demonstrates agent-to-agent communication using Tianshou's communication framework.

**Features:**
- Message encoding and decoding between agents
- Attention-based message aggregation
- Comparative training with and without communication
- Cooperative environment (PistonBall) where communication provides benefits

**Usage:**
```bash
# Train without communication
python communication_demo.py

# Train with communication enabled
python communication_demo.py --with-communication

# Adjust message dimension and other parameters
python communication_demo.py --with-communication --message-dim 64 --epoch 50
```

## Quick Start

1. **Install dependencies:**
```bash
pip install pettingzoo[classic]
```

2. **Run a simple example:**
```bash
cd examples/multiagent
python marl_comprehensive_demo.py --algorithm shared --epoch 10
```

3. **Compare communication vs no communication:**
```bash
# First run without communication
python communication_demo.py --epoch 20

# Then run with communication
python communication_demo.py --with-communication --epoch 20
```

## Environment Requirements

The examples use PettingZoo environments. Install the required environment groups:

```bash
# For classic games (Tic-Tac-Toe, Connect Four)
pip install pettingzoo[classic]

# For butterfly environments (PistonBall)
pip install pettingzoo[butterfly]

# For all environments
pip install pettingzoo[all]
```

## Algorithm Comparison

| Algorithm | Description | Use Case | Communication |
|-----------|-------------|----------|---------------|
| Independent | Each agent learns independently | Simple scenarios, baseline | No |
| Shared | All agents share parameters | Homogeneous agents, fast training | No |
| QMIX | CTDE with value mixing | Complex coordination, non-stationary | No |
| Communication | Message passing between agents | Partial observability, coordination | Yes |

## Key Components Used

- `EnhancedPettingZooEnv`: Enhanced PettingZoo wrapper with auto-detection
- `FlexibleMultiAgentPolicyManager`: Flexible policy management with multiple modes
- `QMIXPolicy`: QMIX implementation for CTDE
- `CommunicatingPolicy`: Policy with communication capabilities
- `SimultaneousTrainer`: Coordinated training of multiple agents

## Performance Tips

1. **Start with parameter sharing** for homogeneous agents - it's often faster and more stable
2. **Use independent learning** when agents have different roles or capabilities
3. **Try QMIX** for environments requiring complex coordination
4. **Enable communication** in partially observable environments where agents can benefit from sharing information
5. **Adjust exploration parameters** (eps_train, eps_test) based on environment complexity

## Extending the Examples

These examples provide a foundation for your MARL experiments. You can:

- Add new environments by following the PettingZoo wrapper pattern
- Implement custom communication topologies
- Create custom training coordinators for specific scenarios
- Add new CTDE algorithms using the provided framework

For more advanced usage, see the test files in `test/multiagent/` for additional examples and edge cases.