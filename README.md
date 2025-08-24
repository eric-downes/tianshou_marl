# Tianshou MARL - Multi-Agent Reinforcement Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/eric-downes/tianshou_marl/workflows/Tests/badge.svg)](https://github.com/eric-downes/tianshou_marl/actions)

A comprehensive Multi-Agent Reinforcement Learning (MARL) extension for Tianshou, providing production-ready implementations of state-of-the-art MARL algorithms with flexible policy management and training coordination.

## 🚀 Quick Start

### Prerequisites

- **Python 3.11 or higher** (required)
- Poetry or uv for dependency management (recommended)

### Installation with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Python 3.11 if needed
uv python install 3.11

# Clone and install
git clone https://github.com/eric-downes/tianshou_marl.git
cd tianshou_marl
uv sync --extra eval
```

### Installation with Poetry

```bash
# Ensure Python 3.11+ is available
python --version  # Should show 3.11.x or higher

# Clone and install
git clone https://github.com/eric-downes/tianshou_marl.git
cd tianshou_marl
poetry install --with dev --extras "eval"
```

### Installation with pip

```bash
git clone https://github.com/eric-downes/tianshou_marl.git
cd tianshou_marl
pip install -e ".[eval]"
```

### Verify Installation

```bash
python test_marl_installation.py
```

You should see all tests passing with green checkmarks ✅.

## 🎯 What's Included

### Core MARL Features

- **🤖 Enhanced Multi-Agent Environments**
  - `EnhancedPettingZooEnv`: Full PettingZoo compatibility with parallel support
  - Automatic AEC ↔ Parallel environment detection
  - Dict observation space handling

- **🧠 Flexible Policy Management**
  - Parameter sharing (all agents share one policy)
  - Independent policies (each agent has its own)
  - Grouped policies (team-based parameter sharing)
  - Custom policy mappings

- **⚡ Advanced Training Modes**
  - Simultaneous training (all agents learn together)
  - Sequential training (round-robin learning)
  - Self-play training (agent vs. past versions)
  - League play (population-based training)

- **🎛️ State-of-the-Art Algorithms**
  - CTDE (Centralized Training, Decentralized Execution)
  - Multi-agent communication
  - Global state construction
  - Custom policy architectures

## 📖 Usage Examples

### Basic Multi-Agent Setup

```python
import torch.nn as nn
from tianshou.env import EnhancedPettingZooEnv
from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager
from pettingzoo.mpe import simple_spread_v3

# Create environment
env = simple_spread_v3.parallel_env()
wrapped_env = EnhancedPettingZooEnv(env)

# Create shared policy (parameter sharing)
class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, obs):
        return self.net(obs)

shared_policy = SimplePolicy()

# Setup policy manager with parameter sharing
manager = FlexibleMultiAgentPolicyManager(
    policies=shared_policy,
    env=env,
    mode="shared"  # All agents share the same policy
)

# Use the manager like any Tianshou policy
obs, info = wrapped_env.reset()
actions = manager(obs)
```

### CTDE (Centralized Training, Decentralized Execution)

```python
from tianshou.algorithm.multiagent import CTDEPolicy

# Local actor (uses only local observations)
actor = nn.Sequential(
    nn.Linear(local_obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim)
)

# Centralized critic (uses global state)  
critic = nn.Sequential(
    nn.Linear(global_state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

# Create CTDE policy
policy = CTDEPolicy(
    actor=actor,
    critic=critic,
    optim_actor=torch.optim.Adam(actor.parameters()),
    optim_critic=torch.optim.Adam(critic.parameters()),
    observation_space=env.observation_space,
    action_space=env.action_space
)
```

### Training Coordination

```python
from tianshou.algorithm.multiagent import SimultaneousTrainer

# Create trainer
trainer = SimultaneousTrainer(policy_manager=manager)

# Training loop
for epoch in range(1000):
    # Collect experience
    obs, info = env.reset()
    episode_data = []
    
    while not done:
        actions = manager(obs)
        obs_next, rewards, done, info = env.step(actions)
        episode_data.append((obs, actions, rewards, obs_next, done))
        obs = obs_next
    
    # Train all agents
    batch = format_batch(episode_data)  # Your data formatting
    losses = trainer.train_step(batch)
    print(f"Epoch {epoch}: {losses}")
```

## 🏗️ Development Workflow

This project uses a tiered branch strategy for efficient development:

### Branch Structure

- **`master`** (Production) - Stable releases with full CI/CD validation
- **`staging`** (Integration) - Pre-release testing with comprehensive checks  
- **`dev`** (Development) - Fast development with 43-second CI feedback

### For Contributors

```bash
# Development work (fast feedback)
git checkout dev
git pull origin dev
git checkout -b feature/my-feature

# Make changes...
git commit -m "Add feature"
git push origin feature/my-feature

# Create PR targeting dev branch
gh pr create --base dev --title "Add my feature"
```

See [`DEVELOPMENT_WORKFLOW.md`](DEVELOPMENT_WORKFLOW.md) for complete details.

## 🧪 Testing

### Quick Test Suite

```bash
# Install test dependencies
poetry install --with dev

# Run fast tests (< 30 seconds)
pytest test/multiagent/test_marl_integration_fast.py -v

# Run all MARL tests
pytest test/multiagent/ -v

# Run with coverage
pytest test/multiagent/ --cov=tianshou.algorithm.multiagent
```

### Full Test Suite

```bash
# Run all tests (takes several minutes)
pytest test/ -v

# Or use the provided script
./runtests.sh
```

## 📚 Documentation & Examples

- **[DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)** - Branch strategy and development process
- **[MARL_IMPLEMENTATION_STATUS.md](MARL_IMPLEMENTATION_STATUS.md)** - Feature implementation status
- **`test/multiagent/`** - Comprehensive test examples showing usage patterns
- **`tianshou/algorithm/multiagent/`** - Source code with detailed docstrings

## 🤝 Contributing

We welcome contributions! Please:

1. Use the `dev` branch for fast iteration
2. Follow the existing code style and patterns  
3. Add tests for new features
4. Update documentation as needed

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋 Support & Questions

- **Issues**: [GitHub Issues](https://github.com/eric-downes/tianshou_marl/issues)
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the docstrings and test files for detailed examples

---

Built with ❤️ for the MARL research community. Star ⭐ if this helps your research!