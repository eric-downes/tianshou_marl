# Installation Guide for Tianshou with MARL Enhancements

This guide provides step-by-step instructions for setting up Tianshou with the new Multi-Agent Reinforcement Learning (MARL) enhancements.

## Prerequisites

- Python >= 3.11
- Git
- uv (recommended) or pip

## Installation Methods

### Method 1: Using UV (Recommended)

UV is a fast Python package installer that we recommend for this project.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/thu-ml/tianshou.git tianshou_marl
cd tianshou_marl

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base package in editable mode
uv pip install -e .

# Install development dependencies for testing
uv pip install pytest pytest-cov black ruff mypy nbqa
uv pip install networkx pygame pymunk scipy

# Install PettingZoo for multi-agent environments (required for MARL features)
uv pip install "pettingzoo[all]>=1.24.0"

# Optional: Install additional RL environment dependencies
uv pip install gymnasium[all]  # For additional Gymnasium environments
uv pip install supersuit  # For environment wrappers
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/thu-ml/tianshou.git tianshou_marl
cd tianshou_marl

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install base package in editable mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black ruff mypy nbqa
pip install networkx pygame pymunk scipy

# Install PettingZoo for multi-agent environments
pip install "pettingzoo[all]>=1.24.0"

# Optional: Install additional dependencies
pip install gymnasium[all]
pip install supersuit
```

### Method 3: Using Poetry (Original Method)

```bash
# Install poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/thu-ml/tianshou.git tianshou_marl
cd tianshou_marl

# Install with development dependencies
poetry install --with dev

# Install PettingZoo separately (not in pyproject.toml yet)
poetry run pip install "pettingzoo[all]>=1.24.0"
```

## Required Dependencies for MARL Features

The following packages are required for the MARL enhancements to work:

### Core Dependencies (automatically installed)
- `torch >= 2.0.0`
- `numpy >= 1.0`
- `gymnasium >= 0.28.0`
- `tensorboard >= 2.5.0`
- `tqdm`
- `pandas >= 2.0.0`
- `matplotlib >= 3.0.0`

### MARL-Specific Dependencies (must install separately)
- `pettingzoo >= 1.24.0` - Multi-agent environment API
- `supersuit` (optional) - Environment wrappers for PettingZoo

### Testing Dependencies
- `pytest >= 7.0.0` - Test framework
- `pytest-cov` - Coverage reporting
- `networkx` - Used in some tests
- `scipy` - Scientific computing (used in tests)

## Verifying Installation

After installation, verify everything is working:

```bash
# Check Tianshou is installed
python -c "import tianshou; print(tianshou.__version__)"

# Check MARL features are available
python -c "from tianshou.env import EnhancedPettingZooEnv; print('Enhanced env: OK')"
python -c "from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager; print('Flexible policy: OK')"

# Check PettingZoo is installed
python -c "import pettingzoo; print(f'PettingZoo {pettingzoo.__version__}')"

# Run tests (use the provided script to avoid argparse conflicts)
chmod +x runtests.sh
./runtests.sh -q  # Quick test of fast tests

# Or run specific MARL tests
python -m pytest test/multiagent test/pettingzoo/test_enhanced_pettingzoo_env.py -v
```

## Running Tests

Due to argparse conflicts in some test files, use the provided test runner:

```bash
# Run fast tests only (default)
./runtests.sh

# Run all tests including slow training tests
./runtests.sh --all

# Run with pytest options
./runtests.sh -v  # Verbose
./runtests.sh -q  # Quiet
```

## Example: Testing MARL Features

```python
# test_marl_setup.py
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from tianshou.env import EnhancedPettingZooEnv
from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager

# Create parallel environment
env = simple_spread_v3.parallel_env()
wrapped_env = EnhancedPettingZooEnv(env, mode="parallel")

print("Environment created successfully!")
print(f"Agents: {wrapped_env.agents}")
print(f"Mode: {wrapped_env.mode}")

# Test reset and step
obs, info = wrapped_env.reset()
print(f"Observation keys: {obs.keys()}")

# Random actions for all agents
actions = {agent: wrapped_env.action_space.sample() for agent in wrapped_env.agents}
obs, rewards, term, trunc, info = wrapped_env.step(actions)
print(f"Step completed! Rewards: {rewards}")
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pettingzoo'`
**Solution**: Install PettingZoo: `pip install "pettingzoo[all]>=1.24.0"`

### Issue: `pytest` command shows many errors
**Solution**: Use the provided `./runtests.sh` script instead of direct pytest

### Issue: Tests hanging
**Solution**: Some tests in `test_collector.py` and `test_env.py` are known to hang. These are excluded in the runner script.

### Issue: Import errors for MARL features
**Solution**: Ensure you're in the correct virtual environment and have installed the package in editable mode (`pip install -e .`)

## Next Steps

1. Check out the example notebooks in `docs/02_notebooks/`
2. Read the MARL implementation documentation in `MARL_IMPLEMENTATION_STATUS.md`
3. Try the examples in `examples/` directory
4. Review the test implementations in `test/multiagent/` and `test/pettingzoo/`

## Support

For issues specific to MARL features, please check:
- `MARL_IMPLEMENTATION_STATUS.md` - Current implementation status
- `tianshou_marl_requirements.md` - Full requirements specification
- `TEST_REPORT.md` - Test status and known issues