# README Addition for MARL Features

Add this section to the main README.md after the "Installation" section:

---

## 🚀 New: Enhanced Multi-Agent RL Support

This fork includes significant enhancements to Tianshou's Multi-Agent Reinforcement Learning (MARL) capabilities:

### New Features

1. **Enhanced Parallel Environment Support** (`tianshou.env.EnhancedPettingZooEnv`)
   - Full support for both AEC and Parallel PettingZoo environments
   - Auto-detection of environment type
   - Simultaneous action handling for parallel environments
   - Backward compatible with existing `PettingZooEnv`

2. **Flexible Policy Configuration** (`tianshou.algorithm.multiagent.FlexibleMultiAgentPolicyManager`)
   - **Parameter sharing**: All agents share the same policy (memory efficient)
   - **Independent policies**: Each agent has its own policy
   - **Grouped policies**: Agents in the same group share policies
   - **Custom mapping**: User-defined agent-to-policy mapping

### Installation for MARL Features

For detailed installation instructions including all MARL dependencies:
```bash
# See INSTALLATION_MARL.md for complete setup
pip install -e .
pip install "pettingzoo[all]>=1.24.0"
```

### Quick Example

```python
from pettingzoo.mpe import simple_spread_v3
from tianshou.env import EnhancedPettingZooEnv
from tianshou.algorithm.multiagent import FlexibleMultiAgentPolicyManager

# Parallel environment with simultaneous actions
env = simple_spread_v3.parallel_env()
wrapped_env = EnhancedPettingZooEnv(env, mode="parallel")

# Parameter sharing - all agents use same policy
shared_policy = YourPolicy(...)
manager = FlexibleMultiAgentPolicyManager(
    policies=shared_policy,
    env=wrapped_env,
    mode="shared"
)
```

### Documentation

- **Installation Guide**: See [`INSTALLATION_MARL.md`](INSTALLATION_MARL.md) for detailed setup
- **Implementation Status**: See [`MARL_IMPLEMENTATION_STATUS.md`](MARL_IMPLEMENTATION_STATUS.md)
- **Running Tests**: See [`HOW_TO_RUN_TESTS.md`](HOW_TO_RUN_TESTS.md)
- **Full Requirements**: See [`tianshou_marl_requirements.md`](tianshou_marl_requirements.md)

### Testing

```bash
# Use the provided test runner to avoid argparse conflicts
./runtests.sh        # Run fast tests
./runtests.sh --all  # Run all tests including slow training tests

# Or run MARL-specific tests
python -m pytest test/multiagent test/pettingzoo/test_enhanced_pettingzoo_env.py -v
```

---