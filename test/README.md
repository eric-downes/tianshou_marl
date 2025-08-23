# Tianshou Test Suite

## Quick Start

By default, pytest runs only fast tests (skipping slow training tests):

```bash
# Run fast tests only (default)
pytest

# Run ALL tests (including slow training tests)
pytest -m ""

# Run only slow tests
pytest -m slow

# Run specific test directory
pytest test/base

# Run with coverage
pytest --cov=tianshou
```

## Test Organization

Tests are organized into categories:

- **test/base/** - Fast unit tests for core components (batches, buffers, collectors, etc.)
- **test/continuous/** - RL algorithm tests for continuous action spaces (PPO, SAC, DDPG, etc.) - SLOW
- **test/discrete/** - RL algorithm tests for discrete action spaces (DQN, A2C, etc.) - SLOW
- **test/modelbased/** - Model-based RL tests (PSRL, ICM) - SLOW
- **test/offline/** - Offline RL tests (BCQ, CQL, etc.) - SLOW
- **test/pettingzoo/** - Multi-agent environment tests - SLOW
- **test/highlevel/** - High-level API tests - SLOW

## Test Markers

Tests are marked with pytest markers:
- `@pytest.mark.slow` - Training/optimization tests that take significant time

The default configuration in `pytest.ini` skips slow tests for faster development iteration.

## CI/CD

For continuous integration and releases, run all tests:
```bash
pytest -m ""
```

This ensures all algorithms are tested before deployment.