"""Pytest configuration for tianshou tests."""

import pytest


def pytest_configure(config):
    """Configure pytest before running tests."""
    # Prevent argparse conflicts in test modules
    # Many test files have if __name__ == "__main__" blocks with argparse
    # which can interfere with pytest's own argument parsing

    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle special cases."""
    # Skip tests that require special setup if not available
    skip_no_env = pytest.mark.skip(reason="PettingZoo environment not installed")

    for item in items:
        # Skip PettingZoo tests if pettingzoo is not installed
        if "pettingzoo" in item.nodeid:
            try:
                import pettingzoo  # noqa: F401
            except ImportError:
                item.add_marker(skip_no_env)
