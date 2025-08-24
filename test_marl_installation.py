#!/usr/bin/env python3
"""Test script to verify MARL installation and features are working."""

import sys


def test_import(module_path: str, feature_name: str) -> tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            exec(f"from {parts[0]} import {parts[1]}")
        else:
            exec(f"import {module_path}")
        return True, f"✅ {feature_name}"
    except ImportError as e:
        return False, f"❌ {feature_name}: {e}"
    except Exception as e:
        return False, f"❌ {feature_name}: Unexpected error: {e}"


def test_marl_features() -> tuple[bool, str]:
    """Test MARL features work correctly."""
    try:
        from tianshou.env import EnhancedPettingZooEnv

        # Create mock environment to test without PettingZoo
        class MockEnv:
            def __init__(self):
                self.possible_agents = ["agent_0", "agent_1"]
                self.observation_spaces = {"agent_0": None, "agent_1": None}
                self.action_spaces = {"agent_0": None, "agent_1": None}

        # Test EnhancedPettingZooEnv detects parallel env
        mock_env = MockEnv()
        env = EnhancedPettingZooEnv(mock_env, mode="auto")
        assert env.mode == "parallel", "Failed to detect parallel environment"

        return True, "✅ MARL features working"
    except Exception as e:
        return False, f"❌ MARL features test failed: {e}"


def main():
    """Run all installation tests."""
    print("=" * 60)
    print("Tianshou MARL Installation Test")
    print("=" * 60)
    print()

    # Core dependencies
    print("Testing Core Dependencies:")
    print("-" * 30)

    tests = [
        ("tianshou", "Tianshou"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("gymnasium", "Gymnasium"),
        ("pytest", "Pytest"),
    ]

    results = []
    for module, name in tests:
        success, msg = test_import(module, name)
        print(msg)
        results.append(success)

    # MARL-specific dependencies
    print("\nTesting MARL Dependencies:")
    print("-" * 30)

    marl_tests = [
        ("pettingzoo", "PettingZoo"),
        ("tianshou.env.EnhancedPettingZooEnv", "Enhanced PettingZoo Environment"),
        (
            "tianshou.algorithm.multiagent.FlexibleMultiAgentPolicyManager",
            "Flexible Policy Manager",
        ),
    ]

    for module, name in marl_tests:
        success, msg = test_import(module, name)
        print(msg)
        results.append(success)

    # Functional test
    print("\nTesting MARL Functionality:")
    print("-" * 30)
    success, msg = test_marl_features()
    print(msg)
    results.append(success)

    # Version info
    print("\nVersion Information:")
    print("-" * 30)
    try:
        import tianshou

        print(f"Tianshou version: {tianshou.__version__}")
    except:
        pass

    try:
        import pettingzoo

        print(f"PettingZoo version: {pettingzoo.__version__}")
    except:
        print("PettingZoo not installed (optional but recommended for MARL)")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
    except:
        pass

    # Summary
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)

    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("\nYour installation is ready for MARL experiments!")
        print("\nNext steps:")
        print("1. Run the test suite: ./runtests.sh")
        print("2. Try the examples in test/multiagent/")
        print("3. Check MARL_IMPLEMENTATION_STATUS.md for feature details")
        return 0
    else:
        print(f"⚠️  Some tests failed ({passed}/{total} passed)")
        print("\nPlease check the installation guide: INSTALLATION_MARL.md")
        print("\nCommon fixes:")
        print("1. Install PettingZoo: pip install 'pettingzoo[all]>=1.24.0'")
        print("2. Install in editable mode: pip install -e .")
        print("3. Check you're in the right virtual environment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
