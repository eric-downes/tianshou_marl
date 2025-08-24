#!/usr/bin/env python3
"""Test runner for Tianshou MARL - Python alternative to runtests.sh"""

import os
import sys
import warnings
import argparse

# Suppress pygame and pkg_resources warnings before any imports
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:pygame.pkgdata,ignore::DeprecationWarning:pkg_resources"

import pytest


def main():
    parser = argparse.ArgumentParser(
        description="Run Tianshou MARL tests",
        epilog="Additional pytest arguments can be passed after --"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "fast", "slow", "marl"],
        default="fast",
        help="Test mode to run (default: fast)"
    )
    
    # Parse known args to allow pytest args to pass through
    args, pytest_args = parser.parse_known_args()
    
    print("Running Tianshou tests...")
    print("=" * 50)
    
    # Build pytest command
    pytest_cmd = ["test"]
    
    if args.mode == "all":
        print("Running all tests (including slow training tests)...")
        print("This may take several minutes...")
        pytest_cmd.extend(["-m", ""])
    elif args.mode == "slow":
        print("Running only slow tests...")
        pytest_cmd.extend(["-m", "slow"])
    elif args.mode == "fast":
        print("Running fast tests only...")
        pytest_cmd.extend(["-m", "not slow"])
    elif args.mode == "marl":
        print("Running MARL tests only...")
        pytest_cmd = ["test/multiagent", "test/pettingzoo"]
    
    # Add any additional pytest arguments
    pytest_cmd.extend(pytest_args)
    
    # Run pytest
    sys.exit(pytest.main(pytest_cmd))


if __name__ == "__main__":
    main()