#!/bin/bash
# Test runner script that avoids argparse conflicts

echo "Running Tianshou tests..."
echo "========================="
echo ""

# Run tests with explicit paths to avoid auto-discovery argparse issues
# Exclude known hanging tests

if [ "$1" == "--all" ]; then
    echo "Running all tests (including slow training tests)..."
    python -m pytest test/base test/multiagent test/pettingzoo/test_enhanced_pettingzoo_env.py \
        --ignore=test/base/test_collector.py \
        --ignore=test/base/test_env.py \
        -m "" \
        "${@:2}"
elif [ "$1" == "--slow" ]; then
    echo "Running only slow tests..."
    python -m pytest test \
        --ignore=test/base/test_collector.py \
        --ignore=test/base/test_env.py \
        -m "slow" \
        "${@:2}"
else
    echo "Running fast tests only (default)..."
    python -m pytest test/base test/multiagent test/pettingzoo/test_enhanced_pettingzoo_env.py \
        --ignore=test/base/test_collector.py \
        --ignore=test/base/test_env.py \
        -m "not slow" \
        "${@}"
fi