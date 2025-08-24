#!/bin/bash
# Test runner script for Tianshou MARL
# Note: argparse conflicts have been resolved, so we can now use pytest directly

# Suppress pygame and pkg_resources warnings at import time
export PYTHONWARNINGS="ignore::DeprecationWarning:pygame.pkgdata,ignore::DeprecationWarning:pkg_resources"

echo "Running Tianshou tests..."
echo "========================="
echo ""

# Parse command line arguments
if [ "$1" == "--all" ]; then
    echo "Running all tests (including slow training tests)..."
    echo "This may take several minutes..."
    echo "Warnings are shown to help identify potential issues."
    python -m pytest test \
        -m "" \
        "${@:2}"
elif [ "$1" == "--slow" ]; then
    echo "Running only slow tests..."
    echo "Warnings are shown to help identify potential issues."
    python -m pytest test \
        -m "slow" \
        "${@:2}"
elif [ "$1" == "--fast" ]; then
    echo "Running fast tests only..."
    # Check if user wants to see warnings
    if [[ " ${@:2} " =~ " -W " ]]; then
        echo "Warnings enabled as requested."
        python -m pytest test \
            -m "not slow" \
            "${@:2}"
    else
        echo "Warnings suppressed for cleaner output. Use -W default to show them."
        python -m pytest test \
            -m "not slow" \
            --disable-warnings \
            "${@:2}"
    fi
elif [ "$1" == "--marl" ]; then
    echo "Running MARL tests only..."
    echo "Warnings are shown to help identify potential issues."
    python -m pytest test/multiagent test/pettingzoo \
        "${@:2}"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./runtests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all    Run all tests (fast and slow)"
    echo "  --fast   Run only fast tests (default)"
    echo "  --slow   Run only slow tests"
    echo "  --marl   Run only MARL-related tests"
    echo "  --help   Show this help message"
    echo ""
    echo "Additional pytest arguments can be passed after the option."
    echo "Example: ./runtests.sh --fast -v -x"
else
    # Default: run fast tests with clean output
    echo "Running fast tests only (default)..."
    echo "Use './runtests.sh --all' to run all tests"
    # Check if user wants to see warnings
    if [[ " $@ " =~ " -W " ]]; then
        echo "Warnings enabled as requested."
        python -m pytest test \
            -m "not slow" \
            "${@}"
    else
        echo "Warnings suppressed for cleaner output. Use -W default to show them."
        python -m pytest test \
            -m "not slow" \
            --disable-warnings \
            "${@}"
    fi
fi