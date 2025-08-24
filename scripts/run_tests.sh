#!/bin/bash
# Script to run various test suites for Tianshou MARL

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Tianshou MARL Test Runner${NC}"
echo "=========================="

# Parse command line arguments
TEST_TYPE=${1:-"fast"}
VERBOSE=""

if [[ "$2" == "-v" ]] || [[ "$2" == "--verbose" ]]; then
    VERBOSE="-v"
fi

case $TEST_TYPE in
    fast)
        echo -e "${YELLOW}Running fast tests only (default)...${NC}"
        pytest test $VERBOSE --tb=short
        ;;
    
    marl-fast)
        echo -e "${YELLOW}Running MARL fast tests...${NC}"
        pytest test/multiagent/test_marl_fast_comprehensive.py $VERBOSE --tb=short
        ;;
    
    marl-slow)
        echo -e "${YELLOW}Running MARL slow tests (CTDE)...${NC}"
        pytest test/multiagent/test_ctde.py -m slow $VERBOSE --tb=short
        ;;
    
    marl-all)
        echo -e "${YELLOW}Running all MARL tests...${NC}"
        pytest test/multiagent/ -m "slow or not slow" $VERBOSE --tb=short
        ;;
    
    all)
        echo -e "${YELLOW}Running ALL tests including slow...${NC}"
        pytest test -m "" $VERBOSE --tb=short
        ;;
    
    coverage)
        echo -e "${YELLOW}Running all tests with coverage report...${NC}"
        pytest test -m "" --cov=tianshou --cov-report=term-missing --cov-report=html $VERBOSE
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo ""
        echo "Usage: $0 [test-type] [-v|--verbose]"
        echo ""
        echo "Test types:"
        echo "  fast       - Run only fast tests (default)"
        echo "  marl-fast  - Run MARL fast tests only"
        echo "  marl-slow  - Run MARL slow tests (CTDE)"
        echo "  marl-all   - Run all MARL tests"
        echo "  all        - Run all tests including slow"
        echo "  coverage   - Run all tests with coverage report"
        exit 1
        ;;
esac

echo -e "${GREEN}Tests completed!${NC}"