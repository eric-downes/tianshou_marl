# CI/CD Setup for Tianshou MARL

## Overview
This document describes the Continuous Integration and Continuous Deployment (CI/CD) setup for the Tianshou MARL implementation.

## GitHub Actions Workflows

### 1. Main Test Suite (`pytest.yml`)
- **Trigger**: PRs to master, pushes to master
- **Tests**: Runs default fast tests only
- **Coverage**: Reports to Codecov

### 2. All Tests Suite (`pytest_all.yml`)
- **Trigger**: PRs to master, pushes to master
- **Tests**: Runs ALL tests including slow tests
- **Purpose**: Ensures comprehensive testing before merging
- **Coverage**: Full coverage report to Codecov

### 3. MARL-Specific Tests (`marl_tests.yml`)
- **Trigger**: PRs/pushes that modify MARL files
- **Tests**: 
  - Fast MARL tests (45 tests, <2 seconds)
  - Slow CTDE tests (21 tests)
- **Coverage**: MARL-specific coverage report
- **Smart Triggers**: Only runs when MARL files change

## Test Categories

### Fast Tests (Default)
- Run by default with `pytest`
- Complete in seconds
- Cover critical functionality
- Suitable for rapid development iteration

### Slow Tests
- Marked with `@pytest.mark.slow`
- Include comprehensive integration tests
- Training simulations
- CTDE algorithms testing

### MARL Tests
- **Fast**: `test/multiagent/test_marl_fast_comprehensive.py` (45 tests)
- **Slow**: `test/multiagent/test_ctde.py` (21 tests)

## Local Testing

### Using the Test Runner Script
```bash
# Run fast tests (default)
./scripts/run_tests.sh

# Run MARL fast tests
./scripts/run_tests.sh marl-fast

# Run MARL slow tests
./scripts/run_tests.sh marl-slow

# Run all MARL tests
./scripts/run_tests.sh marl-all

# Run ALL tests including slow
./scripts/run_tests.sh all

# Run with coverage report
./scripts/run_tests.sh coverage

# Add verbose output
./scripts/run_tests.sh marl-all -v
```

### Using pytest directly
```bash
# Fast tests only (default)
pytest

# All tests including slow
pytest -m ""

# Only slow tests
pytest -m slow

# MARL tests
pytest test/multiagent/ -m "slow or not slow"

# With coverage
pytest --cov=tianshou --cov-report=html
```

## GitHub Pull Request Process

### What Happens on PR to Master
1. **Main Test Suite** runs fast tests
2. **All Tests Suite** runs comprehensive tests (including slow)
3. **MARL Tests** run if MARL files are modified
4. **Code Coverage** is reported to Codecov
5. **Status Checks** must pass before merge

### Test Requirements for Merge
- ✅ All fast tests passing
- ✅ All slow tests passing (via `pytest_all.yml`)
- ✅ MARL tests passing (if MARL code modified)
- ✅ No significant coverage regression

## Workflow Features

### Smart Triggers
- MARL workflow only runs when MARL files change
- Reduces unnecessary CI runs
- Faster feedback for developers

### Debugging Support
- All workflows support tmate debugging
- Can be triggered manually with debug mode
- Useful for troubleshooting CI failures

### Caching
- Virtual environment cached between runs
- Speeds up CI execution
- Cache keys based on `poetry.lock`

## Best Practices

### For Developers
1. **Before Pushing**: Run `./scripts/run_tests.sh marl-fast` for quick validation
2. **Before PR**: Run `./scripts/run_tests.sh marl-all` to ensure all tests pass
3. **Mark Slow Tests**: Use `@pytest.mark.slow` for time-consuming tests
4. **Write Fast Tests**: Prefer fast unit tests over slow integration tests when possible

### For Reviewers
1. Check CI status on all workflows
2. Review coverage reports for new code
3. Ensure slow tests are appropriate
4. Verify backward compatibility

## Monitoring and Maintenance

### Coverage Tracking
- Coverage reports sent to Codecov
- Track coverage trends over time
- Aim for >80% coverage on new code

### Performance Monitoring
- Test durations reported in CI logs
- Monitor for test suite slowdown
- Consider moving tests to slow category if >5 seconds

### Workflow Updates
- Review and update workflows quarterly
- Keep dependencies up to date
- Monitor for new GitHub Actions features

## Troubleshooting

### Common Issues

#### Tests Pass Locally but Fail in CI
- Check Python version (CI uses 3.11)
- Verify all dependencies in `pyproject.toml`
- Check for environment-specific issues

#### Slow Test Timeout
- Increase timeout in workflow if needed
- Consider breaking up large tests
- Use `@pytest.mark.slow` appropriately

#### Cache Issues
- Clear cache by updating cache key
- Check `poetry.lock` is committed
- Verify virtual environment setup

### Getting Help
1. Check CI logs for detailed error messages
2. Use tmate debugging for interactive troubleshooting
3. Review this documentation
4. Ask in PR comments for assistance

## Future Improvements

### Planned Enhancements
- [ ] Parallel test execution for faster CI
- [ ] Separate workflows for different test categories
- [ ] Performance regression testing
- [ ] Automated benchmark comparisons
- [ ] Integration with external MARL benchmarks

### Optimization Opportunities
- Matrix testing across Python versions
- GPU testing for applicable algorithms
- Distributed testing for scalability tests
- Automated release workflow

## Summary

The CI/CD setup ensures:
1. **Quality**: All code is thoroughly tested
2. **Speed**: Fast tests provide quick feedback
3. **Coverage**: Comprehensive testing including slow tests
4. **Efficiency**: Smart triggers reduce unnecessary runs
5. **Reliability**: Consistent testing environment

For questions or improvements, please open an issue or submit a PR.