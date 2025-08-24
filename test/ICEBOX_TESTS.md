# Removed Unstable Tests

These tests were removed because they were unstable and unreliable. This document describes what they were testing and proposes alternative testing strategies.

## Why These Tests Were Removed
- Unstable results between runs
- No clear performance bounds established
- Not suitable for CI/CD pipeline
- The `@pytest.mark.slow` marker is for reliable but time-consuming tests, not unstable ones

---

## 1. test_piston_ball
**File:** `test/pettingzoo/test_pistonball.py`  
**Reason:** Performance bound was never tested, no point in running this for now  
**Current State:**
- The test trains a pistonball agent but has no assertion to verify performance
- The assertion line is commented out: `# assert result.best_reward >= args.win_rate`

**What It Was Testing:**
- Training convergence on PistonBall environment
- Multi-agent coordination in cooperative task
- PPO algorithm performance in continuous action space

**Alternative Testing Strategy:**
Instead of full training runs, we should test:
1. **Integration test**: Verify agent can interact with PistonBall environment (1-2 episodes)
2. **Policy update test**: Verify gradients flow and loss decreases over a few updates
3. **Action validity test**: Ensure actions are within valid bounds
4. **Multi-agent coordination test**: Verify agents receive and process observations correctly
5. Create a separate benchmark script (not a test) for performance validation

---

## 2. test_piston_ball_continuous
**File:** `test/pettingzoo/test_pistonball_continuous.py`  
**Reason:** Runtime too long and unstable result  
**Current State:**
- Test takes excessive time to run
- Results are not consistent between runs
- The assertion is commented out: `# assert result.best_reward >= 30.0`

**What It Was Testing:**
- Continuous action space handling in PistonBall
- SAC/TD3 algorithm convergence 
- Long-horizon multi-agent learning

**Alternative Testing Strategy:**
Instead of full training runs, we should test:
1. **Environment wrapper test**: Verify continuous action space is properly handled
2. **Batch collection test**: Ensure proper data collection over a few steps
3. **Replay buffer test**: Verify experience replay works with continuous actions
4. **Actor-critic update test**: Check that both networks update properly
5. Move performance testing to dedicated benchmark suite outside of unit tests

---

## Implementation Plan

### Short-term (Quick Wins)
Create lightweight integration tests that verify:
- Environment can be instantiated and reset
- Agents can take actions and receive observations
- Basic training loop executes without errors (1-2 iterations)

### Medium-term
Develop a test suite that validates components without full training:
- Policy gradient computation
- Value function updates
- Multi-agent batch handling
- Communication between agents (if applicable)

### Long-term
Create a separate benchmarking framework:
- Not part of the test suite
- Runs nightly or on-demand
- Tracks performance metrics over time
- Uses fixed seeds and hyperparameters for reproducibility

## Key Principle
**Unit tests should be fast and deterministic.** Performance benchmarking should be separate from the test suite.