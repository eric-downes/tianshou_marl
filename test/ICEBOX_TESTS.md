# Icebox Tests

These tests are currently disabled due to various issues. They should be fixed and re-enabled when possible.

## Status: 🧊 Icebox
These are long-term issues not actively being worked on.

---

## 1. test_piston_ball
**File:** `test/pettingzoo/test_pistonball.py`  
**Reason:** Performance bound was never tested, no point in running this for now  
**Current State:**
- The test trains a pistonball agent but has no assertion to verify performance
- The assertion line is commented out: `# assert result.best_reward >= args.win_rate`

**Work Needed:**
1. Determine appropriate performance bounds for pistonball environment
2. Validate that the training achieves consistent results
3. Uncomment and update the assertion with validated performance threshold
4. Remove the @pytest.mark.skip decorator

---

## 2. test_piston_ball_continuous
**File:** `test/pettingzoo/test_pistonball_continuous.py`  
**Reason:** Runtime too long and unstable result  
**Current State:**
- Test takes excessive time to run
- Results are not consistent between runs
- The assertion is commented out: `# assert result.best_reward >= 30.0`

**Work Needed:**
1. Optimize training parameters to reduce runtime
2. Investigate source of instability (random seeds, environment dynamics, etc.)
3. Establish stable performance bounds
4. Consider marking as @pytest.mark.slow instead of skipping entirely
5. Remove the @pytest.mark.skip decorator once stable

---

## Notes for Contributors

When working on these tests:
1. Run tests in isolation first to establish baseline behavior
2. Document any changes to expected performance bounds
3. Ensure tests are deterministic (use fixed random seeds)
4. Consider adding @pytest.mark.slow for long-running tests instead of skipping
5. Update this document when tests are re-enabled