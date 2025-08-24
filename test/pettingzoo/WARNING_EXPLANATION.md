# PettingZoo pygame Warning Explanation

## The Warnings

When running `./runtests.sh --marl`, you'll see these warnings:

```
.venv/lib/python3.11/site-packages/pygame/pkgdata.py:25
  DeprecationWarning: pkg_resources is deprecated as an API
  
.venv/lib/python3.11/site-packages/pkg_resources/__init__.py:2871
  DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google')`
```

## Root Cause

The import chain is:
1. `test/pettingzoo/pistonball.py` imports `pettingzoo.butterfly.pistonball_v6`
2. PettingZoo's pistonball environment requires `pygame` for rendering
3. pygame v2.5.2 (current version) uses the deprecated `pkg_resources` API
4. pkg_resources is deprecated in favor of `importlib.resources` (PEP 420)

## Why We Can't Fix This

This is a **dependency chain issue** that requires fixes upstream:

1. **pygame** needs to migrate from `pkg_resources` to `importlib.resources`
   - This is tracked in pygame's issue tracker
   - pygame is a large, established project with many legacy considerations
   
2. **PettingZoo** depends on pygame for rendering certain environments
   - Even if we don't use rendering, the import happens at module level
   - PettingZoo can't remove pygame dependency without breaking visualization

## Workarounds Considered and Rejected

1. **Suppressing warnings**: While possible, this hides potentially important issues
2. **Removing pistonball tests**: We already removed the unstable tests, but the module is still imported
3. **Lazy importing**: Would require restructuring PettingZoo's code (external project)
4. **Using different environment**: Would reduce test coverage of PettingZoo integration

## Current Status

- The warnings are **harmless** - they don't affect functionality
- They come from **external dependencies** we don't control
- pygame and pkg_resources both work fine despite the deprecation
- The warnings will disappear when pygame updates (tracked upstream)

## Recommendation

**Leave the warnings visible** because:
1. They're from legitimate deprecations that should be tracked
2. They don't break any functionality
3. Suppressing them might hide other important warnings
4. They serve as a reminder to check for pygame updates

## Future Resolution

The warnings will be resolved when:
- pygame releases a version using `importlib.resources` instead of `pkg_resources`
- OR when Python eventually removes `pkg_resources` (forcing the update)

## For Contributors

If these warnings bother you during development:
- Use `./runtests.sh --fast` for most development (warnings suppressed)
- The warnings only appear with `--marl`, `--all`, or `--slow`
- Focus on warnings from our code, not dependencies