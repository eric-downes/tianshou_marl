# Development Workflow & Branch Strategy

This repository uses a tiered branch strategy optimized for fast development iteration while maintaining production quality.

## Branch Structure

### 🚀 `dev` - Fast Development Branch
**Purpose**: Rapid feature development and experimentation  
**CI Duration**: < 60 seconds  
**CI Checks**:
- Ultra-minimal linting (Pyflakes F rules only - syntax errors, undefined variables)
- Ignores unused imports/variables for speed
- Fast unit tests (< 30 seconds)
- 5-minute timeout to force fast completion

**Use for**:
- Daily feature development
- Quick bug fixes
- Experimentation
- When you need fast feedback loops

### 🔍 `staging` - Integration Branch
**Purpose**: Pre-release integration and validation  
**CI Duration**: 5-10 minutes  
**CI Checks**:
- Full linting (all ruff rules)
- Type checking (mypy)
- MARL tests
- Core tests
- Linux-only (no multi-platform overhead)

**Use for**:
- Integration testing
- Release candidates
- Feature validation before production

### 🏭 `master` - Production Branch
**Purpose**: Stable releases  
**CI Duration**: 15-30 minutes  
**CI Checks**:
- Everything in staging +
- GPU tests
- Multi-platform (Windows, macOS)
- Documentation builds
- Full test suite
- Performance benchmarks

**Use for**:
- Production releases
- Tagged versions
- When maximum confidence is required

## Development Workflow

### Day-to-Day Development
```bash
# Create feature branch from dev
git checkout dev
git pull origin dev
git checkout -b feature/my-feature

# Work on feature with fast CI feedback
git add .
git commit -m "Add feature"
git push -u origin feature/my-feature

# Create PR targeting dev branch
gh pr create --base dev --title "Add my feature"
```

### Weekly Integration
```bash
# Merge dev into staging for integration testing
git checkout staging
git pull origin staging
git merge dev
git push origin staging

# Or create PR: dev → staging
gh pr create --base staging --head dev --title "Weekly integration"
```

### Release Process
```bash
# Promote staging to master for release
git checkout master
git pull origin master
git merge staging
git tag v1.2.3
git push origin master --tags
```

## CI Performance Targets

| Branch | Target Duration | Actual Purpose |
|--------|----------------|----------------|
| `dev` | < 60 seconds | Catch critical syntax errors |
| `staging` | < 10 minutes | Full quality validation |
| `master` | < 30 minutes | Production-ready verification |

## Linting Strategy

### Dev Branch (Ultra-permissive)
- **Only F rules**: Syntax errors, undefined variables, import issues
- **Ignores**: F401 (unused imports), F841 (unused variables)
- **Philosophy**: Let developers focus on logic, catch critical errors only

### Staging/Master (Comprehensive)
- **All ruff rules**: Style, conventions, best practices
- **Philosophy**: Enforce production code quality

This ensures no conflicts - dev catches what breaks the code, staging/master enforces quality.

## Getting Started

1. **For new features**: Branch from `dev`, PR to `dev`
2. **For releases**: Merge `dev` → `staging` → `master`
3. **For hotfixes**: Branch from `master`, PR to `master`, then merge back to `staging` and `dev`

## Benefits

✅ **Fast iteration**: < 60s feedback on dev branch  
✅ **Quality assurance**: Full checks before production  
✅ **Conflict-free**: Permissive dev rules don't conflict with strict staging rules  
✅ **Scalable**: Team can work fast while maintaining standards  
✅ **Flexible**: Different quality gates for different use cases  