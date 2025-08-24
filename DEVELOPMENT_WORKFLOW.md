# Development Workflow & Branch Strategy

This repository uses a tiered branch strategy optimized for **production stability** while enabling fast development iteration.

## 🎯 Production-First Philosophy

- **`master`** is the **default branch** - stable and production-ready
- **`dev`** provides cutting-edge features with fast CI for active development  
- **`staging`** bridges the gap with comprehensive integration testing

Users wanting stability clone `master`. Developers wanting the latest features use `dev`.

## Branch Structure

### 🏭 `master` - Production Branch (Default)
**Purpose**: Stable, production-ready releases  
**Default Branch**: ✅ This is what users get when they clone  
**CI Duration**: 15-30 minutes  
**CI Checks**:
- Full linting (all ruff rules)
- Type checking (mypy)
- Complete test suite
- GPU tests
- Multi-platform (Windows, macOS, Linux)
- Documentation builds
- Performance benchmarks

**Use for**:
- Production releases and tagged versions
- Stable code that users can depend on
- Default clone target for end users
- Research that needs reliability

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
- Integration testing before production
- Release candidates
- Feature validation
- Quality gate before master

### 🚀 `dev` - Fast Development Branch
**Purpose**: Cutting-edge development and experimentation  
**CI Duration**: **43 seconds** ⚡  
**CI Checks**:
- Ultra-minimal linting (Pyflakes F rules only - syntax errors, undefined variables)
- Ignores unused imports/variables for speed
- Basic Python syntax validation
- 5-minute timeout to force fast completion

**Use for**:
- Daily feature development
- Quick bug fixes
- Experimentation
- When you need fast feedback loops
- MARL research and active development

## Development Workflow

### 🔬 For Researchers/Developers (Fast Iteration)
```bash
# Get cutting-edge features
git clone https://github.com/eric-downes/tianshou_marl.git
cd tianshou_marl
git checkout dev

# Create feature branch
git checkout -b feature/my-research

# Work with 43-second CI feedback
git add .
git commit -m "Add new MARL algorithm"
git push -u origin feature/my-research

# Create PR targeting dev branch
gh pr create --base dev --title "Add new MARL algorithm"
```

### 👥 For End Users (Stable Usage)
```bash
# Get stable, tested code (default)
git clone https://github.com/eric-downes/tianshou_marl.git
cd tianshou_marl
# Already on master - ready to use!

# Install and run
python test_marl_installation.py
```

### 🚀 Release Process

#### Weekly Integration (dev → staging)
```bash
git checkout staging
git pull origin staging
git merge dev
git push origin staging

# Or create PR: dev → staging
gh pr create --base staging --head dev --title "Weekly integration"
```

#### Production Release (staging → master)
```bash
git checkout master
git pull origin master
git merge staging
git tag v1.2.3
git push origin master --tags
```

## CI Performance Targets

| Branch | Target Duration | Purpose | Quality Level |
|--------|----------------|---------|---------------|
| `dev` | **< 60 seconds** | Catch critical syntax errors | Minimal - "Does it parse?" |
| `staging` | < 10 minutes | Full quality validation | High - "Is it good?" |
| `master` | < 30 minutes | Production verification | Maximum - "Is it bulletproof?" |

## Linting Strategy

### Dev Branch (Ultra-permissive)
- **Only F rules**: Syntax errors, undefined variables, import issues
- **Ignores**: F401 (unused imports), F841 (unused variables)  
- **Philosophy**: Let developers focus on logic, catch critical errors only

### Staging/Master (Comprehensive)
- **All ruff rules**: Style, conventions, best practices
- **Philosophy**: Enforce production code quality

This ensures no conflicts - dev catches what breaks the code, staging/master enforces quality.

## Branch Selection Guide

| You want... | Use branch... | Because... |
|-------------|--------------|------------|
| **Stable code for production** | `master` | Full testing, proven reliability |
| **Latest features for research** | `dev` | Cutting-edge, fast iteration |
| **Integration testing** | `staging` | Quality validation without full CI overhead |
| **To contribute a bug fix** | `master` → fix → `master` | Direct to production |
| **To contribute a feature** | `dev` → feature → `dev` | Fast development cycle |

## Getting Started

### I want to USE the library
```bash
git clone https://github.com/eric-downes/tianshou_marl.git
# You're on master - stable and ready!
```

### I want to DEVELOP features  
```bash
git clone https://github.com/eric-downes/tianshou_marl.git
git checkout dev
# Fast 43-second CI feedback
```

### I want to INTEGRATE and TEST
```bash
git checkout staging  
# Full quality checks in 10 minutes
```

## Benefits

✅ **Production-first**: Master is stable by default  
✅ **Fast development**: 43s CI feedback on dev  
✅ **Quality assurance**: Comprehensive checks before production  
✅ **Flexible**: Choose your quality/speed tradeoff  
✅ **Clear separation**: Stability vs cutting-edge clearly separated  
✅ **User-friendly**: Clone and go with stable master  

---

*This workflow ensures production stability while enabling rapid MARL research and development.*