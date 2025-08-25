#!/usr/bin/env python3
"""
Demonstration of the simplified Tianshou MARL API.

This example shows how the new simplified API makes it easier to get started
with multi-agent reinforcement learning.
"""

from pettingzoo.classic import tictactoe_v3

# NEW: Simplified imports - no more deep nesting!
from tianshou_marl import (
    AutoPolicy,
    PolicyManager,
    Collector,
    ReplayBuffer,
    MARLEnv,
)
from tianshou.env import DummyVectorEnv


def main():
    """Demonstrate the simplified API."""
    
    print("=== Tianshou MARL Simplified API Demo ===\n")
    
    # Create environment
    env = tictactoe_v3.env()
    env = MARLEnv(env)
    
    print(f"Environment: Tic-Tac-Toe")
    print(f"Agents: {env.agents}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}\n")
    
    # 1. AUTOMATIC POLICY SELECTION
    print("1. Automatic Policy Selection (AutoPolicy)")
    print("-" * 40)
    
    # OLD WAY: Manual setup with lots of boilerplate
    # - Detect observation/action spaces manually
    # - Calculate state shapes for Dict spaces
    # - Create networks manually
    # - Initialize policies with correct parameters
    
    # NEW WAY: AutoPolicy handles everything!
    auto_policy = AutoPolicy.from_env(
        env,
        config={
            "learning_rate": 1e-3,
            "hidden_sizes": [64, 64],
            "epsilon": 0.1,
        }
    )
    print(f"✓ AutoPolicy created: {type(auto_policy).__name__}")
    print(f"  Mode: {auto_policy.mode}")
    print(f"  Number of policies: {len(auto_policy.policies)}\n")
    
    # 2. PARAMETER SHARING
    print("2. Parameter Sharing (shared mode)")
    print("-" * 40)
    
    # NEW: Easy parameter sharing - all agents use the same policy
    shared_policy = AutoPolicy.from_env(
        env,
        mode="shared",  # All agents share the same policy
        config={"learning_rate": 1e-3}
    )
    print(f"✓ Shared policy created")
    print(f"  Mode: {shared_policy.mode}")
    print(f"  Unique policies: {len(set(id(p) for p in shared_policy.policies.values()))}\n")
    
    # 3. FACTORY METHODS
    print("3. Factory Methods (PolicyManager.from_env)")
    print("-" * 40)
    
    # NEW: Create specific algorithm with factory method
    dqn_manager = PolicyManager.from_env(
        env,
        algorithm="DQN",  # Specify algorithm
        mode="independent",
        config={
            "learning_rate": 1e-3,
            "hidden_sizes": [128, 128],
            "epsilon": 0.1,
        }
    )
    print(f"✓ DQN PolicyManager created")
    print(f"  Algorithm: DQN")
    print(f"  Mode: {dqn_manager.mode}\n")
    
    # 4. TRAINING SETUP (simplified)
    print("4. Simplified Training Setup")
    print("-" * 40)
    
    # Create vectorized environments
    train_envs = DummyVectorEnv([lambda: MARLEnv(tictactoe_v3.env()) for _ in range(4)])
    test_envs = DummyVectorEnv([lambda: MARLEnv(tictactoe_v3.env()) for _ in range(2)])
    
    # Create replay buffer
    buffer = ReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    
    # Create collector
    train_collector = Collector(
        auto_policy,
        train_envs,
        buffer,
        exploration_noise=True,
    )
    
    test_collector = Collector(
        auto_policy,
        test_envs,
        exploration_noise=False,
    )
    
    print("✓ Training components created:")
    print(f"  Train environments: {len(train_envs)}")
    print(f"  Test environments: {len(test_envs)}")
    print(f"  Buffer size: {buffer.maxsize}")
    
    print("\n" + "=" * 50)
    print("Demo complete! The simplified API provides:")
    print("- ✓ Cleaner imports (tianshou_marl.*)")
    print("- ✓ Automatic policy selection (AutoPolicy)")
    print("- ✓ Easy parameter sharing (mode='shared')")
    print("- ✓ Factory methods (PolicyManager.from_env)")
    print("- ✓ Consistent configuration interface")
    print("- ✓ Sensible defaults")
    print("\nCompare this to the complex setup in marl_comprehensive_demo.py!")


if __name__ == "__main__":
    main()