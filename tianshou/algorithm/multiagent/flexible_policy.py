"""Flexible multi-agent policy management with parameter sharing support."""

from collections.abc import Callable
from typing import Any, Literal

from overrides import override

from tianshou.algorithm.algorithm_base import Policy
from tianshou.algorithm.multiagent.marl import MultiAgentPolicy
from tianshou.data import Batch


class FlexibleMultiAgentPolicyManager(MultiAgentPolicy):
    """Enhanced policy manager with flexible agent-policy mapping.

    This manager extends MultiAgentPolicy to support various policy configurations:
    - Independent: Each agent has its own policy
    - Shared: All agents share the same policy (parameter sharing)
    - Grouped: Agents in the same group share a policy
    - Custom: User-defined policy mapping

    Example usage for parameter sharing:
    ::
        # All agents share the same policy
        shared_policy = PPOPolicy(...)
        manager = FlexibleMultiAgentPolicyManager(
            policies=shared_policy,
            env=env,
            mode="shared"
        )

    Example usage for grouped agents:
    ::
        # Team-based configuration
        agent_groups = {
            "team_red": ["agent_0", "agent_1"],
            "team_blue": ["agent_2", "agent_3"]
        }
        policies = {
            "team_red": PPOPolicy(...),
            "team_blue": PPOPolicy(...)
        }
        manager = FlexibleMultiAgentPolicyManager(
            policies=policies,
            env=env,
            mode="grouped",
            agent_groups=agent_groups
        )
    """

    def __init__(
        self,
        policies: Policy | list[Policy] | dict[str, Policy],
        env: Any,  # PettingZooEnv type
        mode: Literal["independent", "shared", "grouped", "custom"] = "independent",
        agent_groups: dict[str, list[str]] | None = None,
        policy_mapping_fn: Callable[[str], str] | None = None,
        **kwargs: Any,
    ):
        """Initialize flexible multi-agent policy manager.

        Args:
            policies: Single policy (shared), list (one per agent), or dict (named policies)
            env: PettingZoo environment
            mode: Policy assignment mode
                - "independent": Each agent has its own policy
                - "shared": All agents share the same policy
                - "grouped": Agents in same group share policy
                - "custom": Use policy_mapping_fn
            agent_groups: Dict mapping group names to agent IDs (for "grouped" mode)
            policy_mapping_fn: Custom function mapping agent_id to policy_id
            **kwargs: Additional arguments passed to parent class

        """
        self.mode = mode
        self.agent_groups = agent_groups or {}
        self.policy_mapping_fn = policy_mapping_fn
        self.env = env
        self.agents = env.agents  # Store agents list

        # Validate configuration
        self._validate_config(policies)

        # Build policy mapping based on mode
        self.policy_map = self._build_policy_map(policies, env.agents)

        # Initialize parent class with the policy map
        super().__init__(policies=self.policy_map)  # type: ignore[arg-type]

        # Store original policies for reference
        self._original_policies = policies

        # Create policy_mapping attribute for compatibility
        self.policy_mapping = self.policy_map

        # Create policies dict with unique policies for compatibility
        if self.mode == "shared":
            self.policies = {"shared": next(iter(self.policy_map.values()))}
        elif self.mode == "grouped":
            self.policies = {
                group: self.policy_map[self.agent_groups[group][0]]
                for group in self.agent_groups.keys()
            }
        else:
            # For independent and custom, create dict with unique policies
            unique_policies: dict[str, Policy] = {}
            for agent, policy in self.policy_map.items():
                if policy not in unique_policies.values():
                    unique_policies[agent] = policy
            self.policies = unique_policies

    def _validate_config(self, policies):
        """Validate configuration based on mode."""
        if self.mode == "independent":
            if isinstance(policies, Policy):
                raise ValueError(
                    "Independent mode requires list or dict of policies, got single policy"
                )

        elif self.mode == "grouped":
            if not self.agent_groups:
                raise ValueError("Grouped mode requires agent_groups to be specified")
            if isinstance(policies, Policy):
                raise ValueError(
                    "Grouped mode requires dict of policies mapped to group names"
                )

        elif self.mode == "custom":
            if not self.policy_mapping_fn:
                raise ValueError(
                    "Custom mode requires policy_mapping_fn to be specified"
                )

    def _build_policy_map(self, policies, agents) -> dict[str, Policy]:
        """Build agent-to-policy mapping based on configuration mode."""
        if self.mode == "shared":
            # All agents use the same policy
            if isinstance(policies, Policy):
                shared_policy = policies
            elif isinstance(policies, list):
                shared_policy = policies[0]
            elif isinstance(policies, dict):
                shared_policy = next(iter(policies.values()))
            else:
                raise ValueError(f"Invalid policies type: {type(policies)}")

            return dict.fromkeys(agents, shared_policy)

        elif self.mode == "grouped":
            # Agents in same group share policy
            policy_map = {}
            for group_name, group_agents in self.agent_groups.items():
                if isinstance(policies, dict):
                    if group_name not in policies:
                        raise ValueError(f"No policy found for group {group_name}")
                    group_policy = policies[group_name]
                else:
                    raise ValueError("Grouped mode requires dict of policies")

                for agent in group_agents:
                    if agent in agents:
                        policy_map[agent] = group_policy

            # Check all agents are assigned
            unassigned = set(agents) - set(policy_map.keys())
            if unassigned:
                raise ValueError(f"Agents {unassigned} are not assigned to any group")

            return policy_map

        elif self.mode == "custom":
            # Use custom mapping function
            if not isinstance(policies, dict):
                raise ValueError("Custom mode requires dict of policies")

            policy_map = {}
            for agent in agents:
                policy_id = self.policy_mapping_fn(agent)
                if policy_id not in policies:
                    raise ValueError(f"Policy {policy_id} not found for agent {agent}")
                policy_map[agent] = policies[policy_id]

            return policy_map

        else:  # "independent"
            # Each agent has its own policy
            if isinstance(policies, list):
                if len(policies) != len(agents):
                    raise ValueError(
                        f"Number of policies ({len(policies)}) must match number of agents ({len(agents)})"
                    )
                return dict(zip(agents, policies, strict=False))
            elif isinstance(policies, dict):
                # Check all agents have policies
                missing = set(agents) - set(policies.keys())
                if missing:
                    raise ValueError(f"Missing policies for agents: {missing}")
                return policies
            else:
                raise ValueError("Independent mode requires list or dict of policies")

    @override
    def forward(
        self, batch: Batch, state: dict | Batch | None = None, **kwargs: Any
    ) -> Batch:
        """Forward pass through policies.

        Overrides parent to handle shared policies more efficiently.
        """
        if self.mode == "shared" and hasattr(batch.obs, "agent_id"):
            # Optimize for shared policy - batch all observations together
            return self._forward_shared(batch, state, **kwargs)
        else:
            # Use parent implementation for other modes
            return super().forward(batch, state, **kwargs)

    def _forward_shared(
        self, batch: Batch, state: dict | Batch | None = None, **kwargs: Any
    ) -> Batch:
        """Optimized forward for shared policy mode."""
        # Get the shared policy (all agents use the same one)
        shared_policy = next(iter(self.policy_map.values()))

        # Process all observations at once
        if hasattr(batch.obs, "obs"):
            # Extract actual observations if wrapped
            obs_batch = Batch(obs=batch.obs.obs)
        else:
            obs_batch = batch

        # Single forward pass for all agents
        out = shared_policy(batch=obs_batch, state=state, **kwargs)

        # Format output to match expected structure
        holder = Batch()
        holder["act"] = out.act

        # Create per-agent outputs for compatibility
        out_dict = {}
        state_dict = {}
        for agent_id in self.env.agents:
            out_dict[agent_id] = out
            state_dict[agent_id] = out.state if hasattr(out, "state") else Batch()

        holder["out"] = out_dict
        holder["state"] = state_dict

        return holder

    def get_shared_parameters(self) -> bool:
        """Check if policies share parameters."""
        unique_policies = set(id(policy) for policy in self.policy_map.values())
        return len(unique_policies) < len(self.policy_map)

    def get_policy_groups(self) -> dict[str, list[str]]:
        """Get mapping of policies to agents using them."""
        policy_to_agents = {}
        for agent_id, policy in self.policy_map.items():
            policy_key = id(policy)
            if policy_key not in policy_to_agents:
                policy_to_agents[policy_key] = []
            policy_to_agents[policy_key].append(agent_id)

        # Convert to readable format
        groups = {}
        for i, (_, agents) in enumerate(policy_to_agents.items()):
            group_name = f"group_{i}" if len(policy_to_agents) > 1 else "shared"
            groups[group_name] = agents

        return groups
