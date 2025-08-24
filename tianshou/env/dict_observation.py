"""Dict observation space support for MARL environments.

This module provides automatic handling of Dict/nested observation spaces,
which are common in complex multi-agent environments where observations
consist of multiple components (e.g., different sensor modalities).
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from tianshou.data import Batch
from tianshou.algorithm.algorithm_base import Policy


class DictObservationWrapper:
    """Automatic handling of Dict observation spaces.

    This wrapper allows policies to work with Dict observation spaces by
    automatically preprocessing them into tensors suitable for neural networks.

    Example usage:
    ::
        # Environment with dict observations
        env = gym.make("SomeEnvWithDictObs-v0")
        policy = PPOPolicy(...)

        # Wrap policy to handle dict observations
        wrapper = DictObservationWrapper(policy, env.observation_space)

        # Use wrapper like regular policy
        obs = env.reset()  # Returns dict
        batch = Batch(obs=obs)
        action = wrapper.forward(batch)
    """

    def __init__(self, policy: Policy, observation_space: spaces.Dict):
        """Initialize Dict observation wrapper.

        Args:
            policy: Base policy to wrap
            observation_space: Dict observation space from environment
        """
        self.policy = policy
        self.observation_space = observation_space
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        """Build preprocessor for Dict observations."""
        if hasattr(self.policy, "preprocessing") and self.policy.preprocessing is not None:
            # User-defined preprocessing
            return self.policy.preprocessing
        else:
            # Auto-build preprocessor
            return DictToTensorPreprocessor(self.observation_space)

    def forward(self, batch: Batch, state: Optional[Any] = None, **kwargs):
        """Process Dict observations before policy forward.

        Args:
            batch: Batch containing observations (may be dict or regular)
            state: Optional RNN state
            **kwargs: Additional arguments for policy

        Returns:
            Policy output batch with actions
        """
        # Check if observations are dict type or Batch with dict-like structure
        if isinstance(batch.obs, (dict, Batch)):
            # Convert dict observations to tensors
            if isinstance(batch.obs, Batch):
                # Batch objects can act like dicts
                obs_dict = dict(batch.obs)
            else:
                obs_dict = batch.obs

            processed_obs = self._preprocess_dict_obs(obs_dict)

            # Create new batch with processed observations
            processed_batch = Batch(obs=processed_obs)

            # Copy other batch attributes if they exist
            for key in batch.keys():
                if key != "obs":
                    processed_batch[key] = batch[key]

            return self.policy.forward(processed_batch, state, **kwargs)
        else:
            # Non-dict observations pass through
            return self.policy.forward(batch, state, **kwargs)

    def _preprocess_dict_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Preprocess dict observations to tensor.

        Args:
            obs: Dict of observations

        Returns:
            Processed tensor observations
        """
        # Check if batched or single observation
        is_batched = False
        for key, value in obs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if len(value.shape) > 0:
                    is_batched = True
                    break

        if not is_batched:
            # Single observation - add batch dimension
            obs_tensors = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.from_numpy(value).unsqueeze(0).float()
                elif isinstance(value, (int, float)):
                    obs_tensors[key] = torch.tensor([value])
                elif isinstance(value, torch.Tensor):
                    obs_tensors[key] = value.unsqueeze(0)
                else:
                    obs_tensors[key] = torch.tensor([value])
        else:
            # Batched observations
            obs_tensors = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.from_numpy(value).float()
                elif isinstance(value, torch.Tensor):
                    obs_tensors[key] = value
                else:
                    obs_tensors[key] = torch.tensor(value)

        # Apply preprocessor
        return self.preprocessor(obs_tensors)

    def learn(self, batch: Batch, **kwargs):
        """Learn from batch with dict observations.

        Args:
            batch: Training batch
            **kwargs: Additional arguments

        Returns:
            Learning statistics
        """
        # Preprocess observations if dict
        if isinstance(batch.obs, dict):
            batch.obs = self._preprocess_dict_obs(batch.obs)

        return self.policy.learn(batch, **kwargs)

    def __getattr__(self, name):
        """Forward other attributes to wrapped policy."""
        return getattr(self.policy, name)


class DictToTensorPreprocessor(nn.Module):
    """Automatic Dict-to-Tensor converter for neural network policies.

    This preprocessor automatically builds feature extractors for each
    component of a Dict observation space and concatenates them into
    a single tensor suitable for neural network input.
    """

    def __init__(self, observation_space: spaces.Dict):
        """Initialize preprocessor for Dict observation space.

        Args:
            observation_space: Dict observation space to process
        """
        super().__init__()
        self.keys = sorted(observation_space.spaces.keys())
        self.spaces = observation_space.spaces

        # Build feature extractors for each component
        self.extractors = nn.ModuleDict()
        self.output_dims = {}

        for key, space in self.spaces.items():
            self.extractors[key], dim = self._build_extractor(key, space)
            self.output_dims[key] = dim

        # Calculate total output dimension
        self.output_dim = sum(self.output_dims.values())

    def _build_extractor(self, key: str, space: spaces.Space) -> tuple[nn.Module, int]:
        """Build feature extractor for space component.

        Args:
            key: Name of the observation component
            space: Gymnasium space for this component

        Returns:
            Tuple of (extractor module, output dimension)
        """
        if isinstance(space, spaces.Box):
            # Linear layer for continuous features
            input_dim = int(np.prod(space.shape))
            output_dim = 64  # Fixed size for now
            return nn.Linear(input_dim, output_dim), output_dim

        elif isinstance(space, spaces.Discrete):
            # Embedding for discrete features
            num_embeddings = int(space.n)
            embedding_dim = 32  # Fixed size for now
            return nn.Embedding(num_embeddings, embedding_dim), embedding_dim

        elif isinstance(space, spaces.MultiDiscrete):
            # List of embeddings for multi-discrete
            embeddings = nn.ModuleList([nn.Embedding(int(n), 16) for n in space.nvec])
            output_dim = 16 * len(space.nvec)
            return embeddings, output_dim

        else:
            raise NotImplementedError(f"Space type {type(space)} not supported")

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert Dict observation to flat tensor.

        Args:
            obs: Dict of observation tensors

        Returns:
            Concatenated feature tensor [batch_size, total_features]
        """
        features = []

        for key in self.keys:
            if key not in obs:
                continue

            obs_component = obs[key]
            extractor = self.extractors[key]

            if isinstance(self.spaces[key], spaces.Box):
                # Flatten if needed
                if len(obs_component.shape) > 2:
                    obs_component = obs_component.view(obs_component.shape[0], -1)
                feature = extractor(obs_component)

            elif isinstance(self.spaces[key], spaces.Discrete):
                # Ensure integer type for embedding
                if obs_component.dtype in [torch.float32, torch.float64]:
                    obs_component = obs_component.long()
                feature = extractor(obs_component)

            elif isinstance(self.spaces[key], spaces.MultiDiscrete):
                # Process each discrete component
                sub_features = []
                for i, embedding in enumerate(extractor):
                    sub_obs = obs_component[..., i]
                    if sub_obs.dtype in [torch.float32, torch.float64]:
                        sub_obs = sub_obs.long()
                    sub_features.append(embedding(sub_obs))
                feature = torch.cat(sub_features, dim=-1)

            else:
                raise NotImplementedError(f"Space type {type(self.spaces[key])} not supported")

            features.append(feature)

        # Concatenate all features
        return torch.cat(features, dim=-1)
