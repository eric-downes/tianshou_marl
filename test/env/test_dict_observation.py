"""Tests for Dict observation space support in MARL environments."""


import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from tianshou.algorithm.algorithm_base import Policy
from tianshou.data import Batch
from tianshou.env.dict_observation import (
    DictObservationWrapper,
    DictToTensorPreprocessor,
)


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, observation_space=None, action_space=None, preprocessing=None):
        # Provide default action space if not given
        if action_space is None:
            action_space = spaces.Discrete(2)
        if observation_space is None:
            observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        super().__init__(observation_space=observation_space, action_space=action_space)
        self.preprocessing = preprocessing
        self.forward_count = 0

    def forward(self, batch: Batch, state=None, **kwargs):
        """Simple forward that returns random actions."""
        self.forward_count += 1
        # Handle tensor observations
        if isinstance(batch.obs, torch.Tensor):
            batch_size = batch.obs.shape[0]
        elif hasattr(batch.obs, "__len__"):
            batch_size = len(batch.obs)
        else:
            batch_size = 1
        return Batch(act=np.random.randint(0, 2, size=batch_size), state=state)

    def learn(self, batch: Batch, **kwargs):
        """Mock learn method."""
        return {"loss": 0.0}


class MockDictEnv:
    """Mock environment with Dict observation space."""

    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                "velocity": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
                "discrete_state": spaces.Discrete(5),
                "multi_discrete": spaces.MultiDiscrete([3, 4, 5]),
            }
        )
        self.action_space = spaces.Discrete(4)

    def reset(self):
        """Reset and return dict observation."""
        return {
            "position": np.array([0.5, -0.3, 0.1], dtype=np.float32),
            "velocity": np.array([1.0, -2.0, 0.5], dtype=np.float32),
            "discrete_state": np.array(2),  # Make it an array for consistency
            "multi_discrete": np.array([1, 2, 3]),
        }

    def step(self, action):
        """Step and return dict observation."""
        obs = self.reset()
        return obs, 0.0, False, False, {}


@pytest.mark.slow
class TestDictObservationWrapper:
    """Test suite for DictObservationWrapper."""

    def test_wrapper_initialization(self):
        """Test wrapper can be initialized with dict observation space."""
        env = MockDictEnv()
        policy = MockPolicy(observation_space=env.observation_space, action_space=env.action_space)

        wrapper = DictObservationWrapper(policy, env.observation_space)

        assert wrapper.policy == policy
        assert wrapper.observation_space == env.observation_space
        assert wrapper.preprocessor is not None

    def test_auto_preprocessor_creation(self):
        """Test automatic preprocessor creation when not provided."""
        env = MockDictEnv()
        policy = MockPolicy()

        wrapper = DictObservationWrapper(policy, env.observation_space)

        # Should create a DictToTensorPreprocessor automatically
        assert isinstance(wrapper.preprocessor, DictToTensorPreprocessor)

    def test_custom_preprocessor(self):
        """Test using custom preprocessor."""
        env = MockDictEnv()

        class CustomPreprocessor(nn.Module):
            def forward(self, obs_dict):
                return torch.zeros(1, 10)

        custom_prep = CustomPreprocessor()
        policy = MockPolicy(preprocessing=custom_prep)

        wrapper = DictObservationWrapper(policy, env.observation_space)

        assert wrapper.preprocessor == custom_prep

    def test_forward_with_dict_obs(self):
        """Test forward pass with dict observations."""
        env = MockDictEnv()
        policy = MockPolicy()
        wrapper = DictObservationWrapper(policy, env.observation_space)

        # Create batch with dict observations
        obs = env.reset()
        batch = Batch(obs=obs)

        # Forward should process dict obs
        result = wrapper.forward(batch)

        assert policy.forward_count == 1
        assert "act" in result

    def test_forward_with_batched_dict_obs(self):
        """Test forward with batched dict observations."""
        env = MockDictEnv()
        policy = MockPolicy()
        wrapper = DictObservationWrapper(policy, env.observation_space)

        # Create batch with multiple dict observations
        obs1 = env.reset()
        obs2 = env.reset()
        obs2["position"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        batch_obs = {key: np.stack([obs1[key], obs2[key]]) for key in obs1.keys()}
        batch = Batch(obs=batch_obs)

        result = wrapper.forward(batch)

        assert policy.forward_count == 1
        assert "act" in result
        assert len(result.act) == 2  # Two actions for two observations

    def test_preserves_non_dict_obs(self):
        """Test that non-dict observations pass through unchanged."""
        policy = MockPolicy()
        env = MockDictEnv()
        wrapper = DictObservationWrapper(policy, env.observation_space)

        # Use regular array observation
        regular_obs = np.array([1, 2, 3, 4], dtype=np.float32)
        batch = Batch(obs=regular_obs)

        # Should handle gracefully
        result = wrapper.forward(batch)
        assert "act" in result


@pytest.mark.slow
class TestDictToTensorPreprocessor:
    """Test suite for DictToTensorPreprocessor."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization with dict space."""
        env = MockDictEnv()
        preprocessor = DictToTensorPreprocessor(env.observation_space)

        assert preprocessor.keys == sorted(env.observation_space.spaces.keys())
        assert len(preprocessor.extractors) == len(env.observation_space.spaces)

    def test_box_space_extractor(self):
        """Test feature extractor for Box spaces."""
        space = spaces.Dict(
            {"box_feature": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)}
        )
        preprocessor = DictToTensorPreprocessor(space)

        # Check Box space gets linear layer
        assert "box_feature" in preprocessor.extractors
        assert isinstance(preprocessor.extractors["box_feature"], nn.Linear)
        assert preprocessor.extractors["box_feature"].in_features == 4

    def test_discrete_space_extractor(self):
        """Test feature extractor for Discrete spaces."""
        space = spaces.Dict({"discrete_feature": spaces.Discrete(10)})
        preprocessor = DictToTensorPreprocessor(space)

        # Check Discrete space gets embedding
        assert "discrete_feature" in preprocessor.extractors
        assert isinstance(preprocessor.extractors["discrete_feature"], nn.Embedding)
        assert preprocessor.extractors["discrete_feature"].num_embeddings == 10

    def test_multi_discrete_space_extractor(self):
        """Test feature extractor for MultiDiscrete spaces."""
        space = spaces.Dict({"multi_discrete": spaces.MultiDiscrete([3, 4, 5])})
        preprocessor = DictToTensorPreprocessor(space)

        # Check MultiDiscrete gets list of embeddings
        assert "multi_discrete" in preprocessor.extractors
        assert isinstance(preprocessor.extractors["multi_discrete"], nn.ModuleList)
        assert len(preprocessor.extractors["multi_discrete"]) == 3

    def test_forward_single_observation(self):
        """Test forward pass with single dict observation."""
        env = MockDictEnv()
        preprocessor = DictToTensorPreprocessor(env.observation_space)

        obs = env.reset()
        # Convert to torch tensors
        obs_tensors = {
            "position": torch.tensor(obs["position"]).unsqueeze(0),
            "velocity": torch.tensor(obs["velocity"]).unsqueeze(0),
            "discrete_state": torch.tensor(
                [
                    (
                        obs["discrete_state"].item()
                        if hasattr(obs["discrete_state"], "item")
                        else obs["discrete_state"]
                    )
                ]
            ),
            "multi_discrete": torch.tensor([obs["multi_discrete"]]),
        }

        output = preprocessor(obs_tensors)

        # Should output concatenated tensor
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 2  # [batch, features]
        assert output.shape[0] == 1  # batch size 1

    def test_forward_batched_observations(self):
        """Test forward pass with batched dict observations."""
        env = MockDictEnv()
        preprocessor = DictToTensorPreprocessor(env.observation_space)

        batch_size = 4
        obs_tensors = {
            "position": torch.randn(batch_size, 3),
            "velocity": torch.randn(batch_size, 3),
            "discrete_state": torch.randint(0, 5, (batch_size,)),
            "multi_discrete": torch.randint(0, 3, (batch_size, 3)),
        }

        output = preprocessor(obs_tensors)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size

    def test_output_dimension_calculation(self):
        """Test that output dimension is correctly calculated."""
        env = MockDictEnv()
        preprocessor = DictToTensorPreprocessor(env.observation_space)

        # Calculate expected output dim
        # Box spaces: 64 each (position, velocity)
        # Discrete: 32
        # MultiDiscrete: 16 * 3 = 48
        expected_dim = 64 + 64 + 32 + 48

        assert preprocessor.output_dim == expected_dim

    def test_unsupported_space_error(self):
        """Test error handling for unsupported space types."""
        # Create space with unsupported type
        space = spaces.Dict(
            {"unsupported": spaces.Tuple([spaces.Discrete(2), spaces.Box(0, 1, (2,))])}
        )

        with pytest.raises(NotImplementedError):
            DictToTensorPreprocessor(space)


@pytest.mark.slow
class TestIntegration:
    """Integration tests with real environments and policies."""

    def test_with_pettingzoo_dict_obs(self):
        """Test with PettingZoo environment that has dict observations."""
        # This would test with actual PettingZoo envs if they have dict obs
        # For now, we'll use our mock
        env = MockDictEnv()
        policy = MockPolicy()
        wrapper = DictObservationWrapper(policy, env.observation_space)

        # Simulate environment interaction
        obs = env.reset()
        batch = Batch(obs=obs)

        for _ in range(10):
            result = wrapper.forward(batch)
            action = result.act
            obs, reward, done, trunc, info = env.step(action)
            batch = Batch(obs=obs)

        assert policy.forward_count == 10

    def test_memory_efficiency(self):
        """Test that preprocessing doesn't create memory leaks."""
        import gc
        import tracemalloc

        tracemalloc.start()

        env = MockDictEnv()
        policy = MockPolicy()
        wrapper = DictObservationWrapper(policy, env.observation_space)

        # Run many iterations
        for _ in range(100):
            obs = env.reset()
            batch = Batch(obs=obs)
            result = wrapper.forward(batch)

        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (< 10MB for this simple test)
        assert peak < 10 * 1024 * 1024  # 10MB
