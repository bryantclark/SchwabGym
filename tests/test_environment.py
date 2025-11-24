"""
Environment Tests
=================

Unit tests for the generic SchwabTradingEnv and ZScoreNormalizer.

Author: Bryant Clark
"""

from unittest.mock import MagicMock, Mock

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from schwabgym import SchwabTradingEnv
from schwabgym.environment import ZScoreNormalizer


class TestZScoreNormalizer:
    """Test ZScoreNormalizer functionality."""

    def test_normalization(self):
        """Test that normalization produces zero mean and unit variance."""
        norm = ZScoreNormalizer(shape=(1,))
        data = np.random.normal(loc=10.0, scale=2.0, size=(1000, 1))

        normalized_data = []
        for x in data:
            normalized_data.append(norm.normalize(x))

        normalized_data = np.array(normalized_data)

        # Check last 100 samples (after convergence)
        recent = normalized_data[-100:]
        assert abs(np.mean(recent)) < 0.5
        assert abs(np.std(recent) - 1.0) < 0.5

    def test_clipping(self):
        """Test that output is clipped."""
        norm = ZScoreNormalizer(shape=(1,), clip_range=2.0)

        # Feed extreme value
        norm.mean = np.array([0.0])
        norm.var = np.array([1.0])
        norm.count = 100

        res = norm.normalize(np.array([100.0]))
        assert abs(res[0]) <= 2.0

    def test_reset(self):
        """Test that statistics can be reset."""
        norm = ZScoreNormalizer(shape=(1,))
        norm.mean = np.array([100.0])
        norm.reset()
        assert norm.mean[0] == 0.0
        assert norm.count == 1e-4


class TestSchwabTradingEnv:
    """Test the generic SchwabTradingEnv shell."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked MockClient."""
        client = MagicMock()
        client.current_step = 0
        # Default behaviors
        client.advance_time.return_value = True
        client._calculate_equity.return_value = 25000.0
        return client

    @pytest.fixture
    def strategies(self):
        """Create mock strategy functions."""
        obs_fn = Mock(return_value=np.zeros(8, dtype=np.float32))
        reward_fn = Mock(return_value=1.5)
        action_fn = Mock()
        term_fn = Mock(return_value=False)
        return obs_fn, reward_fn, action_fn, term_fn

    @pytest.fixture
    def env(self, mock_client, strategies):
        """Create the generic environment with mocked components."""
        obs_fn, reward_fn, action_fn, term_fn = strategies

        return SchwabTradingEnv(
            client=mock_client,
            observation_fn=obs_fn,
            reward_fn=reward_fn,
            action_fn=action_fn,
            termination_fn=term_fn,
            observation_space=spaces.Box(
                low=-10, high=10, shape=(8,), dtype=np.float32
            ),
            action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        )

    def test_initialization(self, env):
        """Test that spaces are correctly assigned."""
        assert env.observation_space.shape == (8,)
        assert env.action_space.shape == (2,)

    def test_reset(self, env, mock_client, strategies):
        """Test that reset triggers client and strategy resets."""
        obs_fn, _, _, _ = strategies

        obs, info = env.reset()

        # Check client reset
        mock_client.reset.assert_called_once()

        # Check observation function called
        obs_fn.assert_called_with(mock_client)

        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_orchestration(self, env, mock_client, strategies):
        """Test that step() calls all injected functions in correct order."""
        obs_fn, reward_fn, action_fn, term_fn = strategies
        action = np.array([1.0, 0.5])

        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)

        # 1. Check Action Execution
        action_fn.assert_called_once_with(mock_client, action)

        # 2. Check Simulator Advance
        mock_client.advance_time.assert_called_once()

        # 3. Check Observation & Reward Calculation
        obs_fn.assert_called()
        reward_fn.assert_called_with(mock_client)

        # 4. Check Termination Check
        term_fn.assert_called_with(mock_client)

        # Verify return values match mocks
        assert reward == 1.5
        assert terminated is False
        assert info["nav"] == 25000.0

    def test_termination_from_simulator(self, env, mock_client):
        """Test termination when simulator runs out of data."""
        # Simulator returns False for advance_time
        mock_client.advance_time.return_value = False

        _, _, terminated, _, _ = env.step(np.array([0, 0]))

        assert terminated is True

    def test_termination_from_strategy(self, env, mock_client, strategies):
        """Test termination from injected function (e.g. margin call)."""
        _, _, _, term_fn = strategies
        term_fn.return_value = True  # Strategy says terminate

        _, _, terminated, _, _ = env.step(np.array([0, 0]))

        assert terminated is True

    def test_stateful_obs_reset(self, mock_client):
        """Test that reset() calls reset() on a stateful observation builder."""

        class StatefulObs:
            def __init__(self):
                self.reset_called = False

            def reset(self):
                self.reset_called = True

            def __call__(self, client):
                return np.zeros(8)

        stateful_fn = StatefulObs()

        env = SchwabTradingEnv(
            client=mock_client,
            observation_fn=stateful_fn,  # Pass instance directly
            reward_fn=lambda c: 0,
            action_fn=lambda c, a: None,
            observation_space=spaces.Box(0, 1, (8,)),
            action_space=spaces.Box(0, 1, (2,)),
        )

        env.reset()
        assert stateful_fn.reset_called is True
