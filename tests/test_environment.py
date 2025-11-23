"""
Environment Tests
=================

Unit tests for the SchwabTradingEnv Gymnasium environment.

Author: Bryant Clark
"""

import numpy as np
import pytest
from unittest.mock import patch

from schwabgym import SchwabTradingEnv, load_and_clean_data
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


class TestSchwabTradingEnv:
    """Test SchwabTradingEnv functionality."""

    @pytest.fixture
    def env(self, sample_data):
        """Create environment fixture."""
        return SchwabTradingEnv(sample_data, ticker="TEST")

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.ticker == "TEST"
        assert env.initial_cash == 25000.0
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (8,)

    def test_reset(self, env):
        """Test reset returns valid observation."""
        obs, info = env.reset()
        assert obs.shape == (8,)
        assert isinstance(info, dict)
        assert env.client.current_step == 0

    def test_step_hold(self, env):
        """Test stepping with HOLD action."""
        env.reset()
        # Action: [0.0 (hold), 0.0 (size)]
        action = np.array([0.0, 0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert info["action_taken"] == "HOLD"

    def test_step_buy(self, env):
        """Test stepping with BUY action."""
        env.reset()
        # Action: [1.0 (buy), 0.5 (50% size)]
        action = np.array([1.0, 0.5], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        assert info["action_taken"] == "BUY"
        assert info["shares"] > 0

    def test_step_sell(self, env):
        """Test stepping with SELL action."""
        env.reset()

        # First buy
        env.step(np.array([1.0, 0.5], dtype=np.float32))

        # Then sell
        # Action: [-1.0 (sell), 1.0 (100% size)]
        action = np.array([-1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert info["action_taken"] == "SELL"
        assert info["shares"] == 0  # Should be closed

    def test_step_short(self, env):
        """Test stepping with SHORT action."""
        env.reset()
        # Action: [-1.0 (sell), 0.5 (50% size)]
        action = np.array([-1.0, 0.5], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        assert info["action_taken"] == "SHORT"
        assert info["shares"] < 0

    def test_step_cover(self, env):
        """Test stepping with COVER action."""
        env.reset()

        # First short
        env.step(np.array([-1.0, 0.5], dtype=np.float32))

        # Then cover
        # Action: [1.0 (buy), 1.0 (100% size)]
        action = np.array([1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert info["action_taken"] == "COVER"
        assert info["shares"] == 0

    def test_margin_call_termination(self, env):
        """Test termination on margin call."""
        env.reset()
        # Force account value low
        env.client.cash = 10000.0  # Below 15k threshold

        # Take a step
        _, reward, terminated, _, _ = env.step(np.array([0.0, 0.0]))

        assert terminated is True
        assert reward == -10.0

    def test_render_chart(self, env):
        """Test chart rendering."""
        obs, _ = env.reset()
        
        # Step a few times to generate history
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)
            
        # Test render 'chart' mode
        # We mock plt.show to avoid opening a window during tests
        with patch('matplotlib.pyplot.show'):
            env.render_mode = 'chart'
            env.render()
            
        # Test render 'human' mode
    def test_rsi_flat_prices(self, env):
        """Test RSI calculation with flat prices (no gains/losses)."""
        prices = np.array([100.0] * 20)
        rsi = env._calc_rsi(prices)
        assert rsi == 50.0

    def test_rsi_only_gains(self, env):
        """Test RSI with only gains."""
        prices = np.array([100.0 + i for i in range(20)])
        rsi = env._calc_rsi(prices)
        assert rsi == 100.0

    def test_rsi_only_losses(self, env):
        """Test RSI with only losses."""
        prices = np.array([100.0 - i for i in range(20)])
        rsi = env._calc_rsi(prices)
        # Should be close to 0
        assert rsi < 10.0
