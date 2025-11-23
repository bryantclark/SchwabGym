"""
Environment Tests
=================

Unit tests for the SchwabTradingEnv Gymnasium environment.

Author: Bryant Clark
"""

import pytest
import numpy as np
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
        return SchwabTradingEnv(sample_data, ticker='TEST')

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.ticker == 'TEST'
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
        assert info['action_taken'] == 'HOLD'

    def test_step_buy(self, env):
        """Test stepping with BUY action."""
        env.reset()
        # Action: [1.0 (buy), 0.5 (50% size)]
        action = np.array([1.0, 0.5], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['action_taken'] == 'BUY'
        assert info['shares'] > 0

    def test_step_sell(self, env):
        """Test stepping with SELL action."""
        env.reset()
        
        # First buy
        env.step(np.array([1.0, 0.5], dtype=np.float32))
        
        # Then sell
        # Action: [-1.0 (sell), 1.0 (100% size)]
        action = np.array([-1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['action_taken'] == 'SELL'
        assert info['shares'] == 0 # Should be closed

    def test_step_short(self, env):
        """Test stepping with SHORT action."""
        env.reset()
        # Action: [-1.0 (sell), 0.5 (50% size)]
        action = np.array([-1.0, 0.5], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['action_taken'] == 'SHORT'
        assert info['shares'] < 0

    def test_step_cover(self, env):
        """Test stepping with COVER action."""
        env.reset()
        
        # First short
        env.step(np.array([-1.0, 0.5], dtype=np.float32))
        
        # Then cover
        # Action: [1.0 (buy), 1.0 (100% size)]
        action = np.array([1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['action_taken'] == 'COVER'
        assert info['shares'] == 0

    def test_margin_call_termination(self, env):
        """Test termination on margin call."""
        env.reset()
        # Force account value low
        env.client.cash = 10000.0 # Below 15k threshold
        
        # Take a step
        _, reward, terminated, _, _ = env.step(np.array([0.0, 0.0]))
        
        assert terminated is True
        assert reward == -10.0

    def test_render_chart(self, env):
        """Test render chart (smoke test)."""
        env.reset()
        env.step(np.array([1.0, 0.1]))
        env.step(np.array([-1.0, 0.1]))
        
        # uncomment to actually test. I just dont like the popups
        # env.render_chart()
