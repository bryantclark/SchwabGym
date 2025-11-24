"""
Schwab Trading Simulator - Gymnasium Environment
================================================

Generic Reinforcement Learning environment wrapper for SchwabGym.
This module provides the shell that connects the MockClient simulator
to a Gymnasium interface.

Author: Bryant Clark
License: MIT
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from schwabgym.client import MockClient

logger = logging.getLogger(__name__)


class ZScoreNormalizer:
    """
    Online z-score normalization helper.

    Useful for normalizing observation vectors in your ObservationBuilder.
    """

    def __init__(self, shape: Tuple[int, ...], clip_range: float = 10.0):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4
        self.clip = clip_range

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        self.count += 1
        delta = obs - self.mean
        new_mean = self.mean + delta / self.count
        m_a = self.var * (self.count - 1)
        m_b = (obs - self.mean) * (obs - new_mean)
        self.var = (m_a + m_b) / self.count
        self.mean = new_mean
        std = np.sqrt(self.var) + 1e-8
        z = (obs - self.mean) / std
        return np.clip(z, -self.clip, self.clip)

    def reset(self):
        """Reset statistics."""
        self.mean = np.zeros(self.shape, dtype=np.float32)
        self.var = np.ones(self.shape, dtype=np.float32)
        self.count = 1e-4


class SchwabTradingEnv(gym.Env):
    """
    Generic Gymnasium environment for SchwabGym.

    This environment is unopinionated. It delegates:
    - 'State' definition to `observation_fn`
    - 'Reward' calculation to `reward_fn`
    - 'Action' interpretation to `action_fn`

    This allows you to use the same simulator for:
    - Price action agents (raw candles)
    - Indicator-based agents (RSI/MACD)
    - Order execution agents (Level 2 data)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        client: MockClient,
        observation_fn: Callable[[MockClient], np.ndarray],
        reward_fn: Callable[[MockClient], float],
        action_fn: Callable[[MockClient, Any], None],
        observation_space: spaces.Space,
        action_space: spaces.Space,
        termination_fn: Optional[Callable[[MockClient], bool]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the generic environment.

        Args:
            client: An instance of schwabgym.MockClient
            observation_fn: Function(client) -> np.ndarray
            reward_fn: Function(client) -> float
            action_fn: Function(client, action) -> None
            observation_space: Gym space definition for observations
            action_space: Gym space definition for actions
            termination_fn: Optional Function(client) -> bool (e.g. margin call check)
            render_mode: 'human' or None
        """
        super().__init__()
        self.client = client
        self.observation_fn = observation_fn
        self.reward_fn = reward_fn
        self.action_fn = action_fn
        self.termination_fn = termination_fn or (lambda c: False)

        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode

        # Helper for tracking resets for stateful wrappers
        self._observation_wrapper_ref = getattr(observation_fn, "__self__", None)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # 1. Reset Simulator
        self.client.reset()

        # 2. Reset Observation Helper (if stateful)
        # If the user passed a bound method of a class (like a wrapper),
        # try to call .reset() on it if it exists.
        if hasattr(self.observation_fn, "reset"):
            self.observation_fn.reset()
        elif self._observation_wrapper_ref and hasattr(
            self._observation_wrapper_ref, "reset"
        ):
            self._observation_wrapper_ref.reset()

        # 3. Get initial state
        obs = self.observation_fn(self.client)
        return obs, {}

    def step(self, action):
        # 1. Interpret and execute action (User Logic)
        self.action_fn(self.client, action)

        # 2. Advance Simulator (Physics)
        has_next = self.client.advance_time()

        # 3. Calculate Observables (User Logic)
        obs = self.observation_fn(self.client)
        reward = self.reward_fn(self.client)

        # 4. Check termination
        terminated = (not has_next) or self.termination_fn(self.client)

        info = {
            "current_step": self.client.current_step,
            "nav": self.client._calculate_equity(),  # Helper access for logging
        }

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            nav = self.client._calculate_equity()
            logger.info(f"Step {self.client.current_step}: NAV = ${nav:,.2f}")
