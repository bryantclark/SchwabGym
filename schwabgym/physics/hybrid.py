"""
SchwabGym Hybrid Execution Engine
==================================

Domain randomization wrapper for execution engines.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import logging

import numpy as np

from schwabgym.physics.base import ExecutionEngine, PhysicsMode
from schwabgym.physics.fast import FastExecutionEngine
from schwabgym.physics.realistic import RealisticExecutionEngine

logger = logging.getLogger(__name__)


class HybridExecutionEngine(ExecutionEngine):
    """
    Domain Randomization Engine.

    Randomly switches between Fast and Realistic modes to create
    a robust training environment.

    Usage in RL:
        The agent never knows which physics engine is active.
        This forces it to learn strategies that work under both
        perfect (fast) and imperfect (realistic) execution.
    """

    def __init__(
        self,
        realistic_probability: float = 0.3,
        fast_engine: FastExecutionEngine | None = None,
        realistic_engine: RealisticExecutionEngine | None = None,
        seed: int | None = None,
    ):
        """
        Initialize hybrid execution engine.

        Args:
            realistic_probability (float): Probability of using realistic physics
                - 0.0 = always fast
                - 0.3 = 30% realistic, 70% fast (recommended)
                - 1.0 = always realistic
            fast_engine (FastExecutionEngine, optional): Custom fast engine
            realistic_engine (RealisticExecutionEngine, optional): Custom realistic engine
            seed (int, optional): Random seed for reproducibility
        """
        self.p_realistic = realistic_probability
        self.fast = fast_engine or FastExecutionEngine()
        self.realistic = realistic_engine or RealisticExecutionEngine()
        self.current_mode: PhysicsMode | None = None
        self._active_engine: ExecutionEngine | None = None
        self._rng = np.random.default_rng(seed)

        logger.info(
            f"HybridExecutionEngine initialized "
            f"(realistic_prob={realistic_probability * 100:.0f}%)"
        )

    def select_engine_for_step(self) -> None:
        """Select engine for the current time step. Call once per step."""
        if self._rng.random() < self.p_realistic:
            self.current_mode = PhysicsMode.REALISTIC
            self._active_engine = self.realistic
        else:
            self.current_mode = PhysicsMode.FAST
            self._active_engine = self.fast

    def prepare_step(self) -> None:
        """Pick a fresh engine for the next simulation step."""
        self.select_engine_for_step()

    def reset_episode(self) -> None:
        """Clear any active mode so a new episode can re-sample physics."""
        self.current_mode = None
        self._active_engine = None

    def _get_engine(self) -> ExecutionEngine:
        """Return the active engine, selecting one if needed."""
        if self._active_engine is None:
            self.select_engine_for_step()
        return self._active_engine  # type: ignore[return-value]

    def calculate_execution_price(
        self, base_price: float, quantity: int, instruction: str, market_data: dict
    ) -> float:
        """Delegate to the active engine (consistent per step)."""
        return self._get_engine().calculate_execution_price(
            base_price, quantity, instruction, market_data
        )

    def should_limit_fill(
        self,
        limit_price: float,
        market_high: float,
        market_low: float,
        volume: int,
        quantity: int,
    ) -> bool:
        """Delegate to the active engine (consistent per step)."""
        return self._get_engine().should_limit_fill(
            limit_price, market_high, market_low, volume, quantity
        )

    def get_statistics(self) -> dict:
        """
        Return usage statistics.

        Returns:
            dict: Current mode and configuration
        """
        return {
            "mode": "hybrid",
            "realistic_probability": self.p_realistic,
            "current_mode": self.current_mode.value if self.current_mode else None,
        }
