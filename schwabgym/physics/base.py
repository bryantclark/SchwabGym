"""
SchwabGym Physics Engine Base Classes
======================================

Abstract base classes for execution engines.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

from abc import ABC, abstractmethod
from enum import Enum


class PhysicsMode(Enum):
    """Execution physics complexity levels."""

    FAST = "fast"  # Simple model, maximum speed
    REALISTIC = "realistic"  # Full microstructure simulation
    HYBRID = "hybrid"  # Domain randomization


class ExecutionEngine(ABC):
    """
    Abstract base class for execution models.

    All execution engines must implement these methods to be compatible
    with MockClient.
    """

    @abstractmethod
    def calculate_execution_price(
        self, base_price: float, quantity: int, instruction: str, market_data: dict
    ) -> float:
        """
        Calculate the actual execution price including impact.

        Args:
            base_price (float): Market price from data
            quantity (int): Order size in shares
            instruction (str): BUY, SELL, etc.
            market_data (dict): OHLCV + volatility for current bar

        Returns:
            float: Actual execution price
        """
        pass  # pragma: no cover

    @abstractmethod
    def should_limit_fill(
        self,
        limit_price: float,
        market_high: float,
        market_low: float,
        volume: int,
        quantity: int,
    ) -> bool:
        """
        Determine if a limit order should fill.

        Args:
            limit_price (float): Limit order price
            market_high (float): Bar high
            market_low (float): Bar low
            volume (int): Bar volume
            quantity (int): Order size

        Returns:
            bool: True if order fills
        """
        pass  # pragma: no cover
