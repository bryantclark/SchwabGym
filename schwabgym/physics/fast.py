"""
SchwabGym Fast Execution Engine
================================

Simplified physics for rapid prototyping.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import logging

from schwabgym.physics.base import ExecutionEngine

logger = logging.getLogger(__name__)


class FastExecutionEngine(ExecutionEngine):
    """
    Fast execution model for rapid training.

    - Simple additive slippage
    - Binary limit order fills
    - Minimal computational overhead

    Use when:
    - Debugging strategy logic on CPU
    - Quick prototypes
    - Testing basic functionality
    """

    def __init__(self, base_slippage: float = 0.01):
        """
        Initialize fast execution engine.

        Args:
            base_slippage (float): Fixed slippage in dollars (default: $0.01)
        """
        self.base_slippage = base_slippage
        logger.info(f"FastExecutionEngine initialized (slippage=${base_slippage:.4f})")

    _BUY_INSTRUCTIONS = frozenset(
        {"BUY", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE"}
    )

    def calculate_execution_price(
        self, base_price: float, quantity: int, instruction: str, market_data: dict
    ) -> float:
        """
        Simple slippage model.

        Buy orders: base_price + slippage
        Sell orders: base_price - slippage
        """
        if instruction in self._BUY_INSTRUCTIONS:
            return base_price + self.base_slippage
        return base_price - self.base_slippage

    def should_limit_fill(
        self,
        limit_price: float,
        market_high: float,
        market_low: float,
        volume: int,
        quantity: int,
    ) -> bool:
        """
        Binary fill logic - price touched = filled.

        Buy limit fills if market touched or went below limit.
        Sell limit fills if market touched or went above limit.
        """
        return market_low <= limit_price <= market_high
