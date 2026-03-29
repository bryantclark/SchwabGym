"""
SchwabGym Realistic Execution Engine
=====================================

Market microstructure simulation engine.
This is the default engine for simulations.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import logging

import numpy as np

from schwabgym.physics.base import ExecutionEngine

logger = logging.getLogger(__name__)


class RealisticExecutionEngine(ExecutionEngine):
    """
    Execution model based on empirical market microstructure.

    Implements:
    - Square Root Law of market impact: ΔP = Y×σ×sqrt(Q/V)
    - Volume-based fill probabilities
    - Brownian Bridge for intraday paths (optional)
    """

    def __init__(
        self,
        impact_coefficient: float = 0.7,
        participation_rate: float = 0.10,
        queue_depth_factor: float = 2.0,
        seed: int | None = None,
    ):
        """
        Initialize realistic execution engine.

        Args:
            impact_coefficient (float): Y in Square Root Law (0.5-1.0 typical)
                - 0.5: Liquid large-cap stocks
                - 0.7: Typical stocks (default)
                - 1.0: Illiquid small-cap stocks
            participation_rate (float): Max fraction of volume (default: 10%)
            queue_depth_factor (float): Estimated orders ahead (default: 2.0)
            seed (int, optional): Random seed for reproducibility
        """
        self.Y = impact_coefficient
        self.alpha = participation_rate
        self.beta = queue_depth_factor
        self._rng = np.random.default_rng(seed)
        logger.info(
            f"RealisticExecutionEngine initialized "
            f"(Y={impact_coefficient:.2f}, α={participation_rate:.2f})"
        )

    def calculate_execution_price(
        self, base_price: float, quantity: int, instruction: str, market_data: dict
    ) -> float:
        """
        Square Root Law implementation.

        Formula: ΔP = Y × σ × sqrt(Q/V)

        Where:
            Y = impact coefficient (calibrated from data)
            σ = volatility (estimated from High-Low range)
            Q = order quantity
            V = period volume
        """
        # Extract market microstructure data
        high = market_data.get("High", base_price * 1.01)
        low = market_data.get("Low", base_price * 0.99)
        volume = market_data.get("Volume", 100000)

        # Parkinson volatility estimator: σ = ln(H/L) / sqrt(4·ln(2))
        # More statistically efficient than (H-L)/C; unbiased for GBM prices
        if base_price > 0 and high > low > 0:
            volatility = np.log(high / low) / np.sqrt(4 * np.log(2))
        else:
            volatility = 0.01

        # Prevent division by zero
        volume = max(volume, 1)

        # Square Root Law
        impact = self.Y * volatility * np.sqrt(quantity / volume)

        # Apply impact directionally
        if instruction in ["BUY", "BUY_TO_COVER", "BUY_TO_OPEN"]:
            execution_price = base_price * (1 + impact)
        else:
            execution_price = base_price * (1 - impact)

        logger.debug(
            f"Impact: Q={quantity}, V={volume}, σ={volatility:.4f} "
            f"→ ΔP={impact * 100:.3f}%"
        )

        return float(execution_price)

    def should_limit_fill(
        self,
        limit_price: float,
        market_high: float,
        market_low: float,
        volume: int,
        quantity: int,
    ) -> bool:
        """
        Probabilistic fill based on volume and queue depth.

        Formula: P(fill) = min(1.0, (V × α) / (Q + β))

        Where:
            V = bar volume
            α = participation rate (can't be more than X% of volume)
            Q = our order size
            β = estimated queue depth ahead of us
        """
        # First check: did price even touch our limit?
        price_touched = market_low <= limit_price <= market_high

        if not price_touched:
            return False

        # Calculate fill probability based on liquidity
        available_liquidity = volume * self.alpha
        required_liquidity = quantity + self.beta

        fill_probability = min(1.0, available_liquidity / required_liquidity)

        # Stochastic fill decision
        filled = self._rng.random() < fill_probability

        if not filled:
            logger.debug(
                f"Limit order NOT filled: price touched but insufficient liquidity "
                f"(P={fill_probability * 100:.1f}%, V={volume}, Q={quantity})"
            )

        return filled

    def simulate_intrabar_path(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        n_ticks: int = 60,
    ) -> np.ndarray:
        """
        Brownian Bridge: Generate realistic intraday path.

        Creates a stochastic path that:
        - Starts at open_price
        - Ends at close
        - Respects high and low bounds

        Args:
            open_price (float): Bar open
            high (float): Bar high
            low (float): Bar low
            close (float): Bar close
            n_ticks (int): Number of micro-ticks to simulate

        Returns:
            np.ndarray: Simulated price path of length n_ticks
        """
        # Generate standard Brownian bridge
        t = np.linspace(0, 1, n_ticks)
        W = self._rng.standard_normal(n_ticks).cumsum() * np.sqrt(1 / n_ticks)

        # Bridge formula: B(t) = W(t) - t*W(1)
        bridge = W - t * W[-1]

        # Scale to match open->close drift
        drift = close - open_price
        path = open_price + drift * t + bridge * (high - low) / 4

        # Ensure we hit high and low at some point
        path[n_ticks // 4] = high  # Hit high at 25% through bar
        path[3 * n_ticks // 4] = low  # Hit low at 75% through bar

        # Ensure bounds
        path = np.clip(path, low, high)

        # Force exact endpoints
        path[0] = open_price
        path[-1] = close

        return np.asarray(path)
