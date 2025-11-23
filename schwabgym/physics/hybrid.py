"""
SchwabGym Hybrid Execution Engine
==================================

Domain randomization wrapper for execution engines.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import numpy as np
import logging
from typing import Dict, Optional
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
        fast_engine: Optional[FastExecutionEngine] = None,
        realistic_engine: Optional[RealisticExecutionEngine] = None,
        seed: Optional[int] = None
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
        self.current_mode = None
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(
            f"HybridExecutionEngine initialized "
            f"(realistic_prob={realistic_probability*100:.0f}%)"
        )
    
    def _select_engine(self) -> ExecutionEngine:
        """Randomly select engine based on probability."""
        if np.random.random() < self.p_realistic:
            self.current_mode = PhysicsMode.REALISTIC
            return self.realistic
        else:
            self.current_mode = PhysicsMode.FAST
            return self.fast
    
    def calculate_execution_price(
        self,
        base_price: float,
        quantity: int,
        instruction: str,
        market_data: Dict
    ) -> float:
        """Delegate to randomly selected engine."""
        engine = self._select_engine()
        return engine.calculate_execution_price(
            base_price, quantity, instruction, market_data
        )
    
    def should_limit_fill(
        self,
        limit_price: float,
        market_high: float,
        market_low: float,
        volume: int,
        quantity: int
    ) -> bool:
        """Delegate to randomly selected engine."""
        engine = self._select_engine()
        return engine.should_limit_fill(
            limit_price, market_high, market_low, volume, quantity
        )
    
    def get_statistics(self) -> Dict:
        """
        Return usage statistics.
        
        Returns:
            dict: Current mode and configuration
        """
        return {
            "mode": "hybrid",
            "realistic_probability": self.p_realistic,
            "current_mode": self.current_mode.value if self.current_mode else None
        }
