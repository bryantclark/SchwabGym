"""
SchwabGym Physics Engines
==========================

Execution models for order simulation.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

from schwabgym.physics.almgren_chriss import AlmgrenChrissOptimalExecutor
from schwabgym.physics.base import ExecutionEngine, PhysicsMode
from schwabgym.physics.fast import FastExecutionEngine
from schwabgym.physics.hybrid import HybridExecutionEngine
from schwabgym.physics.realistic import RealisticExecutionEngine

__all__ = [
    "AlmgrenChrissOptimalExecutor",
    "ExecutionEngine",
    "FastExecutionEngine",
    "HybridExecutionEngine",
    "PhysicsMode",
    "RealisticExecutionEngine",
    "create_execution_engine",
]


def create_execution_engine(mode: str = "realistic", **kwargs) -> ExecutionEngine:
    """
    Factory function for creating execution engines.

    Args:
        mode (str): "fast", "realistic", or "hybrid"
        **kwargs: Engine-specific parameters

    Returns:
        ExecutionEngine: Configured engine instance

    Example:
        >>> # Fast mode for rapid prototyping
        >>> engine = create_execution_engine("fast")

        >>> # Realistic mode for production (DEFAULT)
        >>> engine = create_execution_engine("realistic", impact_coefficient=0.8)

        >>> # Hybrid mode for robust RL training
        >>> engine = create_execution_engine("hybrid", realistic_probability=0.3)
    """
    mode = mode.lower()

    if mode == "fast":
        return FastExecutionEngine(**kwargs)
    elif mode == "realistic":
        return RealisticExecutionEngine(**kwargs)
    elif mode == "hybrid":
        return HybridExecutionEngine(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fast', 'realistic', or 'hybrid'")
