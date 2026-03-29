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
        mode: One of "fast", "realistic", or "hybrid".
        **kwargs: Engine-specific parameters.

    Returns:
        Configured engine instance.

    Example:
        >>> engine = create_execution_engine("fast")
        >>> engine = create_execution_engine("realistic", impact_coefficient=0.8)
        >>> engine = create_execution_engine("hybrid", realistic_probability=0.3)
    """
    _engines: dict[str, type[ExecutionEngine]] = {
        "fast": FastExecutionEngine,
        "realistic": RealisticExecutionEngine,
        "hybrid": HybridExecutionEngine,
    }
    cls = _engines.get(mode.lower())
    if cls is None:
        raise ValueError(f"Unknown mode: {mode}. Use one of: {sorted(_engines.keys())}")
    return cls(**kwargs)
