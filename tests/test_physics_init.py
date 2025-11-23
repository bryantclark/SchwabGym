import pytest
from schwabgym.physics import create_execution_engine, FastExecutionEngine, RealisticExecutionEngine, HybridExecutionEngine, AlmgrenChrissOptimalExecutor

def test_create_engine_fast():
    engine = create_execution_engine('fast')
    assert isinstance(engine, FastExecutionEngine)

def test_create_engine_realistic():
    engine = create_execution_engine('realistic')
    assert isinstance(engine, RealisticExecutionEngine)

def test_create_engine_hybrid():
    engine = create_execution_engine('hybrid')
    assert isinstance(engine, HybridExecutionEngine)

def test_create_engine_invalid():
    with pytest.raises(ValueError):
        create_execution_engine('unknown')
