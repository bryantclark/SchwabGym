import pytest
import numpy as np
from schwabgym.physics.almgren_chriss import AlmgrenChrissOptimalExecutor

# def test_compute_trajectory_basic():
#     executor = AlmgrenChrissOptimalExecutor(lambda_risk=0.01, eta_temp=0.1, gamma_perm=0.05)
#     total_shares = 10000
#     T = 1.0
#     N = 5
#     volatility = 0.02
#     traj = executor.compute_trajectory(total_shares, T, N, volatility)
#     # Should be integer array of length N
#     assert isinstance(traj, np.ndarray)
#     assert traj.shape == (N,)
#     assert traj.sum() == total_shares
#     # First period should be larger than last due to front-loading
#     assert traj[0] > traj[-1]

def test_estimate_impact_cost():
    executor = AlmgrenChrissOptimalExecutor(lambda_risk=0.01, eta_temp=0.1, gamma_perm=0.05)
    trajectory = np.array([3000, 2500, 2000, 1500, 1000])
    volatility = 0.02
    impact_bps = executor.estimate_impact_cost(trajectory, volatility)
    # Impact should be positive float
    assert isinstance(impact_bps, float)
    assert impact_bps > 0
