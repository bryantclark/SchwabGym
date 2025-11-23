"""
SchwabGym Almgren-Chriss Optimal Execution
===========================================

Optimal execution framework for large orders.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT

Reference:
    Almgren, R., & Chriss, N. (2000). "Optimal execution of portfolio transactions."
    Journal of Risk, 3, 5-40.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class AlmgrenChrissOptimalExecutor:
    """
    Almgren-Chriss Optimal Execution Framework.
    
    Splits a large "parent" order into multiple "child" orders
    over time to balance market impact vs. timing risk.
    
    The framework solves:
        minimize: E[Cost] + λ × Var[Cost]
    
    Where λ is the trader's risk aversion parameter.
    """
    
    def __init__(
        self,
        lambda_risk: float = 0.01,
        eta_temp: float = 0.1,
        gamma_perm: float = 0.05
    ):
        """
        Initialize Almgren-Chriss executor.
        
        Args:
            lambda_risk (float): Risk aversion parameter
                - 0.0 = risk neutral (minimize expected cost)
                - 0.01 = typical institutional risk aversion (default)
                - 0.1+ = very risk averse (execute quickly)
            eta_temp (float): Temporary impact coefficient
            gamma_perm (float): Permanent impact coefficient
        """
        self.lambda_risk = lambda_risk
        self.eta = eta_temp
        self.gamma = gamma_perm
        
        logger.info(
            f"AlmgrenChrissOptimalExecutor initialized "
            f"(λ={lambda_risk:.4f}, η={eta_temp:.4f}, γ={gamma_perm:.4f})"
        )
    
    def compute_trajectory(
        self,
        total_shares: int,
        T: float,
        N: int,
        volatility: float
    ) -> np.ndarray:
        """
        Compute optimal execution schedule.
        
        Returns array of shares to trade in each period.
        
        Formula (closed-form solution):
            x_k = X * sinh(κ(T - t_k)) / sinh(κT)
        
        Where κ = sqrt(λσ² / η) captures urgency.
        
        Args:
            total_shares (int): Total position to liquidate
            T (float): Time horizon (in days or hours)
            N (int): Number of trading periods
            volatility (float): Asset volatility (daily or per-period)
            
        Returns:
            np.ndarray: Shares to trade each period [q_1, q_2, ..., q_N]
        """
        X = total_shares
        tau = T / N
        
        # Compute urgency parameter kappa
        if self.eta == 0:
            # Risk-neutral case: TWAP (Time-Weighted Average Price)
            kappa = 0
        else:
            kappa = np.sqrt((self.lambda_risk * volatility**2) / self.eta)
        
        # Generate time grid
        t = np.linspace(0, T, N + 1)
        
        # Optimal holdings at each time point
        if kappa == 0:
            # TWAP: Linear trajectory
            x = X * (1 - t / T)
        else:
            # Almgren-Chriss: Exponential trajectory
            x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        
        # Convert holdings to trades (differences)
        trades = -np.diff(x)  # Negative diff because we're selling
        
        logger.debug(
            f"Trajectory computed: X={X}, T={T:.2f}, N={N} "
            f"→ Front-loaded={trades[0]/X*100:.1f}% in first period"
        )
        
        # Round trades to nearest integer and adjust to match total_shares
        trades_int = np.rint(trades).astype(int)
        diff = total_shares - trades_int.sum()
        if diff != 0:
            # Adjust the first trade to compensate the rounding error
            trades_int[0] += diff
        return trades_int
    
    def estimate_impact_cost(
        self,
        trajectory: np.ndarray,
        volatility: float
    ) -> float:
        """
        Estimate total impact cost for a given trajectory.
        
        This provides a Transaction Cost Analysis (TCA) prediction.
        
        Args:
            trajectory (np.ndarray): Trading schedule
            volatility (float): Asset volatility
            
        Returns:
            float: Estimated total impact (in basis points)
        """
        total_impact = 0.0
        
        for q in trajectory:
            # Temporary impact
            temp_impact = self.eta * q
            
            # Permanent impact
            perm_impact = self.gamma * q
            
            total_impact += temp_impact + perm_impact
        
        # Convert to basis points
        total_shares = trajectory.sum()
        impact_bps = (total_impact / total_shares) * 10000
        
        return impact_bps
