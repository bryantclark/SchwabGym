"""
Schwab Trading Simulator - Gymnasium Environment
================================================

Reinforcement learning environment for training trading agents with realistic
market simulation and API-compatible interface.

Author: Bryant Clark
License: MIT
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Any

from schwabgym.client import MockClient
from schwabgym.orders import MockEquities as eq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZScoreNormalizer:
    """
    Online z-score normalization for observation spaces.
    
    Implements Welford's algorithm for numerically stable online mean and
    variance calculation. This ensures observations are properly scaled
    without requiring full dataset statistics upfront.
    
    Attributes:
        shape (tuple): Shape of observations
        mean (np.ndarray): Running mean
        var (np.ndarray): Running variance
        count (float): Number of observations seen
        clip (float): Maximum absolute z-score value
        
    Example:
        >>> normalizer = ZScoreNormalizer(shape=(8,))
        >>> norm_obs = normalizer.normalize(raw_obs)
    """
    
    def __init__(self, shape: Tuple[int, ...], clip_range: float = 10.0):
        """
        Initialize normalizer.
        
        Args:
            shape (tuple): Shape of observation vectors
            clip_range (float): Maximum absolute z-score (prevents outliers)
        """
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4  # Small initial count to avoid division by zero
        self.clip = clip_range
        
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation using running statistics.
        
        Uses Welford's online algorithm for numerical stability.
        
        Args:
            obs (np.ndarray): Raw observation vector
            
        Returns:
            np.ndarray: Normalized observation with zero mean, unit variance
        """
        self.count += 1
        
        # Update mean
        delta = obs - self.mean
        new_mean = self.mean + delta / self.count
        
        # Update variance using M2 accumulator
        m_a = self.var * (self.count - 1)
        m_b = (obs - self.mean) * (obs - new_mean)
        self.var = (m_a + m_b) / self.count
        
        self.mean = new_mean
        
        # Normalize and clip
        std = np.sqrt(self.var) + 1e-8  # Avoid division by zero
        z = (obs - self.mean) / std
        
        return np.clip(z, -self.clip, self.clip)


class SchwabTradingEnv(gym.Env):
    """
    Gymnasium environment for training trading agents on Schwab market data.
    
    The environment provides a realistic trading simulation with:
    - Continuous action space (signal strength, position size)
    - Normalized observation space (technical indicators, time features)
    - Accurate execution with market impact and fees
    - Pattern Day Trader rule enforcement
    - Account value tracking and visualization
    
    Observation Space (8 features):
        [0] RSI (0-100)
        [1] Price / SMA-20
        [2] Price / Upper Bollinger Band
        [3] MACD (12-26)
        [4] Position size (shares / 1000)
        [5] Price / Avg cost basis
        [6] Time sin (cyclical hour encoding)
        [7] Time cos (cyclical hour encoding)
        
    Action Space (2 continuous values):
        [0] Signal: -1 (sell) to +1 (buy)
        [1] Size: 0 (no trade) to 1 (max position)
        
    Reward:
        Log return of account value per step
        
    Attributes:
        df (pd.DataFrame): Historical market data
        ticker (str): Symbol being traded
        client (MockClient): Simulator instance
        history_log (List): Episode history for visualization
        trades_log (List): Trade execution log
        
    Example:
        >>> from trading_env import SchwabTradingEnv
        >>> from stable_baselines3 import PPO
        >>> 
        >>> env = SchwabTradingEnv(df, ticker='AAPL')
        >>> model = PPO('MlpPolicy', env, verbose=1)
        >>> model.learn(total_timesteps=10000)
        >>> env.render_chart()
    """
    
    metadata = {'render_modes': ['human', 'chart']}
    
    def __init__(
        self,
        df,
        ticker: str,
        initial_cash: float = 25000.0,
        render_mode: Optional[str] = None
    ):
        """
        Initialize trading environment.
        
        Args:
            df (pd.DataFrame): Historical OHLCV data
            ticker (str): Ticker symbol to trade
            initial_cash (float): Starting cash balance
            render_mode (str, optional): Render mode ('human' or 'chart')
        """
        super(SchwabTradingEnv, self).__init__()
        
        self.df = df
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.render_mode = render_mode
        
        # Initialize simulator
        self.client = MockClient(self.df, initial_cash=initial_cash)
        self.account_hash = self.client.get_account_numbers().json()['hashValue']
        
        # Action space: [signal (-1 to 1), size (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: 8 normalized features
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # Observation normalizer
        self.normalizer = ZScoreNormalizer(shape=(8,))
        
        # Episode tracking
        self.history_log: List[Dict] = []
        self.trades_log: List[Dict] = []
        
        logger.info(f"Environment initialized for {ticker}")
        logger.info(f"Initial cash: ${initial_cash:,.2f}")

    def _get_obs(self) -> Tuple[np.ndarray, float, float, int]:
        """
        Construct observation vector from current market state.
        
        Returns:
            tuple: (normalized_obs, nav, current_price, position_shares)
        """
        # === CRITICAL: Parse JSON responses like real API ===
        
        # Get historical data for indicators
        hist_resp = self.client.get_price_history(self.ticker)
        candles = hist_resp.json()['candles']
        hist_prices = np.array([c['close'] for c in candles], dtype=np.float32)
        
        # Get current quote (raw price for PnL)
        quote_resp = self.client.get_quotes([self.ticker])
        raw_price = float(quote_resp.json()[self.ticker]['quote']['lastPrice'])
        
        # Get account state
        acct_resp = self.client.get_account(self.account_hash)
        acct = acct_resp.json()['securitiesAccount']
        nav = float(acct['currentBalances']['liquidationValue'])
        
        # === Calculate Technical Indicators ===
        
        # RSI (Relative Strength Index)
        rsi = self._calc_rsi(hist_prices)
        
        # Simple Moving Average
        sma = float(np.mean(hist_prices[-20:])) if len(hist_prices) >= 20 else raw_price
        
        # Bollinger Bands
        std = float(np.std(hist_prices[-20:])) if len(hist_prices) >= 20 else 1.0
        upper_bb = sma + 2 * std
        
        # MACD (Moving Average Convergence Divergence)
        if len(hist_prices) >= 26:
            macd = float(np.mean(hist_prices[-12:]) - np.mean(hist_prices[-26:]))
        else:
            macd = 0.0
        
        # === Time Features (cyclical encoding) ===
        t = self.client._get_current_time()
        mins = t.hour * 60 + t.minute
        t_sin = np.sin(2 * np.pi * (mins / 1440))  # 1440 mins in a day
        t_cos = np.cos(2 * np.pi * (mins / 1440))
        
        # === Position Information ===
        shares = 0
        avg_cost = raw_price
        
        for pos in acct['positions']:
            if pos['instrument']['symbol'] == self.ticker:
                shares = pos['longQuantity'] if pos['longQuantity'] > 0 else -pos['shortQuantity']
                avg_cost = pos['averagePrice']
                break
        
        # === Construct Raw Observation ===
        raw_obs = np.array([
            rsi,                        # [0] Momentum indicator
            raw_price / sma,            # [1] Trend indicator
            raw_price / upper_bb,       # [2] Volatility indicator
            macd,                       # [3] Trend momentum
            shares / 1000.0,            # [4] Position size (normalized)
            raw_price / avg_cost if avg_cost > 0 else 1.0,       # [5] Profit factor
            t_sin,                      # [6] Time (cyclical)
            t_cos                       # [7] Time (cyclical)
        ], dtype=np.float32)
        
        # Normalize observation
        norm_obs = self.normalizer.normalize(raw_obs)
        
        return norm_obs, nav, raw_price, shares

    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices (np.ndarray): Price array
            period (int): RSI period (default 14)
            
        Returns:
            float: RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        delta = np.diff(prices[-period-1:])
        gains = delta[delta > 0].sum() / period
        losses = -delta[delta < 0].sum() / period
        
        if losses == 0:
            return 100.0
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Reset options
            
        Returns:
            tuple: (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Reset simulator
        self.client = MockClient(self.df, initial_cash=self.initial_cash)
        self.account_hash = self.client.get_account_numbers().json()['hashValue']
        
        # Clear logs
        self.history_log = []
        self.trades_log = []
        
        # Get initial observation
        obs, _, _, _ = self._get_obs()
        
        logger.info("Environment reset")
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action (np.ndarray): [signal, size] where:
                signal: -1 (sell) to +1 (buy)
                size: 0 (no trade) to 1 (max size)
                
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Parse action
        signal = float(action[0])
        size_pct = np.clip(float(action[1]), 0.0, 1.0)
        
        # Get current state
        _, prev_nav, _, prev_shares = self._get_obs()
        
        executed_type = "HOLD"
        
        # === Execute Trading Logic ===
        
        if signal > 0.33:  # BUY signal
            if prev_shares < 0:
                # Cover short position
                qty = int(abs(prev_shares) * size_pct)
                if qty > 0:
                    order = eq.equity_buy_to_cover_market(self.ticker, qty)
                    resp = self.client.place_order(self.account_hash, order)
                    if resp.status_code == 201:
                        executed_type = "COVER"
            else:
                # Increase long position
                acct = self.client.get_account(self.account_hash).json()['securitiesAccount']
                bp = acct['currentBalances']['buyingPower']
                raw_price = self.client._get_current_raw_price(self.ticker)
                qty = int((bp * size_pct) // raw_price)
                
                if qty > 0:
                    order = eq.equity_buy_market(self.ticker, qty)
                    resp = self.client.place_order(self.account_hash, order)
                    if resp.status_code == 201:
                        executed_type = "BUY"

        elif signal < -0.33:  # SELL signal
            if prev_shares > 0:
                # Reduce long position
                qty = int(prev_shares * size_pct)
                if qty > 0:
                    order = eq.equity_sell_market(self.ticker, qty)
                    resp = self.client.place_order(self.account_hash, order)
                    if resp.status_code == 201:
                        executed_type = "SELL"
            else:
                # Increase short position
                acct = self.client.get_account(self.account_hash).json()['securitiesAccount']
                bp = acct['currentBalances']['buyingPower']
                raw_price = self.client._get_current_raw_price(self.ticker)
                qty = int((bp * size_pct) // raw_price)
                
                if qty > 0:
                    order = eq.equity_sell_short_market(self.ticker, qty)
                    resp = self.client.place_order(self.account_hash, order)
                    if resp.status_code == 201:
                        executed_type = "SHORT"

        # Advance time
        has_next = self.client.advance_time()
        terminated = not has_next
        
        # Get new state
        obs, nav, price, shares = self._get_obs()
        
        # Calculate reward (log return)
        if prev_nav > 0:
            ret = np.log(nav / prev_nav)
        else:
            ret = 0.0
        
        reward = float(ret)
        
        # Terminal conditions
        if nav < 15000:  # Margin call threshold
            terminated = True
            reward = -10.0
            logger.warning(f"Episode terminated: Account value below threshold (${nav:,.2f})")
        
        # Log history
        self.history_log.append({
            'nav': nav,
            'price': price,
            'step': self.client.current_step
        })
        
        if executed_type != "HOLD":
            self.trades_log.append({
                'step': self.client.current_step,
                'price': price,
                'type': executed_type
            })
        
        info = {
            'nav': nav,
            'price': price,
            'shares': shares,
            'action_taken': executed_type
        }
        
        return obs, reward, terminated, False, info

    def render(self):
        """Render environment (placeholder for future implementations)."""
        if self.render_mode == 'human':
            logger.info(f"Step {self.client.current_step}: NAV = ${self.history_log[-1]['nav']:,.2f}")
        elif self.render_mode == 'chart':
            self.render_chart()

    def render_chart(self) -> None:
        """
        Render trading performance chart.
        
        Creates a two-panel chart showing:
        - Top: Price action with buy/sell markers
        - Bottom: Account value over time
        """
        if len(self.history_log) == 0:
            logger.warning("No data to plot")
            return
        
        try:
            # Extract data
            steps = [x['step'] for x in self.history_log]
            prices = [x['price'] for x in self.history_log]
            navs = [x['nav'] for x in self.history_log]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot price with trades
            ax1.plot(steps, prices, color='gray', alpha=0.6, linewidth=1.5, label='Price')
            
            for trade in self.trades_log:
                if trade['type'] in ['BUY', 'COVER']:
                    color = 'green'
                    marker = '^'
                else:
                    color = 'red'
                    marker = 'v'
                
                ax1.scatter(
                    trade['step'],
                    trade['price'],
                    c=color,
                    marker=marker,
                    s=100,
                    zorder=5,
                    alpha=0.7,
                    label=trade['type'] if trade == self.trades_log[0] else ""
                )
            
            ax1.set_ylabel("Price ($)", fontsize=12)
            ax1.set_title(f"{self.ticker} Trading Performance", fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Plot account value
            ax2.plot(steps, navs, color='blue', linewidth=2, label='Account Value')
            ax2.axhline(y=self.initial_cash, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
            ax2.set_xlabel("Time Step", fontsize=12)
            ax2.set_ylabel("Account Value ($)", fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            # Format y-axis
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            plt.show()
            
            # Print summary
            final_nav = navs[-1]
            total_return = ((final_nav - self.initial_cash) / self.initial_cash) * 100
            logger.info(f"Episode Summary:")
            logger.info(f"  Initial Capital: ${self.initial_cash:,.2f}")
            logger.info(f"  Final Value: ${final_nav:,.2f}")
            logger.info(f"  Return: {total_return:.2f}%")
            logger.info(f"  Total Trades: {len(self.trades_log)}")
            
        except Exception as e:
            logger.error(f"Error rendering chart: {e}")

    def close(self):
        """Clean up resources."""
        pass