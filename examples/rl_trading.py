"""
Reinforcement Learning Training Example
========================================

Train a PPO agent to trade using the Schwab simulator environment.
"""

import os
import sys

import numpy as np
from gymnasium import spaces

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from schwabgym import MockClient, load_and_clean_data, split_train_test
from schwabgym.environment import SchwabTradingEnv, ZScoreNormalizer
from schwabgym.orders import MockEquities as eq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# STRATEGY DEFINITIONS (The "Opinions")
# =============================================================================


class StrategyObservationBuilder:
    """Defines the Agent's view of the world (State)."""

    def __init__(self, ticker):
        self.ticker = ticker
        self.normalizer = ZScoreNormalizer(shape=(8,))

    def reset(self):
        self.normalizer.reset()

    def __call__(self, client: MockClient) -> np.ndarray:
        # 1. Get Data
        hist = client.get_price_history(self.ticker).json()["candles"]
        prices = np.array([c["close"] for c in hist], dtype=np.float32)

        # 2. Calc Indicators (The "Opinionated" part)
        rsi = self._calc_rsi(prices)
        sma = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]

        # 3. Get Account State
        account_hash = client.get_account_numbers().json()["hashValue"]
        acct = client.get_account(account_hash).json()["securitiesAccount"]

        # Find position
        position_size = 0
        for pos in acct.get("positions", []):
            if pos["instrument"]["symbol"] == self.ticker:
                position_size = (
                    pos["longQuantity"]
                    if pos["longQuantity"] > 0
                    else -pos["shortQuantity"]
                )

        # 4. Assemble Vector
        current_price = prices[-1]
        raw_obs = np.array(
            [
                rsi,
                current_price / sma,
                position_size / 1000.0,
                # ... add other features as desired ...
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # Padding to match old shape of 8 for demo
            ],
            dtype=np.float32,
        )

        return self.normalizer.normalize(raw_obs)

    def _calc_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        delta = np.diff(prices[-period - 1 :])
        gains = delta[delta > 0].sum() / period
        losses = -delta[delta < 0].sum() / period
        if losses == 0:
            return 100.0 if gains > 0 else 50.0
        return 100 - (100 / (1 + gains / losses))


def strategy_reward_fn(client: MockClient) -> float:
    """Defines the Agent's goal (Reward)."""
    # Simple Log Returns
    current_nav = client._calculate_equity()  # Using helper for speed

    # We need previous NAV to calc return.
    # A simple stateless hack is to store it on the client or use a wrapper class.
    # For this demo, let's assume a simple attribute injection:
    prev_nav = getattr(client, "_prev_nav", client.initial_cash)

    ret = np.log(current_nav / prev_nav) if prev_nav > 0 else 0.0
    client._prev_nav = current_nav

    return float(ret)


def strategy_action_fn(client: MockClient, action: np.ndarray):
    """Defines how Agent actions translate to Orders."""
    ticker = "AAPL"  # Hardcoded for single-agent demo
    account_hash = client.get_account_numbers().json()["hashValue"]

    # Continuous action: [Signal (-1 to 1), Size (0 to 1)]
    signal = float(action[0])
    size_pct = float(action[1])

    if signal > 0.33:  # BUY
        # Logic to place buy order...
        order = eq.equity_buy_market(ticker, 10)  # Simplified size for demo
        client.place_order(account_hash, order)

    elif signal < -0.33:  # SELL
        # Logic to place sell order...
        order = eq.equity_sell_market(ticker, 10)  # Simplified size
        client.place_order(account_hash, order)


def strategy_termination_fn(client: MockClient) -> bool:
    """Defines fail states."""
    nav = client._calculate_equity()
    return nav < 15000.0  # Margin call


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train_rl_agent(
    ticker="AAPL",
    data_path="../data/AAPL_5min.csv",
    total_timesteps=100000,
    save_path="./models/",
):
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    df = load_and_clean_data(data_path, symbol=ticker)
    train_df, _ = split_train_test(df, train_ratio=0.8)

    def make_env():
        # 1. Create the Simulator
        client = MockClient(train_df, initial_cash=25000)

        # 2. Create the Strategy Wrappers
        obs_builder = StrategyObservationBuilder(ticker)

        # 3. Inject into the Generic Env
        return SchwabTradingEnv(
            client=client,
            observation_fn=obs_builder,
            reward_fn=strategy_reward_fn,
            action_fn=strategy_action_fn,
            termination_fn=strategy_termination_fn,
            observation_space=spaces.Box(
                low=-10, high=10, shape=(8,), dtype=np.float32
            ),
            action_space=spaces.Box(
                low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32
            ),
        )

    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env, norm_obs=False, norm_reward=True
    )  # Obs normalized manually in builder

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(save_path, "final_model"))

    return model


if __name__ == "__main__":
    train_rl_agent()
