"""
Basic Trading Example
=====================

Demonstrates simple mean reversion strategy using the Schwab simulator.
This code is identical to what you'd write for live trading - just change
the import statement!

Author: Your Name
"""

import os
import sys

# Add parent directory to path so we can import schwabgym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from schwabgym.client import MockClient
from schwabgym.data import load_and_clean_data
from schwabgym.orders import MockEquities as eq

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simple_mean_reversion_strategy():
    """
    Simple mean reversion strategy example.

    Strategy Logic:
    - Buy when price is 2% below 20-period SMA
    - Sell when price is 2% above 20-period SMA
    - Position size: $5,000 per trade
    """
    # Load your historical data
    ticker = "SPY"
    # Assuming data is in ../data relative to examples/
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "SPY_1min.csv")

    # For demo purposes, if file doesn't exist, generate dummy data
    df = load_and_clean_data(data_path, symbol=ticker)

    # Initialize simulator with $100k starting capital
    client = MockClient(df, initial_cash=100000.0)
    account_hash = client.get_account_numbers().json()["hashValue"]

    logger.info("=" * 60)
    logger.info("MEAN REVERSION STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbol: {ticker}")
    logger.info(f"Data points: {len(df)}")
    logger.info("Starting capital: $100,000")
    logger.info("=" * 60)

    trade_count = 0

    # Main trading loop
    while client.advance_time():
        current_step = client.current_step

        # Get recent price history for indicator calculation
        hist_resp = client.get_price_history(ticker)
        candles = hist_resp.json()["candles"]

        if len(candles) < 20:
            continue  # Need at least 20 bars for SMA

        # Extract prices
        prices = [c["close"] for c in candles]
        current_price = prices[-1]

        # Calculate 20-period Simple Moving Average
        sma_20 = sum(prices[-20:]) / 20

        # Get account state
        acct_resp = client.get_account(account_hash)
        acct = acct_resp.json()["securitiesAccount"]
        cash = acct["currentBalances"]["cashBalance"]

        # Find current position
        current_position = 0
        for pos in acct.get("positions", []):
            if pos["instrument"]["symbol"] == ticker:
                current_position = pos["longQuantity"]
                break

        # === TRADING LOGIC ===

        # BUY SIGNAL: Price 2% below SMA and we have cash
        if current_price < sma_20 * 0.98 and cash > 5000 and current_position == 0:
            qty = int(5000 / current_price)

            if qty > 0:
                order = eq.equity_buy_market(ticker, qty)
                resp = client.place_order(account_hash, order)

                if resp.status_code == 201:
                    trade_count += 1
                    logger.info(
                        f"[{current_step}] BUY {qty} @ ${current_price:.2f} | SMA: ${sma_20:.2f}"
                    )

        # SELL SIGNAL: Price 2% above SMA and we have position
        elif current_price > sma_20 * 1.02 and current_position > 0:
            order = eq.equity_sell_market(ticker, int(current_position))
            resp = client.place_order(account_hash, order)

            if resp.status_code == 201:
                trade_count += 1
                logger.info(
                    f"[{current_step}] SELL {current_position} @ ${current_price:.2f} | SMA: ${sma_20:.2f}"
                )

        # Print progress every 100 steps
        if current_step % 100 == 0:
            nav = acct["currentBalances"]["liquidationValue"]
            logger.debug(f"Step {current_step}: NAV = ${nav:,.2f}")

    # === FINAL RESULTS ===

    final_acct = client.get_account(account_hash).json()["securitiesAccount"]
    final_nav = final_acct["currentBalances"]["liquidationValue"]

    initial_capital = 100000.0
    total_return = ((final_nav - initial_capital) / initial_capital) * 100

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Final Value: ${final_nav:,.2f}")
    logger.info(f"Total Return: {total_return:+.2f}%")
    logger.info(f"Total Trades: {trade_count}")
    logger.info(f"Day Trades: {len(client.day_trades)}")
    logger.info("=" * 60)

    return final_nav, total_return


def run_multiple_strategies():
    """
    Compare multiple strategy variations.

    This demonstrates how to test different parameters quickly.
    """
    results = []

    # Test different SMA thresholds
    for threshold in [0.01, 0.02, 0.03]:
        logger.info(f"\nTesting threshold: {threshold * 100:.0f}%")
        # You would modify the strategy here to use this threshold
        final_nav, ret = simple_mean_reversion_strategy()
        results.append({"threshold": threshold, "return": ret, "final_nav": final_nav})

    # Find best strategy
    best = max(results, key=lambda x: x["return"])
    logger.info(f"\nBest strategy: {best['threshold'] * 100:.0f}% threshold")
    logger.info(f"Return: {best['return']:.2f}%")


if __name__ == "__main__":
    try:
        simple_mean_reversion_strategy()
        # run_multiple_strategies()
    except KeyboardInterrupt:
        logger.info("\nBacktest interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
