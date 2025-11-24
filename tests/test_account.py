"""
Tests for Account Logic
=======================
"""

import pytest
import datetime
from schwabgym.account import Account

@pytest.fixture
def account():
    return Account(initial_cash=30000.0)

def mock_price_lookup(symbol):
    return 100.0

class TestAccount:
    def test_initialization(self, account):
        assert account.cash == 30000.0
        assert len(account.positions) == 0

    def test_calculate_equity(self, account):
        # With cash only
        assert account.calculate_equity(mock_price_lookup) == 30000.0

        # With position
        account.positions["TEST"] = {"quantity": 10, "avgPrice": 90.0, "assetType": "EQUITY"}
        # Equity = 30000 + (10 * 100) = 31000
        assert account.calculate_equity(mock_price_lookup) == 31000.0

        # With short position
        account.positions["SHORT"] = {"quantity": -10, "avgPrice": 110.0, "assetType": "EQUITY"}
        # Equity = 31000 - (10 * 100) = 30000
        assert account.calculate_equity(mock_price_lookup) == 30000.0

    def test_execute_trade_buy(self, account):
        trade_date = datetime.date(2023, 1, 1)
        bp = 60000.0

        account.execute_trade("AAPL", 10, 150.0, "BUY", "EQUITY", trade_date, bp)

        assert account.cash == 30000.0 - 1500.0
        assert account.positions["AAPL"]["quantity"] == 10
        assert account.positions["AAPL"]["avgPrice"] == 150.0

    def test_execute_trade_insufficient_bp(self, account):
        trade_date = datetime.date(2023, 1, 1)
        bp = 100.0 # Low BP

        with pytest.raises(ValueError, match="Insufficient Buying Power"):
            account.execute_trade("AAPL", 10, 150.0, "BUY", "EQUITY", trade_date, bp)

    def test_execute_trade_sell(self, account):
        trade_date = datetime.date(2023, 1, 1)
        bp = 60000.0

        # Buy first
        account.execute_trade("AAPL", 10, 150.0, "BUY", "EQUITY", trade_date, bp)

        # Sell half
        account.execute_trade("AAPL", 5, 160.0, "SELL", "EQUITY", trade_date, bp)

        assert account.positions["AAPL"]["quantity"] == 5
        assert account.cash > (30000.0 - 1500.0 + 800.0 - 1.0) # approx check for fees

    def test_execute_trade_short(self, account):
        trade_date = datetime.date(2023, 1, 1)
        bp = 60000.0

        account.execute_trade("AAPL", 10, 150.0, "SELL_SHORT", "EQUITY", trade_date, bp)

        assert account.positions["AAPL"]["quantity"] == -10
        assert account.cash > 30000.0 + 1500.0 - 1.0 # Cash increases on short sale

    def test_pdt_rule(self, account):
        # Set equity low
        account.cash = 10000.0
        equity = 10000.0
        today = datetime.date(2023, 1, 1)

        # Open positions today
        account.opened_positions_today.add("AAPL")

        # Simulate 3 day trades already
        account.day_trades.extend([today, today, today])

        # 4th day trade (Closing AAPL)
        with pytest.raises(ValueError, match="Pattern Day Trader"):
            account.check_pdt_rule("AAPL", "SELL", 10, equity, today)

    def test_on_new_day(self, account):
        account.opened_positions_today.add("AAPL")
        account.on_new_day()
        assert len(account.opened_positions_today) == 0
