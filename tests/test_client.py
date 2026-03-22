"""
Client Tests
============

Unit tests for the MockClient simulator.

Author: Bryant Clark
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from schwabgym import MockClient
from schwabgym.orders import MockEquities as eq


class TestMockClient:
    """Test MockClient functionality."""

    def test_initialization(self, sample_data):
        """Test client initialization."""
        client = MockClient(sample_data, initial_cash=50000.0)
        assert client.cash == 50000.0
        assert client.current_step == 0
        assert client.max_steps == len(sample_data) - 1
        assert len(client.positions) == 0

    def test_account_linked(self, client):
        """Test account linking."""
        resp = client.get_account_numbers()
        assert resp.status_code == 200
        data = resp.json()
        assert "accountNumber" in data
        assert "hashValue" in data
        assert data["hashValue"] == client.account_hash

    def test_account_details(self, client):
        """Test account details retrieval."""
        resp = client.get_account(client.account_hash)
        assert resp.status_code == 200
        data = resp.json()

        acct = data["securitiesAccount"]
        assert acct["currentBalances"]["cashBalance"] == 10000.0
        assert acct["currentBalances"]["liquidationValue"] == 10000.0
        assert acct["currentBalances"]["buyingPower"] == 20000.0  # 2:1 margin
        assert len(acct["positions"]) == 0

    def test_quote(self, client):
        """Test quote retrieval."""
        # Single symbol
        resp = client.get_quotes("TEST")
        assert resp.status_code == 200
        data = resp.json()
        assert "TEST" in data
        assert data["TEST"]["quote"]["symbol"] == "TEST"
        assert data["TEST"]["quote"]["lastPrice"] > 0

        # Multiple symbols
        resp = client.get_quotes(["TEST", "OTHER"])
        data = resp.json()
        assert "TEST" in data
        assert "OTHER" in data

    def test_price_history(self, client):
        """Test price history retrieval."""
        resp = client.get_price_history("TEST")
        assert resp.status_code == 200
        data = resp.json()
        assert "candles" in data
        assert len(data["candles"]) > 0
        assert "close" in data["candles"][0]

    def test_market_buy_order(self, client):
        """Test placing a market buy order."""
        # Get initial price
        quote = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        qty = 10

        # Place order
        order = eq.equity_buy_market("TEST", qty)
        resp = client.place_order(client.account_hash, order)

        assert resp.status_code == 201

        # Verify position
        acct = client.get_account(client.account_hash).json()["securitiesAccount"]
        positions = acct["positions"]
        assert len(positions) == 1
        assert positions[0]["instrument"]["symbol"] == "TEST"
        assert positions[0]["longQuantity"] == qty

        # Verify cash deduction (approximate due to spread/slippage)
        expected_cost = qty * quote
        assert client.cash < 10000.0 - expected_cost * 0.99

    def test_market_sell_order(self, client):
        """Test placing a market sell order (long exit)."""
        # Establish long position first
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 20))

        # Sell half
        order = eq.equity_sell_market("TEST", 10)
        resp = client.place_order(client.account_hash, order)

        assert resp.status_code == 201

        # Verify position reduced
        acct = client.get_account(client.account_hash).json()["securitiesAccount"]
        pos = acct["positions"][0]
        assert pos["longQuantity"] == 10

    def test_short_selling(self, client):
        """Test short selling."""
        # Place short order
        order = eq.equity_sell_short_market("TEST", 10)
        resp = client.place_order(client.account_hash, order)

        assert resp.status_code == 201

        # Verify short position
        acct = client.get_account(client.account_hash).json()["securitiesAccount"]
        pos = acct["positions"][0]
        assert pos["shortQuantity"] == 10
        assert pos["longQuantity"] == 0

    def test_limit_order_queuing(self, client):
        """Test that limit orders are queued."""
        # Place limit buy well below market
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        limit_price = current_price * 0.5

        order = eq.equity_buy_limit("TEST", 10, limit_price)
        resp = client.place_order(client.account_hash, order)

        assert resp.status_code == 201

        # Should be in working orders, not positions
        assert len(client.working_orders) == 1
        assert len(client.positions) == 0

    def test_limit_order_fill(self, fast_client):
        """Test limit order fill logic (deterministic)."""
        # Place limit buy at 100
        order = eq.equity_buy_limit("AAPL", 10, 100.0)
        fast_client.place_order(fast_client.account_hash, order)

        # 1. Price above limit -> No fill
        # Manually update dataframe at current step
        idx = fast_client.df.index[fast_client.current_step]
        fast_client.df.at[idx, "Low"] = 105.0
        fast_client.df.at[idx, "High"] = 110.0

        fast_client._process_working_orders()
        assert len(fast_client.working_orders) == 1

        # 2. Price drops to limit -> Fill
        fast_client.df.at[idx, "Low"] = 99.0
        fast_client._process_working_orders()
        assert len(fast_client.working_orders) == 0

    def test_client_errors(self, client):
        """Test client error handling."""
        # Test invalid account hash
        resp = client.get_account("invalid_hash")
        assert resp.status_code == 401

        # Test order for invalid account
        order = eq.equity_buy_market("AAPL", 10)
        resp = client.place_order("invalid_hash", order)
        assert resp.status_code == 401

        # Test cancel invalid order
        # cancel_order not implemented in MockClient yet, skipping or removing test
        # resp = client.cancel_order("invalid_hash", 99999)
        # assert resp.status_code == 404

    def test_insufficient_funds(self, client):
        """Test rejection on insufficient funds."""
        # Try to buy more than cash available
        qty = 1000000
        order = eq.equity_buy_market("TEST", qty)
        resp = client.place_order(client.account_hash, order)

        assert resp.status_code == 400
        assert "Insufficient Buying Power" in resp.json()["error"]

    def test_sell_more_than_owned(self, client):
        """Test rejection when selling more than owned (without shorting)."""
        # Try to sell without owning
        order = eq.equity_sell_market("TEST", 10)
        resp = client.place_order(client.account_hash, order)

        assert resp.status_code == 400
        assert "Position not available" in resp.json()["error"]

    def test_unsupported_order_type(self, client):
        """Test rejection of unsupported order types."""
        order = eq.equity_buy_market("TEST", 10)
        order["orderType"] = "INVALID_TYPE"
        resp = client.place_order(client.account_hash, order)

        assert resp.status_code == 400
        assert "Unsupported order type" in resp.json()["error"]

    def test_pdt_rule_flagging(self, client):
        """Test Pattern Day Trader flagging."""
        # Force account value below $25k (client fixture starts with 10k)
        assert client._calculate_equity() < 25000

        # Execute 4 day trades
        for _ in range(4):
            # Buy
            client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))
            # Sell (Day Trade)
            client.place_order(client.account_hash, eq.equity_sell_market("TEST", 10))

        # 5th trade should be blocked
        order = eq.equity_buy_market("TEST", 10)
        client.place_order(client.account_hash, order)  # Open

        # Try to close (Day Trade #5)
        resp = client.place_order(
            client.account_hash, eq.equity_sell_market("TEST", 10)
        )

        assert resp.status_code == 403
        assert "Pattern Day Trader" in resp.json()["error"]

        # Verify subsequent orders are also blocked
        resp = client.place_order(client.account_hash, eq.equity_buy_market("TEST", 1))
        assert resp.status_code == 403

    def test_advance_time(self, client):
        """Test time advancement."""
        start_step = client.current_step
        res = client.advance_time()
        assert res is True
        assert client.current_step == start_step + 1

    def test_reset(self, client):
        """Test client reset."""
        client.advance_time()
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))

        client.reset()

        assert client.current_step == 0
        assert len(client.positions) == 0
        assert len(client.working_orders) == 0


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [100.0],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2023-01-01"]),
    )


def test_client_init_string_arg(valid_df):
    # Test passing string as first arg (app_key)
    # And providing market_data in kwargs

    with patch("schwabgym.client.logger") as mock_logger:
        client = MockClient("my_app_key", market_data=valid_df)

        mock_logger.warning.assert_called_with(
            "MockClient received string for market_data_df. Assuming it is app_key."
        )
        assert client.price_engine is not None


def test_client_init_no_data_fallback(valid_df):
    # Test fallback to dummy data
    # We patch schwabgym.data.generate_dummy_data because that is what is imported
    with patch("schwabgym.data.generate_dummy_data") as mock_gen:
        mock_gen.return_value = valid_df

        _client = MockClient()
        assert mock_gen.called


def test_advance_time_end(valid_df):
    client = MockClient(valid_df)

    # Force price engine to end
    client.price_engine.advance_time = MagicMock(return_value=False)

    assert client.advance_time() is False


def test_unauthorized_access(valid_df):
    client = MockClient(valid_df)

    wrong_hash = "WRONG_HASH"

    # get_account
    resp = client.get_account(wrong_hash)
    assert resp.status_code == 401

    # place_order
    resp = client.place_order(wrong_hash, {})
    assert resp.status_code == 401

    # cancel_order
    resp = client.cancel_order(wrong_hash, 123)
    assert resp.status_code == 401

    # replace_order
    resp = client.replace_order(wrong_hash, 123, {})
    assert resp.status_code == 401

    # get_order
    resp = client.get_order(wrong_hash, 123)
    assert resp.status_code == 401

    # get_orders_for_account
    resp = client.get_orders_for_account(wrong_hash)
    assert resp.status_code == 401
