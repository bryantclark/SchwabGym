"""
Order Builder Tests
===================

Unit tests for MockEquities, MockOptions, and MockResponse.

Author: Bryant Clark
"""

import pytest

from schwabgym.orders import MockEquities, MockOptions, MockResponse


class TestMockResponse:
    """Test MockResponse functionality."""

    def test_json_response(self):
        """Test JSON data retrieval."""
        data = {"key": "value"}
        resp = MockResponse(data)
        assert resp.json() == data
        assert resp.status_code == 200

    def test_status_code(self):
        """Test status code handling."""
        resp = MockResponse({}, status_code=404)
        assert resp.status_code == 404

    def test_raise_for_status(self):
        """Test raise_for_status method."""
        resp = MockResponse({}, status_code=200)
        # Should not raise
        resp.raise_for_status()

        resp_error = MockResponse({"error": "bad"}, status_code=400)
        with pytest.raises(RuntimeError):
            resp_error.raise_for_status()


class TestMockEquities:
    """Test equity order builders."""

    def test_market_buy(self):
        order = MockEquities.equity_buy_market("AAPL", 100)
        assert order["orderType"] == "MARKET"
        assert order["orderLegCollection"][0]["instruction"] == "BUY"
        assert order["orderLegCollection"][0]["quantity"] == 100
        assert order["orderLegCollection"][0]["instrument"]["symbol"] == "AAPL"

    def test_market_sell(self):
        order = MockEquities.equity_sell_market("AAPL", 50)
        assert order["orderType"] == "MARKET"
        assert order["orderLegCollection"][0]["instruction"] == "SELL"

    def test_limit_buy(self):
        order = MockEquities.equity_buy_limit("AAPL", 100, 150.50)
        assert order["orderType"] == "LIMIT"
        assert order["price"] == "150.5000"

    def test_stop_sell(self):
        order = MockEquities.equity_sell_stop("AAPL", 100, 140.00)
        assert order["orderType"] == "STOP"
        assert order["stopPrice"] == "140.0000"

    def test_equity_sell_limit(self):
        order = MockEquities.equity_sell_limit("AAPL", 10, 150.0)
        assert order["orderType"] == "LIMIT"
        assert order["price"] == "150.0000"
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "SELL"

    def test_equity_buy_stop(self):
        order = MockEquities.equity_buy_stop("AAPL", 10, 150.0)
        assert order["orderType"] == "STOP"
        assert order["stopPrice"] == "150.0000"
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "BUY"


class TestMockOptions:
    """Test option order builders."""

    def test_option_buy_to_open_market(self):
        order = MockOptions.option_buy_to_open_market("AAPL_230616C150", 1)
        assert order["orderType"] == "MARKET"
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "BUY_TO_OPEN"
        assert leg["instrument"]["assetType"] == "OPTION"

    def test_option_sell_to_close_market(self):
        order = MockOptions.option_sell_to_close_market("AAPL_230616C150", 1)
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "SELL_TO_CLOSE"

    def test_option_sell_to_open_market(self):
        order = MockOptions.option_sell_to_open_market("AAPL_230616C150", 1)
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "SELL_TO_OPEN"

    def test_option_buy_to_close_market(self):
        order = MockOptions.option_buy_to_close_market("AAPL_230616C150", 1)
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "BUY_TO_CLOSE"

    def test_option_with_price(self):
        # Test internal _base_option_order with price
        order = MockOptions._base_option_order(
            "AAPL_230616C150", 1, "BUY_TO_OPEN", "LIMIT", price=5.5
        )
        assert order["price"] == "5.50"
