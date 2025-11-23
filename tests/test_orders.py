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
        with pytest.raises(Exception):
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


class TestMockOptions:
    """Test option order builders."""

    def test_buy_to_open(self):
        symbol = "AAPL  230616C00170000"
        order = MockOptions.option_buy_to_open_market(symbol, 1)
        assert order["orderType"] == "MARKET"
        assert order["orderLegCollection"][0]["instruction"] == "BUY_TO_OPEN"
        assert order["orderLegCollection"][0]["instrument"]["assetType"] == "OPTION"

    def test_sell_to_close(self):
        symbol = "AAPL  230616C00170000"
        order = MockOptions.option_sell_to_close_market(symbol, 1)
        assert order["orderLegCollection"][0]["instruction"] == "SELL_TO_CLOSE"
