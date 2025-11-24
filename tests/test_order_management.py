"""
Additional Client Tests for Order Management
============================================

Unit tests for the improved MockClient functionalities.
"""

from schwabgym import MockClient
from schwabgym.orders import MockEquities as eq
import pytest

class TestOrderManagement:
    """Test Order Management functionalities."""

    def test_cancel_working_order(self, client):
        """Test cancelling a working order."""
        # Place limit buy well below market
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        limit_price = current_price * 0.5

        order = eq.equity_buy_limit("TEST", 10, limit_price)
        resp = client.place_order(client.account_hash, order)
        assert resp.status_code == 201

        # Get Order ID from location header (schwab style) or just rely on tracking
        # For simplicity in test, let's grab it from working_orders
        assert len(client.working_orders) == 1
        order_id = client.working_orders[0]["orderId"]

        # Cancel the order
        resp = client.cancel_order(client.account_hash, order_id)
        assert resp.status_code == 200
        assert resp.json()["orderId"] == order_id

        # Verify it is removed from working_orders
        assert len(client.working_orders) == 0

        # Verify status is CANCELED in history
        order_info = client.get_order(client.account_hash, order_id).json()
        assert order_info["status"] == "CANCELED"
        assert order_info["cancelTime"] is not None

    def test_cancel_filled_order(self, client):
        """Test cancelling a filled order (should fail)."""
        # Place market order (fills immediately)
        order = eq.equity_buy_market("TEST", 10)
        resp = client.place_order(client.account_hash, order)
        assert resp.status_code == 201

        # Get ID
        orders_resp = client.get_orders_for_account(client.account_hash)
        orders = orders_resp.json()
        assert len(orders) > 0
        order_id = orders[-1]["orderId"]

        # Try to cancel
        resp = client.cancel_order(client.account_hash, order_id)
        assert resp.status_code == 400
        assert "already FILLED" in resp.json()["error"]

    def test_cancel_nonexistent_order(self, client):
        """Test cancelling an unknown order."""
        resp = client.cancel_order(client.account_hash, 999999)
        assert resp.status_code == 404

    def test_replace_order(self, client):
        """Test replacing an order."""
        # Place limit order
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        limit_price = current_price * 0.5

        order1 = eq.equity_buy_limit("TEST", 10, limit_price)
        client.place_order(client.account_hash, order1)

        order_id1 = client.working_orders[0]["orderId"]

        # Replace with slightly higher price
        order2 = eq.equity_buy_limit("TEST", 10, limit_price * 1.1)
        resp = client.replace_order(client.account_hash, order_id1, order2)
        assert resp.status_code == 201

        # Verify old order is canceled
        old_order = client.get_order(client.account_hash, order_id1).json()
        assert old_order["status"] == "CANCELED"

        # Verify new order is working
        assert len(client.working_orders) == 1
        new_order = client.working_orders[0]
        assert new_order["orderId"] != order_id1

        # Compare prices (handle float vs string parity)
        expected_price = limit_price * 1.1
        actual_price = float(new_order["price"])
        assert abs(actual_price - expected_price) < 0.0001

    def test_get_orders(self, client):
        """Test get_orders_for_account."""
        # Place a few orders
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10)) # Filled

        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        client.place_order(client.account_hash, eq.equity_buy_limit("TEST", 5, current_price * 0.5)) # Working

        resp = client.get_orders_for_account(client.account_hash)
        assert resp.status_code == 200
        orders = resp.json()
        assert len(orders) == 2

        # Test filtering
        resp = client.get_orders_for_account(client.account_hash, status="FILLED")
        orders = resp.json()
        assert len(orders) == 1
        assert orders[0]["status"] == "FILLED"

        resp = client.get_orders_for_account(client.account_hash, status="WORKING")
        orders = resp.json()
        assert len(orders) == 1
        assert orders[0]["status"] == "WORKING"

    def test_dynamic_spread(self, client):
        """Test that spread is dynamic based on volatility."""
        # Mock volatility in DF
        # client.df has Volatility column. Let's find a step with known volatility or modify it.
        # Assuming sample data has some volatility.

        # Get quotes twice with different volatilities simulated by modifying DF
        # (This relies on client internals, but it's a test)
        idx = client.df.index[client.current_step]

        # Low volatility
        client.df.at[idx, "Volatility"] = 0.001
        resp1 = client.get_quotes("TEST").json()["TEST"]["quote"]
        spread1 = resp1["askPrice"] - resp1["bidPrice"]

        # High volatility
        client.df.at[idx, "Volatility"] = 0.05
        resp2 = client.get_quotes("TEST").json()["TEST"]["quote"]
        spread2 = resp2["askPrice"] - resp2["bidPrice"]

        assert spread2 > spread1

    def test_limit_order_lifecycle(self, client):
        """Test that a limit order transitions from WORKING to FILLED."""
        # 1. Place limit order below current price (BUY)
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        limit_price = current_price * 0.9

        # Ensure the DF has a low enough Low price in the NEXT step to trigger fill
        idx = client.df.index[client.current_step + 1]
        client.df.at[idx, "Low"] = limit_price * 0.99
        client.df.at[idx, "High"] = current_price  # doesn't matter for buy

        order = eq.equity_buy_limit("TEST", 10, limit_price)
        client.place_order(client.account_hash, order)

        # Get ID
        order_id = client.working_orders[0]["orderId"]

        # 2. Verify status is WORKING
        order_status = client.get_order(client.account_hash, order_id).json()
        assert order_status["status"] == "WORKING"
        assert order_status["closeTime"] is None

        # 3. Advance time to trigger fill
        client.advance_time()

        # 4. Verify status is FILLED
        order_status = client.get_order(client.account_hash, order_id).json()
        assert order_status["status"] == "FILLED"
        assert order_status["closeTime"] is not None

        # 5. Verify removed from working
        assert len(client.working_orders) == 0
