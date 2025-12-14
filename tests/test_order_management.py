"""
Tests for Order Management (Client-Level)
=========================================
"""

import datetime

import pytest

from schwabgym.orders import MockEquities as eq


class TestOrderManagement:
    def test_place_market_order_updates_status(self, client):
        """Test that placing a market order updates the order status."""
        order = eq.equity_buy_market("TEST", 10)
        client.place_order(client.account_hash, order)

        # Should be filled immediately (default engine)
        assert client.orders[1000]["status"] == "FILLED"

        # Verify executed quantity
        assert client.positions["TEST"]["quantity"] == 10

    def test_place_limit_order_updates_status(self, client):
        """Test that limit orders stay WORKING."""
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        limit_price = current_price * 0.9  # Well below market

        order = eq.equity_buy_limit("TEST", 10, limit_price)
        client.place_order(client.account_hash, order)

        assert client.orders[1000]["status"] == "WORKING"
        assert "TEST" not in client.positions

    def test_cancel_working_order(self, client):
        """Test cancelling a working order."""
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        client.place_order(
            client.account_hash, eq.equity_buy_limit("TEST", 10, current_price * 0.5)
        )
        order_id = list(client.orders.keys())[0]

        resp = client.cancel_order(client.account_hash, order_id)
        assert resp.status_code == 200

        status = client.get_order(client.account_hash, order_id).json()["status"]
        assert status == "CANCELED"

    def test_cancel_filled_order(self, client):
        """Test that filled orders cannot be cancelled."""
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))
        order_id = list(client.orders.keys())[0]

        resp = client.cancel_order(client.account_hash, order_id)
        assert resp.status_code == 400

    def test_replace_order(self, client):
        """Test replacing an order."""
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        client.place_order(
            client.account_hash, eq.equity_buy_limit("TEST", 10, current_price * 0.5)
        )
        order_id = list(client.orders.keys())[0]

        new_order = eq.equity_buy_limit("TEST", 20, current_price * 0.6)
        resp = client.replace_order(client.account_hash, order_id, new_order)

        assert resp.status_code == 201

        # Old order cancelled
        assert client.orders[order_id]["status"] == "CANCELED"

        # New order working
        new_id = int(resp.headers["Location"].split("/")[-1])
        assert client.orders[new_id]["status"] == "WORKING"
        assert client.orders[new_id]["orderLegCollection"][0]["quantity"] == 20

    def test_get_orders(self, client):
        """Test get_orders_for_account."""
        # Place a few orders
        client.place_order(
            client.account_hash, eq.equity_buy_market("TEST", 10)
        )  # Filled

        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        client.place_order(
            client.account_hash, eq.equity_buy_limit("TEST", 5, current_price * 0.5)
        )  # Working

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
        # Note: In legacy mode (used by fixtures), order is immediately WORKING
        order_id = client.working_orders[0]["orderId"]

        # 2. Verify status is WORKING
        order_status = client.get_order(client.account_hash, order_id).json()
        assert order_status["status"] == "WORKING"

        # 3. Advance time
        client.advance_time()

        # 4. Verify status is FILLED
        order_status = client.get_order(client.account_hash, order_id).json()
        assert order_status["status"] == "FILLED"
