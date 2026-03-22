"""
Tests for Order Manager
=======================
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from schwabgym.account import Account
from schwabgym.order_manager import OrderManager
from schwabgym.orders import MockEquities as eq
from schwabgym.physics import RealisticExecutionEngine
from schwabgym.prices import PriceEngine


@pytest.fixture
def components():
    # Setup data
    dates = pd.date_range(start="2023-01-01", periods=10, freq="1min")
    data = {
        "Open": [100.0] * 10,
        "High": [105.0] * 10,
        "Low": [95.0] * 10,
        "Close": [100.0] * 10,
        "Volume": [10000] * 10,
        "Volatility": [0.01] * 10,
    }
    df = pd.DataFrame(data, index=dates)

    price_engine = PriceEngine(df)
    account = Account(initial_cash=50000.0)
    exec_engine = RealisticExecutionEngine()

    # Legacy tests assume latency_mode=False
    manager = OrderManager(account, price_engine, exec_engine, latency_mode=False)
    return manager, account, price_engine


class TestOrderManager:
    def test_place_market_order(self, components):
        manager, account, _ = components
        order = eq.equity_buy_market("TEST", 10)

        resp = manager.place_order(order)
        assert resp.status_code == 201

        # Verify filled
        assert len(manager.orders) == 1
        order_id = next(iter(manager.orders.keys()))
        assert manager.orders[order_id]["status"] == "FILLED"

        # Verify account updated
        assert account.positions["TEST"]["quantity"] == 10

    def test_place_limit_order(self, components):
        manager, _, _ = components
        order = eq.equity_buy_limit("TEST", 10, 90.0)

        resp = manager.place_order(order)
        assert resp.status_code == 201

        # Verify queued
        assert len(manager.working_orders) == 1
        assert next(iter(manager.orders.values()))["status"] == "WORKING"

    def test_limit_order_execution(self, components):
        manager, account, prices = components

        # Place limit buy @ 96.0 (Market Low is 95.0, so should fill)
        order = eq.equity_buy_limit("TEST", 10, 96.0)
        manager.place_order(order)

        # Process
        manager.process_working_orders()

        # Verify filled
        assert len(manager.working_orders) == 0
        assert next(iter(manager.orders.values()))["status"] == "FILLED"
        assert account.positions["TEST"]["quantity"] == 10

    def test_cancel_order(self, components):
        manager, _, _ = components
        order = eq.equity_buy_limit("TEST", 10, 90.0)
        manager.place_order(order)
        order_id = manager.working_orders[0]["orderId"]

        resp = manager.cancel_order(order_id)
        assert resp.status_code == 200
        assert manager.orders[order_id]["status"] == "CANCELED"
        assert len(manager.working_orders) == 0

    def test_replace_order(self, components):
        manager, _, _ = components
        order1 = eq.equity_buy_limit("TEST", 10, 90.0)
        manager.place_order(order1)
        order_id1 = manager.working_orders[0]["orderId"]

        order2 = eq.equity_buy_limit("TEST", 10, 92.0)
        resp = manager.replace_order(order_id1, order2)
        assert resp.status_code == 201

        assert manager.orders[order_id1]["status"] == "CANCELED"
        assert len(manager.working_orders) == 1
        assert float(manager.working_orders[0]["price"]) == 92.0

    def test_insufficient_funds_rejection(self, components):
        manager, _, _ = components
        order = eq.equity_buy_market("TEST", 1000000)  # Huge order

        resp = manager.place_order(order)
        assert resp.status_code == 400

        # Verify rejected status
        order_id = next(iter(manager.orders.keys()))
        # Depending on impl, it might be in orders dict or not if 400 immediately
        # But our impl puts it in orders then returns 400 if execution fails immediately in latency_mode=False
        if order_id in manager.orders:
            assert manager.orders[order_id]["status"] == "REJECTED"


@pytest.fixture
def order_manager_setup():
    # Setup minimal dependencies
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [1000, 1000, 1000],
        },
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )

    price_engine = PriceEngine(df)
    account = Account(initial_cash=10000.0)
    execution_engine = RealisticExecutionEngine()

    manager = OrderManager(account, price_engine, execution_engine, latency_mode=False)
    return manager


def test_cancel_non_existent_order(order_manager_setup):
    manager = order_manager_setup
    resp = manager.cancel_order(9999)
    assert resp.status_code == 404
    assert "Order not found" in resp.json()["error"]


def test_cancel_already_filled_order(order_manager_setup):
    manager = order_manager_setup

    # Place market order (fills immediately in non-latency mode)
    order = {
        "orderType": "MARKET",
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }
    resp = manager.place_order(order)
    order_id = int(resp.headers["Location"].split("/")[-1])

    # Try to cancel
    cancel_resp = manager.cancel_order(order_id)
    assert cancel_resp.status_code == 400
    assert "cannot cancel" in cancel_resp.json()["error"]


def test_replace_order_flow(order_manager_setup):
    manager = order_manager_setup
    # Switch to latency mode to keep order working
    manager.latency_mode = True
    manager.latency_steps = 1

    # Place limit order
    order = {
        "orderType": "LIMIT",
        "price": 90.0,  # Below low, won't fill
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }
    resp = manager.place_order(order)
    order_id = int(resp.headers["Location"].split("/")[-1])

    # Advance time to activate it
    manager.price_engine.advance_time()
    manager.process_working_orders()

    assert manager.orders[order_id]["status"] == "WORKING"

    # Replace it
    new_order_spec = order.copy()
    new_order_spec["price"] = 91.0

    replace_resp = manager.replace_order(order_id, new_order_spec)
    assert replace_resp.status_code == 201

    # Old order should be canceled
    assert manager.orders[order_id]["status"] == "CANCELED"

    # New order should be pending/working
    new_order_id = int(replace_resp.headers["Location"].split("/")[-1])
    assert new_order_id != order_id
    assert new_order_id in manager.orders


def test_strict_limit_order_logic(order_manager_setup):
    manager = order_manager_setup
    manager.strict_limit_orders = True
    manager.latency_mode = False

    # Current step 0: Low is 95.0
    # Place limit buy at 95.0
    # Strict mode: Needs to cross or gap.
    # If Low <= Limit, it MIGHT fill if it crossed.
    # The logic in code: if low_p < limit_price (crossed below) -> fill.
    # If low_p == limit_price -> NO fill in strict mode (usually).

    # Let's test the "touch" vs "strict" difference

    # Case 1: Limit = 95.0 (Low is 95.0).
    # Strict: 95.0 < 95.0 is False. Should NOT fill.

    order = {
        "orderType": "LIMIT",
        "price": 95.0,
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }

    manager.place_order(order)  # Adds to working
    manager.process_working_orders()

    # Verify it is still WORKING (did not fill)
    # Note: We need to find the order ID.
    working = [o for o in manager.working_orders if o["price"] == 95.0]
    assert len(working) == 1
    assert working[0]["status"] == "WORKING"

    # Case 2: Limit = 96.0 (Low is 95.0).
    # Strict: 95.0 < 96.0 is True. Should fill.
    order2 = order.copy()
    order2["price"] = 96.0
    manager.place_order(order2)
    manager.process_working_orders()

    # Should be filled
    # We check history because it's removed from working_orders
    filled = [
        o
        for o in manager.orders.values()
        if o.get("price") == 96.0 and o["status"] == "FILLED"
    ]
    assert len(filled) == 1


def test_market_order_rejection_insufficient_funds(order_manager_setup):
    manager = order_manager_setup
    manager.account.cash = 10.0  # Low cash

    order = {
        "orderType": "MARKET",
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 100,  # Cost ~10,000
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }

    resp = manager.place_order(order)
    assert resp.status_code == 400
    assert (
        "rejected" in resp.json()["error"].lower()
        or "buying power" in resp.json()["error"].lower()
    )


def test_cancel_pending_order(order_manager_setup):
    manager = order_manager_setup
    manager.latency_mode = True
    manager.latency_steps = 5

    order = {
        "orderType": "MARKET",
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }
    resp = manager.place_order(order)
    order_id = int(resp.headers["Location"].split("/")[-1])

    # It should be in pending
    assert len(manager.pending_orders) == 1

    # Cancel it
    cancel_resp = manager.cancel_order(order_id)
    assert cancel_resp.status_code == 200
    assert len(manager.pending_orders) == 0

    # Verify status in history (though pending orders might not be in main history dict depending on impl,
    # but place_order puts them there)
    assert manager.orders[order_id]["status"] == "CANCELED"


def test_pending_order_remains_pending(order_manager_setup):
    manager = order_manager_setup
    manager.latency_mode = True
    manager.latency_steps = 10

    order = {
        "orderType": "MARKET",
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }
    manager.place_order(order)

    # Advance only 1 step
    manager.price_engine.advance_time()
    manager.process_working_orders()

    # Should still be pending
    assert len(manager.pending_orders) == 1


def test_activated_market_order(order_manager_setup):
    manager = order_manager_setup
    manager.latency_mode = True
    manager.latency_steps = 1

    order = {
        "orderType": "MARKET",
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }
    resp = manager.place_order(order)
    order_id = int(resp.headers["Location"].split("/")[-1])

    # Advance to activate
    manager.price_engine.advance_time()
    manager.process_working_orders()

    assert manager.orders[order_id]["status"] == "FILLED"


def test_strict_limit_sell_logic(order_manager_setup):
    manager = order_manager_setup
    manager.strict_limit_orders = True
    manager.latency_mode = False

    # High is 105.0

    # Case 1: Limit Sell at 106.0 (High 105.0 < 106.0). No fill.
    order = {
        "orderType": "LIMIT",
        "price": 106.0,
        "orderLegCollection": [
            {
                "instruction": "SELL",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }
    manager.account.positions["DEFAULT"] = {
        "quantity": 100,
        "avgPrice": 100.0,
        "assetType": "EQUITY",
    }

    manager.place_order(order)
    manager.process_working_orders()

    working = [o for o in manager.working_orders if o["price"] == 106.0]
    assert len(working) == 1

    # Case 2: Limit Sell at 104.0 (High 105.0 > 104.0). Fill.
    order2 = order.copy()
    order2["price"] = 104.0
    manager.place_order(order2)
    manager.process_working_orders()

    filled = [
        o
        for o in manager.orders.values()
        if o.get("price") == 104.0 and o["status"] == "FILLED"
    ]
    assert len(filled) == 1


def test_limit_order_execution_exception(order_manager_setup):
    manager = order_manager_setup
    manager.latency_mode = False

    # Mock execute_trade_leg to raise exception
    manager._execute_trade_leg = MagicMock(side_effect=Exception("Simulated Failure"))

    order = {
        "orderType": "LIMIT",
        "price": 102.0,  # Should fill (High 105)
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": 1,
                "instrument": {"symbol": "DEFAULT", "assetType": "EQUITY"},
            }
        ],
    }

    manager.place_order(order)
    manager.process_working_orders()

    # Should be rejected/canceled due to exception
    # Code says: order["status"] = "REJECTED"
    rejected = [o for o in manager.orders.values() if o["status"] == "REJECTED"]
    assert len(rejected) >= 1
