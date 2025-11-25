"""
Tests for Order Manager
=======================
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from schwabgym.order_manager import OrderManager
from schwabgym.account import Account
from schwabgym.prices import PriceEngine
from schwabgym.physics import RealisticExecutionEngine
from schwabgym.orders import MockEquities as eq

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
        "Volatility": [0.01] * 10
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
        order_id = list(manager.orders.keys())[0]
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
        assert list(manager.orders.values())[0]["status"] == "WORKING"

    def test_limit_order_execution(self, components):
        manager, account, prices = components

        # Place limit buy @ 96.0 (Market Low is 95.0, so should fill)
        order = eq.equity_buy_limit("TEST", 10, 96.0)
        manager.place_order(order)

        # Process
        manager.process_working_orders()

        # Verify filled
        assert len(manager.working_orders) == 0
        assert list(manager.orders.values())[0]["status"] == "FILLED"
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
        order = eq.equity_buy_market("TEST", 1000000) # Huge order

        resp = manager.place_order(order)
        assert resp.status_code == 400

        # Verify rejected status
        order_id = list(manager.orders.keys())[0]
        # Depending on impl, it might be in orders dict or not if 400 immediately
        # But our impl puts it in orders then returns 400 if execution fails immediately in latency_mode=False
        if order_id in manager.orders:
             assert manager.orders[order_id]["status"] == "REJECTED"
