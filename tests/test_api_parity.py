"""
API Parity Tests
================

Tests ensuring SchwabGym's API matches schwab-py signatures, response
formats, parameter ordering, and missing method coverage.

Author: Bryant Clark
"""

import inspect
import json

import pytest

from schwabgym import MockClient
from schwabgym.orders import MockEquities as eq
from schwabgym.orders import MockResponse
from schwabgym.physics import FastExecutionEngine

# ==================== 1. Response Format Tests ====================


class TestResponseFormats:
    """Verify MockResponse properties match httpx.Response interface."""

    def test_get_account_numbers_returns_list(self, client):
        """get_account_numbers() must return a list of dicts, not a single dict."""
        resp = client.get_account_numbers()
        data = resp.json()
        assert isinstance(data, list), "Expected list, got dict"
        assert len(data) == 1
        assert isinstance(data[0], dict)
        assert "accountNumber" in data[0]
        assert "hashValue" in data[0]

    def test_mock_response_text_returns_json_string(self):
        """MockResponse.text must return a JSON-encoded string."""
        payload = {"key": "value", "count": 42}
        resp = MockResponse(payload)
        text = resp.text
        assert isinstance(text, str)
        assert json.loads(text) == payload

    def test_mock_response_ok_property(self):
        """MockResponse.ok is True for status < 400."""
        assert MockResponse({}, 200).ok is True
        assert MockResponse({}, 201).ok is True
        assert MockResponse({}, 301).ok is True
        assert MockResponse({}, 399).ok is True
        assert MockResponse({}, 400).ok is False
        assert MockResponse({}, 401).ok is False
        assert MockResponse({}, 500).ok is False

    def test_mock_response_is_success_property(self):
        """MockResponse.is_success is True only for 2xx status codes."""
        assert MockResponse({}, 200).is_success is True
        assert MockResponse({}, 201).is_success is True
        assert MockResponse({}, 299).is_success is True
        assert MockResponse({}, 199).is_success is False
        assert MockResponse({}, 300).is_success is False
        assert MockResponse({}, 400).is_success is False

    def test_mock_response_is_error_property(self):
        """MockResponse.is_error is True for status >= 400."""
        assert MockResponse({}, 200).is_error is False
        assert MockResponse({}, 301).is_error is False
        assert MockResponse({}, 399).is_error is False
        assert MockResponse({}, 400).is_error is True
        assert MockResponse({}, 404).is_error is True
        assert MockResponse({}, 500).is_error is True

    def test_mock_response_content_returns_bytes(self):
        """MockResponse.content must return bytes (UTF-8 encoded JSON)."""
        payload = {"symbol": "AAPL", "price": 150.5}
        resp = MockResponse(payload)
        content = resp.content
        assert isinstance(content, bytes)
        assert json.loads(content.decode("utf-8")) == payload

    def test_mock_response_text_and_content_consistency(self):
        """text and content must be consistent with each other."""
        payload = [{"a": 1}, {"b": 2}]
        resp = MockResponse(payload)
        assert resp.content == resp.text.encode("utf-8")


# ==================== 2. Parameter Order Tests ====================


class TestParameterOrder:
    """Verify schwab-py parameter ordering: order_id first, account_hash second."""

    def test_get_order_param_order(self, client):
        """get_order(order_id, account_hash) must accept order_id first."""
        order = eq.equity_buy_market("TEST", 5)
        client.place_order(client.account_hash, order)
        order_id = next(iter(client.orders.keys()))

        resp = client.get_order(order_id, client.account_hash)
        assert resp.status_code == 200
        assert resp.json()["orderId"] == order_id

    def test_cancel_order_param_order(self, client):
        """cancel_order(order_id, account_hash) must accept order_id first."""
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        order = eq.equity_buy_limit("TEST", 5, current_price * 0.5)
        client.place_order(client.account_hash, order)
        order_id = next(iter(client.orders.keys()))

        resp = client.cancel_order(order_id, client.account_hash)
        assert resp.status_code == 200
        assert client.orders[order_id]["status"] == "CANCELED"

    def test_get_order_wrong_hash_returns_401(self, client):
        """get_order with wrong account hash returns 401."""
        order = eq.equity_buy_market("TEST", 5)
        client.place_order(client.account_hash, order)
        order_id = next(iter(client.orders.keys()))

        resp = client.get_order(order_id, "WRONG_HASH")
        assert resp.status_code == 401

    def test_cancel_order_wrong_hash_returns_401(self, client):
        """cancel_order with wrong account hash returns 401."""
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        order = eq.equity_buy_limit("TEST", 5, current_price * 0.5)
        client.place_order(client.account_hash, order)
        order_id = next(iter(client.orders.keys()))

        resp = client.cancel_order(order_id, "WRONG_HASH")
        assert resp.status_code == 401


# ==================== 3. Missing Method Tests ====================


class TestMissingMethods:
    """Cover API methods that were missing or stubbed."""

    def test_get_accounts_returns_list(self, client):
        """get_accounts() must return a list wrapping account data."""
        resp = client.get_accounts()
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        acct = data[0]
        assert "securitiesAccount" in acct

    def test_get_accounts_fields_keyword_only(self, client):
        """get_accounts(fields=...) must accept fields as keyword-only."""
        resp = client.get_accounts(fields="positions")
        assert resp.status_code == 200

    def test_get_quote_single_symbol(self, client):
        """get_quote() returns quote data for a single symbol."""
        resp = client.get_quote("TEST")
        assert resp.status_code == 200
        data = resp.json()
        assert "TEST" in data
        assert data["TEST"]["quote"]["symbol"] == "TEST"
        assert data["TEST"]["quote"]["lastPrice"] > 0

    def test_get_transactions_returns_filled_orders(self, client):
        """get_transactions() returns filled trade-like transactions."""
        # Place and fill an order
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))

        resp = client.get_transactions(client.account_hash)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert all(t["status"] == "FILLED" for t in data)

    def test_get_transactions_empty_when_no_fills(self, client):
        """get_transactions() returns empty list when nothing is filled."""
        resp = client.get_transactions(client.account_hash)
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_transactions_filters_symbol_and_type(self, client):
        """Transaction filters should narrow the simulated trade list."""
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 1))
        client.advance_time()
        client.place_order(client.account_hash, eq.equity_buy_market("ALT", 1))

        resp = client.get_transactions(
            client.account_hash, symbol="ALT", transaction_types="TRADE"
        )
        data = resp.json()
        assert len(data) == 1
        assert data[0]["orderLegCollection"][0]["instrument"]["symbol"] == "ALT"

    def test_get_transaction_single(self, client):
        """get_transaction() returns a single filled transaction by ID."""
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))
        order_id = next(iter(client.orders.keys()))

        resp = client.get_transaction(client.account_hash, order_id)
        assert resp.status_code == 200
        assert resp.json()["orderId"] == order_id
        assert resp.json()["status"] == "FILLED"

    def test_get_transaction_not_found(self, client):
        """get_transaction() returns 404 for unfilled or missing transaction."""
        resp = client.get_transaction(client.account_hash, 99999)
        assert resp.status_code == 404

    def test_get_user_preferences(self, client):
        """get_user_preferences() returns a preferences dict."""
        resp = client.get_user_preferences()
        assert resp.status_code == 200
        prefs = resp.json()
        assert "accounts" in prefs
        assert "streamerInfo" in prefs
        assert isinstance(prefs["accounts"], list)
        assert len(prefs["accounts"]) >= 1
        assert prefs["accounts"][0]["accountNumber"] == client.account_number

    def test_preview_order(self, client):
        """preview_order() returns the order spec in a preview envelope."""
        order_spec = eq.equity_buy_market("TEST", 10)
        resp = client.preview_order(client.account_hash, order_spec)
        assert resp.status_code == 200
        data = resp.json()
        assert "orderStrategy" in data
        assert isinstance(data["orderStrategy"], dict)
        assert data["orderStrategy"]["orderType"] == "MARKET"

    def test_preview_order_wrong_hash(self, client):
        """preview_order() with wrong account hash returns 401."""
        order_spec = eq.equity_buy_market("TEST", 10)
        resp = client.preview_order("WRONG_HASH", order_spec)
        assert resp.status_code == 401

    def test_get_orders_for_all_linked_accounts(self, client):
        """get_orders_for_all_linked_accounts() returns orders across accounts."""
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 5))

        resp = client.get_orders_for_all_linked_accounts()
        assert resp.status_code == 200
        orders = resp.json()
        assert isinstance(orders, list)
        assert len(orders) >= 1

    def test_get_orders_for_all_linked_accounts_with_filter(self, client):
        """get_orders_for_all_linked_accounts() supports status filter."""
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 5))

        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        client.place_order(
            client.account_hash, eq.equity_buy_limit("TEST", 5, current_price * 0.5)
        )

        resp = client.get_orders_for_all_linked_accounts(status="FILLED")
        orders = resp.json()
        assert all(o["status"] == "FILLED" for o in orders)

        resp = client.get_orders_for_all_linked_accounts(status="WORKING")
        orders = resp.json()
        assert all(o["status"] == "WORKING" for o in orders)


class TestSyntheticMarketDataMethods:
    """Verify simulator-backed market-data helpers are usable."""

    def test_get_instruments_symbol_search(self, client):
        resp = client.get_instruments("TEST", "symbol-search")
        data = resp.json()

        assert resp.status_code == 200
        assert "TEST" in data
        assert data["TEST"]["symbol"] == "TEST"
        assert data["TEST"]["assetType"] == "EQUITY"

    def test_get_instruments_fundamental(self, client):
        resp = client.get_instruments("TEST", "fundamental")
        data = resp.json()["TEST"]

        assert resp.status_code == 200
        assert "fundamental" in data
        assert data["fundamental"]["marketCap"] > 0

    def test_get_instrument_by_cusip(self, client):
        instruments = client.get_instruments("TEST", "symbol-search").json()
        cusip = instruments["TEST"]["cusip"]

        resp = client.get_instrument_by_cusip(cusip)
        data = resp.json()

        assert resp.status_code == 200
        assert data["symbol"] == "TEST"
        assert data["cusip"] == cusip

    def test_get_market_hours(self, client):
        resp = client.get_market_hours(["equity", "option"])
        data = resp.json()

        assert resp.status_code == 200
        assert "equity" in data
        assert "option" in data
        assert "sessionHours" in next(iter(data["equity"].values()))

    def test_get_option_expiration_chain(self, client):
        resp = client.get_option_expiration_chain("TEST")
        data = resp.json()

        assert resp.status_code == 200
        assert data["symbol"] == "TEST"
        assert len(data["expirationList"]) > 0
        assert "daysToExpiration" in data["expirationList"][0]

    def test_get_option_chain(self, client):
        resp = client.get_option_chain(
            "TEST",
            contract_type="CALL",
            strike_count=1,
            include_underlying_quote=True,
        )
        data = resp.json()

        assert resp.status_code == 200
        assert data["status"] == "SUCCESS"
        assert data["callExpDateMap"]
        assert data["putExpDateMap"] == {}
        assert data["underlying"]["symbol"] == "TEST"

    def test_get_movers(self, multi_asset_client):
        client = multi_asset_client
        test_df = client.price_engine.data["TEST"]
        alt_df = client.price_engine.data["ALT"]
        next_ts = test_df.index[1]

        test_prev = float(test_df.iloc[0]["Close"])
        alt_prev = float(alt_df.iloc[0]["Close"])
        test_df.at[next_ts, "Close"] = round(test_prev * 1.10, 4)
        alt_df.at[next_ts, "Close"] = round(alt_prev * 0.95, 4)
        test_df.at[next_ts, "Volume"] = 500_000
        alt_df.at[next_ts, "Volume"] = 100_000
        client.advance_time()

        resp = client.get_movers("NYSE", sort_order="PERCENT_CHANGE_UP")
        data = resp.json()

        assert resp.status_code == 200
        assert len(data) == 2
        assert data[0]["symbol"] == "TEST"
        assert data[0]["percentChange"] > data[1]["percentChange"]


class TestSignatureParity:
    """Check high-value method signatures against installed schwab-py."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "get_price_history",
            "get_transactions",
            "get_option_chain",
            "get_option_expiration_chain",
            "get_movers",
            "get_market_hours",
            "get_instruments",
            "get_instrument_by_cusip",
        ],
    )
    def test_mock_client_signature_matches_schwab_py(self, method_name):
        schwab_client = pytest.importorskip("schwab.client").Client
        ours = inspect.signature(getattr(MockClient, method_name))
        theirs = inspect.signature(getattr(schwab_client, method_name))

        our_params = [
            (name, param.kind, param.default) for name, param in ours.parameters.items()
        ]
        their_params = [
            (name, param.kind, param.default)
            for name, param in theirs.parameters.items()
        ]

        assert our_params == their_params


# ==================== 4. Stop Order Execution Tests ====================


class TestStopOrderExecution:
    """CRITICAL: Verify stop/stop-limit order trigger and fill logic."""

    @pytest.fixture
    def stop_client(self, sample_data):
        """Client with fast engine for deterministic stop order tests."""
        engine = FastExecutionEngine()
        return MockClient(
            sample_data,
            initial_cash=10000.0,
            execution_engine=engine,
            latency_mode=False,
        )

    def test_sell_stop_triggers_on_price_drop(self, stop_client):
        """Place sell stop, drop price below stop, verify fill."""
        client = stop_client

        # Buy shares first
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))
        assert "TEST" in client.positions
        assert client.positions["TEST"]["quantity"] == 10

        # Get current price and set stop below it
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        stop_price = current_price * 0.95

        # Place sell stop order
        order = eq.equity_sell_stop("TEST", 10, stop_price)
        resp = client.place_order(client.account_hash, order)
        assert resp.status_code == 201

        order_id = max(client.orders.keys())
        assert client.orders[order_id]["status"] == "WORKING"

        # Manipulate next bar so Low drops below stop price
        next_idx = client.df.index[client.current_step + 1]
        client.df.at[next_idx, "Low"] = stop_price * 0.98
        client.df.at[next_idx, "Close"] = stop_price * 0.99

        # Advance time to trigger processing
        client.advance_time()

        # The stop order should have filled
        assert client.orders[order_id]["status"] == "FILLED"
        # Position should be closed
        qty = client.positions.get("TEST", {}).get("quantity", 0)
        assert qty == 0

    def test_sell_stop_no_trigger_when_price_stays_above(self, stop_client):
        """Sell stop should NOT trigger when price stays above stop."""
        client = stop_client

        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))

        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        stop_price = current_price * 0.90  # Stop well below

        order = eq.equity_sell_stop("TEST", 10, stop_price)
        client.place_order(client.account_hash, order)
        order_id = max(client.orders.keys())

        # Keep Low above stop price
        next_idx = client.df.index[client.current_step + 1]
        client.df.at[next_idx, "Low"] = stop_price * 1.05
        client.df.at[next_idx, "High"] = current_price * 1.1

        client.advance_time()

        # Order should still be working
        assert client.orders[order_id]["status"] == "WORKING"
        assert client.positions["TEST"]["quantity"] == 10

    def test_buy_stop_triggers_on_price_rise(self, stop_client):
        """Buy stop triggers when High >= stop_price."""
        client = stop_client
        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        stop_price = current_price * 1.05

        order = eq.equity_buy_stop("TEST", 5, stop_price)
        resp = client.place_order(client.account_hash, order)
        assert resp.status_code == 201

        order_id = max(client.orders.keys())
        assert client.orders[order_id]["status"] == "WORKING"

        # Set High above stop price to trigger
        next_idx = client.df.index[client.current_step + 1]
        client.df.at[next_idx, "High"] = stop_price * 1.02
        client.df.at[next_idx, "Close"] = stop_price * 1.01

        client.advance_time()

        assert client.orders[order_id]["status"] == "FILLED"
        assert client.positions["TEST"]["quantity"] == 5

    def test_stop_limit_sell_trigger_and_limit_logic(self, stop_client):
        """Stop-limit: trigger on price drop, then fill only if limit met."""
        client = stop_client

        # Buy shares
        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))

        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        stop_price = current_price * 0.95
        limit_price = current_price * 0.94  # Limit below stop

        order = eq.equity_sell_stop_limit("TEST", 10, limit_price, stop_price)
        resp = client.place_order(client.account_hash, order)
        assert resp.status_code == 201

        order_id = max(client.orders.keys())
        assert client.orders[order_id]["status"] == "WORKING"

        # Drop Low below stop to trigger, and set range that satisfies limit fill
        # The limit_price must be within [Low, High] for the fill check
        next_idx = client.df.index[client.current_step + 1]
        client.df.at[next_idx, "Low"] = limit_price * 0.99
        client.df.at[next_idx, "High"] = limit_price * 1.02
        client.df.at[next_idx, "Close"] = limit_price
        client.df.at[next_idx, "Volume"] = 1_000_000  # Ensure volume is sufficient

        client.advance_time()

        assert client.orders[order_id]["status"] == "FILLED"
        qty = client.positions.get("TEST", {}).get("quantity", 0)
        assert qty == 0

    def test_stop_limit_sell_triggered_but_limit_not_met(self, stop_client):
        """Stop-limit: triggered by stop price but limit price not met stays working."""
        client = stop_client

        client.place_order(client.account_hash, eq.equity_buy_market("TEST", 10))

        current_price = client.get_quotes("TEST").json()["TEST"]["quote"]["lastPrice"]
        stop_price = current_price * 0.95
        # Set limit ABOVE current range so limit cannot be satisfied
        limit_price = current_price * 1.10

        order = eq.equity_sell_stop_limit("TEST", 10, limit_price, stop_price)
        client.place_order(client.account_hash, order)
        order_id = max(client.orders.keys())

        # Trigger the stop (Low below stop_price) but High stays below limit
        next_idx = client.df.index[client.current_step + 1]
        client.df.at[next_idx, "Low"] = stop_price * 0.98
        client.df.at[next_idx, "High"] = current_price  # below limit_price
        client.df.at[next_idx, "Volume"] = 1_000_000

        client.advance_time()

        # Stop triggered but limit not met -- order remains working
        assert client.orders[order_id]["status"] == "WORKING"
        assert client.positions["TEST"]["quantity"] == 10


# ==================== 5. Price History Convenience Methods ====================


class TestPriceHistoryConvenienceMethods:
    """Verify convenience wrappers request different granularities."""

    def test_get_price_history_every_ten_minutes(self, client):
        """Convenience method returns candles."""
        resp = client.get_price_history_every_ten_minutes("TEST")
        assert resp.status_code == 200
        data = resp.json()
        assert "candles" in data
        assert len(data["candles"]) > 0
        assert "close" in data["candles"][0]

    def test_get_price_history_every_thirty_minutes(self, client):
        """Convenience method returns candles."""
        resp = client.get_price_history_every_thirty_minutes("TEST")
        assert resp.status_code == 200
        data = resp.json()
        assert "candles" in data
        assert len(data["candles"]) > 0

    def test_get_price_history_every_week(self, client):
        """Convenience method returns candles."""
        resp = client.get_price_history_every_week("TEST")
        assert resp.status_code == 200
        data = resp.json()
        assert "candles" in data
        assert len(data["candles"]) > 0

    def test_convenience_methods_request_coarser_bars(self, client):
        """Convenience methods should downsample when the source data allows it."""
        for _ in range(59):
            client.advance_time()

        minute = client.get_price_history_every_minute("TEST").json()["candles"]
        five = client.get_price_history_every_five_minutes("TEST").json()["candles"]
        ten = client.get_price_history_every_ten_minutes("TEST").json()["candles"]
        fifteen = client.get_price_history_every_fifteen_minutes("TEST").json()[
            "candles"
        ]
        thirty = client.get_price_history_every_thirty_minutes("TEST").json()["candles"]
        day = client.get_price_history_every_day("TEST").json()["candles"]
        week = client.get_price_history_every_week("TEST").json()["candles"]

        assert len(minute) > len(five) >= len(ten) >= len(fifteen) >= len(thirty) > 0
        assert len(day) > 0
        assert len(week) > 0
        assert len(day) <= len(minute)
        assert len(week) <= len(day)


# ==================== 6. Keyword-Only Enforcement ====================


class TestKeywordOnlyParams:
    """Verify keyword-only parameters work correctly."""

    def test_get_price_history_keyword_args(self, client):
        """get_price_history accepts keyword-only params after symbol."""
        import datetime

        resp = client.get_price_history(
            "TEST",
            start_datetime=datetime.datetime(2024, 1, 1),
            need_previous_close=True,
        )
        assert resp.status_code == 200
        assert "candles" in resp.json()

    def test_get_price_history_all_keyword_params(self, client):
        """get_price_history accepts all schwab-py keyword params."""
        import datetime

        resp = client.get_price_history(
            "TEST",
            period_type="day",
            period=10,
            frequency_type="minute",
            frequency=5,
            start_datetime=datetime.datetime(2024, 1, 1),
            end_datetime=datetime.datetime(2024, 12, 31),
            need_extended_hours_data=True,
            need_previous_close=False,
        )
        assert resp.status_code == 200

    def test_get_account_fields_keyword_only(self, client):
        """get_account(hash, fields=...) accepts fields as keyword."""
        resp = client.get_account(client.account_hash, fields="positions")
        assert resp.status_code == 200
        assert "securitiesAccount" in resp.json()

    def test_get_quote_fields_keyword_only(self, client):
        """get_quote(symbol, fields=...) accepts fields as keyword."""
        resp = client.get_quote("TEST", fields="quote")
        assert resp.status_code == 200

    def test_get_quotes_fields_keyword_only(self, client):
        """get_quotes(symbols, fields=...) accepts fields as keyword."""
        resp = client.get_quotes("TEST", fields="quote", indicative=False)
        assert resp.status_code == 200
        assert "TEST" in resp.json()
