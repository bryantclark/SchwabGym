"""
SchwabGym Order Builders
========================

Compatible order construction matching schwab.orders.equities and
schwab.orders.options.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import copy
import datetime
from collections.abc import Iterator, MutableMapping
from typing import Any, NoReturn


def _coerce_enum_value(value):
    if hasattr(value, "value"):
        return value.value
    return value


def _format_price(value: Any, *, option_style: bool) -> str:
    if isinstance(value, str):
        return value

    decimals = 2 if option_style else 4
    return f"{float(value):.{decimals}f}"


class MockOrderBuilder(MutableMapping[str, Any]):
    """Lightweight fluent order builder compatible with schwab-py's shape."""

    def __init__(self, payload: dict[str, Any] | None = None):
        self._payload: dict[str, Any] = payload.copy() if payload is not None else {}
        self._enforce_enums = True

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._payload[key] = value

    def __delitem__(self, key: str) -> None:
        del self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def __repr__(self) -> str:
        return f"MockOrderBuilder({self._payload!r})"

    def build(self) -> dict[str, Any]:
        """Return the built JSON order payload."""
        payload = copy.deepcopy(self._payload)
        children = payload.get("childOrderStrategies")
        if children is not None:
            payload["childOrderStrategies"] = [
                child.build() if hasattr(child, "build") else copy.deepcopy(child)
                for child in children
            ]
        return payload

    def convert_enum(self, value, _enum_type=None):
        return _coerce_enum_value(value)

    def convert_enum_iterable(self, values, _enum_type=None):
        return [self.convert_enum(value, _enum_type) for value in values]

    def set_enforce_enums(self, enforce: bool) -> "MockOrderBuilder":
        self._enforce_enums = enforce
        return self

    def type_error(self, value, expected_type) -> NoReturn:
        raise ValueError(f"Expected {expected_type}, got {value!r}")

    def _set(self, key: str, value: Any) -> "MockOrderBuilder":
        self._payload[key] = _coerce_enum_value(value)
        return self

    def _clear(self, key: str) -> "MockOrderBuilder":
        self._payload.pop(key, None)
        return self

    def _uses_option_pricing(self) -> bool:
        legs = self._payload.get("orderLegCollection", [])
        return any(
            leg.get("instrument", {}).get("assetType") == "OPTION" for leg in legs
        ) or self._payload.get("orderType") in {"NET_DEBIT", "NET_CREDIT"}

    def set_session(self, session) -> "MockOrderBuilder":
        return self._set("session", session)

    def clear_session(self) -> "MockOrderBuilder":
        return self._clear("session")

    def set_duration(self, duration) -> "MockOrderBuilder":
        return self._set("duration", duration)

    def clear_duration(self) -> "MockOrderBuilder":
        return self._clear("duration")

    def set_order_type(self, order_type) -> "MockOrderBuilder":
        return self._set("orderType", order_type)

    def clear_order_type(self) -> "MockOrderBuilder":
        return self._clear("orderType")

    def set_order_strategy_type(self, order_strategy_type) -> "MockOrderBuilder":
        return self._set("orderStrategyType", order_strategy_type)

    def clear_order_strategy_type(self) -> "MockOrderBuilder":
        return self._clear("orderStrategyType")

    def set_complex_order_strategy_type(self, strategy_type) -> "MockOrderBuilder":
        return self._set("complexOrderStrategyType", strategy_type)

    def clear_complex_order_strategy_type(self) -> "MockOrderBuilder":
        return self._clear("complexOrderStrategyType")

    def set_price(self, price) -> "MockOrderBuilder":
        self._payload["price"] = _format_price(
            price, option_style=self._uses_option_pricing()
        )
        return self

    def clear_price(self) -> "MockOrderBuilder":
        return self._clear("price")

    def set_stop_price(self, stop_price) -> "MockOrderBuilder":
        self._payload["stopPrice"] = _format_price(stop_price, option_style=False)
        return self

    def clear_stop_price(self) -> "MockOrderBuilder":
        return self._clear("stopPrice")

    def copy_price(
        self, other: "MockOrderBuilder | dict[str, Any]"
    ) -> "MockOrderBuilder":
        if hasattr(other, "build"):
            other = other.build()
        if "price" in other:
            self._payload["price"] = other["price"]
        return self

    def copy_stop_price(
        self, other: "MockOrderBuilder | dict[str, Any]"
    ) -> "MockOrderBuilder":
        if hasattr(other, "build"):
            other = other.build()
        if "stopPrice" in other:
            self._payload["stopPrice"] = other["stopPrice"]
        return self

    def set_quantity(self, quantity: int) -> "MockOrderBuilder":
        self._payload["quantity"] = quantity
        return self

    def clear_quantity(self) -> "MockOrderBuilder":
        return self._clear("quantity")

    def add_equity_leg(
        self, instruction, symbol: str, quantity: int
    ) -> "MockOrderBuilder":
        legs = self._payload.setdefault("orderLegCollection", [])
        legs.append(
            {
                "instruction": self.convert_enum(instruction),
                "instrument": {"assetType": "EQUITY", "symbol": symbol},
                "quantity": quantity,
            }
        )
        return self

    def add_option_leg(
        self, instruction, symbol: str, quantity: int
    ) -> "MockOrderBuilder":
        legs = self._payload.setdefault("orderLegCollection", [])
        legs.append(
            {
                "instruction": self.convert_enum(instruction),
                "instrument": {"assetType": "OPTION", "symbol": symbol},
                "quantity": quantity,
            }
        )
        return self

    def clear_order_legs(self) -> "MockOrderBuilder":
        self._payload.pop("orderLegCollection", None)
        return self

    def add_child_order_strategy(
        self, child_order_strategy: "MockOrderBuilder | dict[str, Any]"
    ) -> "MockOrderBuilder":
        children = self._payload.setdefault("childOrderStrategies", [])
        children.append(child_order_strategy)
        return self

    def clear_child_order_strategies(self) -> "MockOrderBuilder":
        return self._clear("childOrderStrategies")


class OptionSymbol:
    """Construct a Schwab-style OCC option symbol."""

    def __init__(
        self,
        underlying_symbol: str,
        expiration_date: datetime.date | datetime.datetime | str,
        contract_type: str,
        strike_price_as_string: str,
    ):
        self.underlying_symbol = underlying_symbol

        contract_type = contract_type.upper()
        if contract_type in ("C", "CALL"):
            self.contract_type = "C"
        elif contract_type in ("P", "PUT"):
            self.contract_type = "P"
        else:
            raise ValueError("contract_type must be one of C, CALL, P, PUT")

        if isinstance(expiration_date, str):
            self.expiration_date = datetime.datetime.strptime(
                expiration_date, "%y%m%d"
            ).date()
        elif isinstance(expiration_date, datetime.datetime):
            self.expiration_date = expiration_date.date()
        elif isinstance(expiration_date, datetime.date):
            self.expiration_date = expiration_date
        else:
            raise ValueError("expiration_date must be %y%m%d, date, or datetime")

        strike = float(strike_price_as_string)
        if strike <= 0:
            raise ValueError("strike_price_as_string must represent a positive float")
        self.strike_price = strike_price_as_string.rstrip("0").rstrip(".")

    @classmethod
    def parse_symbol(cls, symbol: str) -> "OptionSymbol":
        underlying = symbol[:6].rstrip()
        rest = symbol[6:]

        if "P" in rest:
            expiration, strike = rest.split("P", maxsplit=1)
            contract_type = "P"
        elif "C" in rest:
            expiration, strike = rest.split("C", maxsplit=1)
            contract_type = "C"
        else:
            raise ValueError("option symbol must include contract type C or P")

        strike_string = str(int(strike) / 1000.0)
        return cls(underlying, expiration, contract_type, strike_string)

    def build(self) -> str:
        return "{:<6}{}{}{:08d}".format(
            self.underlying_symbol,
            self.expiration_date.strftime("%y%m%d"),
            self.contract_type,
            int(float(self.strike_price) * 1000),
        )


class MockResponse:
    """
    Mock HTTP response object matching httpx/requests interface.

    This class replicates the `httpx.Response` interface returned by `schwab-py`.
    It allows bot code to call `.json()` and check `.status_code` identically
    in both simulation and production environments.
    """

    def __init__(
        self,
        json_data: dict | list,
        status_code: int = 200,
        headers: dict | None = None,
    ):
        """
        Initialize mock response.

        Args:
            json_data (dict): Response body
            status_code (int): HTTP status code
            headers (dict, optional): HTTP headers
        """
        self._json_data = json_data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> Any:
        """Return JSON response body (dict or list, matching httpx.Response)."""
        return self._json_data

    @property
    def text(self) -> str:
        """Return response body as text (matching httpx.Response.text)."""
        import json

        return json.dumps(self._json_data)

    @property
    def content(self) -> bytes:
        """Return response body as bytes (matching httpx.Response.content)."""
        return self.text.encode("utf-8")

    @property
    def ok(self) -> bool:
        """True if status code < 400 (matching httpx.Response.ok)."""
        return self.status_code < 400

    @property
    def is_success(self) -> bool:
        """True if status code is 2xx (matching httpx.Response.is_success)."""
        return 200 <= self.status_code < 300

    @property
    def is_error(self) -> bool:
        """True if status code >= 400 (matching httpx.Response.is_error)."""
        return self.status_code >= 400

    def raise_for_status(self) -> None:
        """Raise exception if status code indicates error."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self._json_data}")


class MockEquities:
    """
    Order builders for equity securities.

    These methods construct order JSON payloads that match the structure
    expected by `schwab.client.Client.place_order()`.

    This class mimics `schwab.orders.equities`.
    """

    @staticmethod
    def _base_order(
        symbol: str,
        quantity: int,
        instruction: str,
        order_type: str = "MARKET",
        price: float | None = None,
        stop_price: float | None = None,
    ) -> MockOrderBuilder:
        """
        Base order template.

        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares
            instruction (str): BUY, SELL, SELL_SHORT, BUY_TO_COVER
            order_type (str): MARKET, LIMIT, STOP, STOP_LIMIT
            price (float, optional): Limit price
            stop_price (float, optional): Stop price

        Returns:
            dict: Order specification
        """
        builder = (
            MockOrderBuilder()
            .set_order_type(order_type)
            .set_session("NORMAL")
            .set_duration("DAY")
            .set_order_strategy_type("SINGLE")
            .add_equity_leg(instruction, symbol, quantity)
        )

        if price is not None:
            builder.set_price(price)

        if stop_price is not None:
            builder.set_stop_price(stop_price)

        return builder

    @staticmethod
    def equity_buy_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """
        Market buy order.

        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares

        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "BUY", "MARKET")

    @staticmethod
    def equity_sell_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """
        Market sell order.

        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares

        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "SELL", "MARKET")

    @staticmethod
    def equity_sell_short_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """
        Market short sell order.

        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares to short

        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "SELL_SHORT", "MARKET")

    @staticmethod
    def equity_buy_to_cover_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """
        Market buy to cover (close short position).

        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares to cover

        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "BUY_TO_COVER", "MARKET")

    @staticmethod
    def equity_buy_limit(symbol: str, quantity: int, price: float) -> MockOrderBuilder:
        """
        Limit buy order.

        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares
            price (float): Limit price

        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "BUY", "LIMIT", price=price)

    @staticmethod
    def equity_sell_limit(symbol: str, quantity: int, price: float) -> MockOrderBuilder:
        """
        Limit sell order.

        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares
            price (float): Limit price

        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "SELL", "LIMIT", price=price)

    @staticmethod
    def equity_sell_short_limit(
        symbol: str, quantity: int, price: float
    ) -> MockOrderBuilder:
        """Limit short sell order."""
        return MockEquities._base_order(
            symbol, quantity, "SELL_SHORT", "LIMIT", price=price
        )

    @staticmethod
    def equity_buy_to_cover_limit(
        symbol: str, quantity: int, price: float
    ) -> MockOrderBuilder:
        """Limit buy to cover (close short position)."""
        return MockEquities._base_order(
            symbol, quantity, "BUY_TO_COVER", "LIMIT", price=price
        )

    @staticmethod
    def equity_buy_stop(
        symbol: str, quantity: int, stop_price: float
    ) -> MockOrderBuilder:
        """Stop buy order (buy when price rises above stop)."""
        return MockEquities._base_order(
            symbol, quantity, "BUY", "STOP", stop_price=stop_price
        )

    @staticmethod
    def equity_sell_stop(
        symbol: str, quantity: int, stop_price: float
    ) -> MockOrderBuilder:
        """Stop sell order (sell when price falls below stop)."""
        return MockEquities._base_order(
            symbol, quantity, "SELL", "STOP", stop_price=stop_price
        )

    @staticmethod
    def equity_buy_stop_limit(
        symbol: str, quantity: int, price: float, stop_price: float
    ) -> MockOrderBuilder:
        """Stop-limit buy order."""
        return MockEquities._base_order(
            symbol, quantity, "BUY", "STOP_LIMIT", price=price, stop_price=stop_price
        )

    @staticmethod
    def equity_sell_stop_limit(
        symbol: str, quantity: int, price: float, stop_price: float
    ) -> MockOrderBuilder:
        """Stop-limit sell order."""
        return MockEquities._base_order(
            symbol, quantity, "SELL", "STOP_LIMIT", price=price, stop_price=stop_price
        )


class MockOptions:
    """
    Order builders for option contracts.

    Provided for API compatibility with `schwab.orders.options`.
    Note: Option support in the simulator is currently limited.
    """

    @staticmethod
    def _base_option_order(
        symbol: str,
        quantity: int,
        instruction: str,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> MockOrderBuilder:
        """Base option order template."""
        builder = (
            MockOrderBuilder()
            .set_order_type(order_type)
            .set_session("NORMAL")
            .set_duration("DAY")
            .set_order_strategy_type("SINGLE")
            .add_option_leg(instruction, symbol, quantity)
        )

        if price is not None:
            builder.set_price(price)

        return builder

    @staticmethod
    def option_buy_to_open_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """
        Buy to open option position (market order).

        Args:
            symbol (str): Option symbol (e.g., 'AAPL  230616C00170000')
            quantity (int): Number of contracts

        Returns:
            dict: Order specification
        """
        return MockOptions._base_option_order(symbol, quantity, "BUY_TO_OPEN", "MARKET")

    @staticmethod
    def option_sell_to_close_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """
        Sell to close option position (market order).

        Args:
            symbol (str): Option symbol
            quantity (int): Number of contracts

        Returns:
            dict: Order specification
        """
        return MockOptions._base_option_order(
            symbol, quantity, "SELL_TO_CLOSE", "MARKET"
        )

    @staticmethod
    def option_sell_to_open_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """
        Sell to open option position (write options).

        Args:
            symbol (str): Option symbol
            quantity (int): Number of contracts

        Returns:
            dict: Order specification
        """
        return MockOptions._base_option_order(
            symbol, quantity, "SELL_TO_OPEN", "MARKET"
        )

    @staticmethod
    def option_buy_to_close_market(symbol: str, quantity: int) -> MockOrderBuilder:
        """Buy to close option position (close short)."""
        return MockOptions._base_option_order(
            symbol, quantity, "BUY_TO_CLOSE", "MARKET"
        )

    # --- Limit order variants ---

    @staticmethod
    def option_buy_to_open_limit(
        symbol: str, quantity: int, price: float
    ) -> MockOrderBuilder:
        """Limit buy to open option position."""
        return MockOptions._base_option_order(
            symbol, quantity, "BUY_TO_OPEN", "LIMIT", price=price
        )

    @staticmethod
    def option_sell_to_close_limit(
        symbol: str, quantity: int, price: float
    ) -> MockOrderBuilder:
        """Limit sell to close option position."""
        return MockOptions._base_option_order(
            symbol, quantity, "SELL_TO_CLOSE", "LIMIT", price=price
        )

    @staticmethod
    def option_sell_to_open_limit(
        symbol: str, quantity: int, price: float
    ) -> MockOrderBuilder:
        """Limit sell to open option position (write options)."""
        return MockOptions._base_option_order(
            symbol, quantity, "SELL_TO_OPEN", "LIMIT", price=price
        )

    @staticmethod
    def option_buy_to_close_limit(
        symbol: str, quantity: int, price: float
    ) -> MockOrderBuilder:
        """Limit buy to close option position."""
        return MockOptions._base_option_order(
            symbol, quantity, "BUY_TO_CLOSE", "LIMIT", price=price
        )

    @staticmethod
    def _vertical_spread(
        *,
        long_symbol: str,
        short_symbol: str,
        quantity: int,
        net_price: float,
        order_type: str,
        long_instruction: str,
        short_instruction: str,
    ) -> MockOrderBuilder:
        return (
            MockOrderBuilder()
            .set_session("NORMAL")
            .set_duration("DAY")
            .set_order_type(order_type)
            .set_complex_order_strategy_type("VERTICAL")
            .set_order_strategy_type("SINGLE")
            .set_quantity(quantity)
            .set_price(net_price)
            .add_option_leg(long_instruction, long_symbol, quantity)
            .add_option_leg(short_instruction, short_symbol, quantity)
        )

    @staticmethod
    def bull_call_vertical_open(
        long_call_symbol: str, short_call_symbol: str, quantity: int, net_debit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_call_symbol,
            short_symbol=short_call_symbol,
            quantity=quantity,
            net_price=net_debit,
            order_type="NET_DEBIT",
            long_instruction="BUY_TO_OPEN",
            short_instruction="SELL_TO_OPEN",
        )

    @staticmethod
    def bull_call_vertical_close(
        long_call_symbol: str, short_call_symbol: str, quantity: int, net_credit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_call_symbol,
            short_symbol=short_call_symbol,
            quantity=quantity,
            net_price=net_credit,
            order_type="NET_CREDIT",
            long_instruction="SELL_TO_CLOSE",
            short_instruction="BUY_TO_CLOSE",
        )

    @staticmethod
    def bear_call_vertical_open(
        short_call_symbol: str, long_call_symbol: str, quantity: int, net_credit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_call_symbol,
            short_symbol=short_call_symbol,
            quantity=quantity,
            net_price=net_credit,
            order_type="NET_CREDIT",
            long_instruction="BUY_TO_OPEN",
            short_instruction="SELL_TO_OPEN",
        )

    @staticmethod
    def bear_call_vertical_close(
        short_call_symbol: str, long_call_symbol: str, quantity: int, net_debit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_call_symbol,
            short_symbol=short_call_symbol,
            quantity=quantity,
            net_price=net_debit,
            order_type="NET_DEBIT",
            long_instruction="SELL_TO_CLOSE",
            short_instruction="BUY_TO_CLOSE",
        )

    @staticmethod
    def bull_put_vertical_open(
        long_put_symbol: str, short_put_symbol: str, quantity: int, net_debit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_put_symbol,
            short_symbol=short_put_symbol,
            quantity=quantity,
            net_price=net_debit,
            order_type="NET_DEBIT",
            long_instruction="BUY_TO_OPEN",
            short_instruction="SELL_TO_OPEN",
        )

    @staticmethod
    def bull_put_vertical_close(
        long_put_symbol: str, short_put_symbol: str, quantity: int, net_credit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_put_symbol,
            short_symbol=short_put_symbol,
            quantity=quantity,
            net_price=net_credit,
            order_type="NET_CREDIT",
            long_instruction="SELL_TO_CLOSE",
            short_instruction="BUY_TO_CLOSE",
        )

    @staticmethod
    def bear_put_vertical_open(
        short_put_symbol: str, long_put_symbol: str, quantity: int, net_credit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_put_symbol,
            short_symbol=short_put_symbol,
            quantity=quantity,
            net_price=net_credit,
            order_type="NET_CREDIT",
            long_instruction="BUY_TO_OPEN",
            short_instruction="SELL_TO_OPEN",
        )

    @staticmethod
    def bear_put_vertical_close(
        short_put_symbol: str, long_put_symbol: str, quantity: int, net_debit: float
    ) -> MockOrderBuilder:
        return MockOptions._vertical_spread(
            long_symbol=long_put_symbol,
            short_symbol=short_put_symbol,
            quantity=quantity,
            net_price=net_debit,
            order_type="NET_DEBIT",
            long_instruction="SELL_TO_CLOSE",
            short_instruction="BUY_TO_CLOSE",
        )


OrderBuilder = MockOrderBuilder

equity_buy_market = MockEquities.equity_buy_market
equity_sell_market = MockEquities.equity_sell_market
equity_sell_short_market = MockEquities.equity_sell_short_market
equity_buy_to_cover_market = MockEquities.equity_buy_to_cover_market
equity_buy_limit = MockEquities.equity_buy_limit
equity_sell_limit = MockEquities.equity_sell_limit
equity_sell_short_limit = MockEquities.equity_sell_short_limit
equity_buy_to_cover_limit = MockEquities.equity_buy_to_cover_limit
equity_buy_stop = MockEquities.equity_buy_stop
equity_sell_stop = MockEquities.equity_sell_stop
equity_buy_stop_limit = MockEquities.equity_buy_stop_limit
equity_sell_stop_limit = MockEquities.equity_sell_stop_limit

option_buy_to_open_market = MockOptions.option_buy_to_open_market
option_sell_to_close_market = MockOptions.option_sell_to_close_market
option_sell_to_open_market = MockOptions.option_sell_to_open_market
option_buy_to_close_market = MockOptions.option_buy_to_close_market
option_buy_to_open_limit = MockOptions.option_buy_to_open_limit
option_sell_to_close_limit = MockOptions.option_sell_to_close_limit
option_sell_to_open_limit = MockOptions.option_sell_to_open_limit
option_buy_to_close_limit = MockOptions.option_buy_to_close_limit
bull_call_vertical_open = MockOptions.bull_call_vertical_open
bull_call_vertical_close = MockOptions.bull_call_vertical_close
bear_call_vertical_open = MockOptions.bear_call_vertical_open
bear_call_vertical_close = MockOptions.bear_call_vertical_close
bull_put_vertical_open = MockOptions.bull_put_vertical_open
bull_put_vertical_close = MockOptions.bull_put_vertical_close
bear_put_vertical_open = MockOptions.bear_put_vertical_open
bear_put_vertical_close = MockOptions.bear_put_vertical_close

__all__ = [
    "MockResponse",
    "MockOrderBuilder",
    "OrderBuilder",
    "OptionSymbol",
    "MockEquities",
    "MockOptions",
    "equity_buy_market",
    "equity_sell_market",
    "equity_sell_short_market",
    "equity_buy_to_cover_market",
    "equity_buy_limit",
    "equity_sell_limit",
    "equity_sell_short_limit",
    "equity_buy_to_cover_limit",
    "equity_buy_stop",
    "equity_sell_stop",
    "equity_buy_stop_limit",
    "equity_sell_stop_limit",
    "option_buy_to_open_market",
    "option_sell_to_close_market",
    "option_sell_to_open_market",
    "option_buy_to_close_market",
    "option_buy_to_open_limit",
    "option_sell_to_close_limit",
    "option_sell_to_open_limit",
    "option_buy_to_close_limit",
    "bull_call_vertical_open",
    "bull_call_vertical_close",
    "bear_call_vertical_open",
    "bear_call_vertical_close",
    "bull_put_vertical_open",
    "bull_put_vertical_close",
    "bear_put_vertical_open",
    "bear_put_vertical_close",
]
