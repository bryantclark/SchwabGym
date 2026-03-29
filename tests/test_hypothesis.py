"""
Property-Based Tests for SchwabGym
===================================

Uses hypothesis to find edge cases through invariant testing.
"""

import datetime

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from schwabgym.account import Account  # noqa: E402
from schwabgym.data import generate_dummy_data  # noqa: E402
from schwabgym.fees import FeeCalculator  # noqa: E402
from schwabgym.physics.fast import FastExecutionEngine  # noqa: E402
from schwabgym.physics.realistic import RealisticExecutionEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

prices = st.floats(
    min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False
)
quantities = st.integers(min_value=1, max_value=100_000)
volumes = st.integers(min_value=1, max_value=10_000_000)


@st.composite
def ohlcv_bar(draw):
    """Generate a valid OHLCV bar where High >= max(O,C) and Low <= min(O,C)."""
    close = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False))
    open_ = draw(
        st.floats(min_value=close * 0.9, max_value=close * 1.1, allow_nan=False)
    )
    high = draw(
        st.floats(
            min_value=max(open_, close),
            max_value=max(open_, close) * 1.05,
            allow_nan=False,
        )
    )
    low = draw(
        st.floats(
            min_value=min(open_, close) * 0.95,
            max_value=min(open_, close),
            allow_nan=False,
        )
    )
    volume = draw(st.integers(min_value=100, max_value=10_000_000))
    return {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Volatility": (high - low) / close if close > 0 else 0.01,
    }


# ---------------------------------------------------------------------------
# Fee invariants
# ---------------------------------------------------------------------------


class TestFeeInvariants:
    """Property-based tests for fee calculations."""

    def setup_method(self):
        self.calc = FeeCalculator()

    @given(
        qty=st.integers(min_value=1, max_value=100_000),
        price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False),
    )
    def test_fees_are_non_negative(self, qty, price):
        """Regulatory fees should never be negative."""
        date = datetime.date(2024, 1, 1)
        fee = self.calc.calculate_total_regulatory_fees(date, qty, price, "EQUITY")
        assert fee >= 0

    @given(
        qty_small=st.integers(min_value=1, max_value=1000),
        qty_large=st.integers(min_value=1001, max_value=100_000),
        price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False),
    )
    def test_fee_monotonicity_in_quantity(self, qty_small, qty_large, price):
        """Larger orders should produce equal or greater fees (up to TAF cap)."""
        date = datetime.date(2024, 1, 1)
        fee_small = self.calc.calculate_total_regulatory_fees(
            date, qty_small, price, "EQUITY"
        )
        fee_large = self.calc.calculate_total_regulatory_fees(
            date, qty_large, price, "EQUITY"
        )
        assert fee_large >= fee_small

    @given(
        qty=st.integers(min_value=1, max_value=100_000),
        price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False),
    )
    def test_taf_never_exceeds_cap(self, qty, price):
        """TAF should never exceed the schedule's cap for the given date."""
        trade_date = datetime.date(2024, 1, 1)
        taf = self.calc.calculate_taf(qty, "EQUITY", trade_date=trade_date)
        info = self.calc.get_fee_schedule_info(trade_date)
        assert taf <= info["taf_cap"] + 1e-10

    @given(
        qty=st.integers(min_value=1, max_value=100_000),
        price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False),
    )
    def test_breakeven_above_entry(self, qty, price):
        """Breakeven price should always be >= entry price."""
        date = datetime.date(2024, 1, 1)
        breakeven = self.calc.estimate_breakeven_profit(date, qty, price, "EQUITY")
        assert breakeven >= price


# ---------------------------------------------------------------------------
# OHLCV data invariants
# ---------------------------------------------------------------------------


class TestOHLCVInvariants:
    """Property-based tests for synthetic data generation."""

    @given(
        periods=st.integers(min_value=10, max_value=500),
        start_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
        volatility=st.floats(min_value=0.001, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_ohlc_consistency(self, periods, start_price, volatility):
        """High >= max(Open, Close) and Low <= min(Open, Close) for all bars."""
        df = generate_dummy_data(
            "TEST",
            periods=periods,
            start_price=start_price,
            volatility=volatility,
            seed=42,
        )
        assert (df["High"] >= df["Open"]).all()
        assert (df["High"] >= df["Close"]).all()
        assert (df["Low"] <= df["Open"]).all()
        assert (df["Low"] <= df["Close"]).all()
        assert (df["High"] >= df["Low"]).all()

    @given(
        periods=st.integers(min_value=10, max_value=500),
        start_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_prices_are_positive(self, periods, start_price):
        """All prices should be strictly positive."""
        df = generate_dummy_data(
            "TEST", periods=periods, start_price=start_price, seed=42
        )
        for col in ["Open", "High", "Low", "Close"]:
            assert (df[col] > 0).all(), f"{col} has non-positive values"

    @given(
        periods=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_volume_is_non_negative(self, periods):
        """Volume should never be negative."""
        df = generate_dummy_data("TEST", periods=periods, seed=42)
        assert (df["Volume"] >= 0).all()


# ---------------------------------------------------------------------------
# Execution engine invariants
# ---------------------------------------------------------------------------


class TestExecutionEngineInvariants:
    """Property-based tests for physics engines."""

    @given(bar=ohlcv_bar(), qty=quantities)
    def test_buy_impact_increases_price(self, bar, qty):
        """Buying should never decrease the execution price."""
        engine = RealisticExecutionEngine(seed=42)
        exec_price = engine.calculate_execution_price(
            base_price=bar["Close"],
            quantity=qty,
            instruction="BUY",
            market_data=bar,
        )
        assert exec_price >= bar["Close"]

    @given(bar=ohlcv_bar(), qty=quantities)
    def test_sell_impact_decreases_price(self, bar, qty):
        """Selling should never increase the execution price."""
        engine = RealisticExecutionEngine(seed=42)
        exec_price = engine.calculate_execution_price(
            base_price=bar["Close"],
            quantity=qty,
            instruction="SELL",
            market_data=bar,
        )
        assert exec_price <= bar["Close"]

    @given(
        bar=ohlcv_bar(),
        qty_small=st.integers(1, 100),
        qty_large=st.integers(101, 10000),
    )
    def test_larger_orders_have_more_impact(self, bar, qty_small, qty_large):
        """Square Root Law: larger orders should have greater market impact."""
        engine = RealisticExecutionEngine(seed=42)
        price_small = engine.calculate_execution_price(
            base_price=bar["Close"],
            quantity=qty_small,
            instruction="BUY",
            market_data=bar,
        )
        price_large = engine.calculate_execution_price(
            base_price=bar["Close"],
            quantity=qty_large,
            instruction="BUY",
            market_data=bar,
        )
        assert price_large >= price_small

    @given(bar=ohlcv_bar(), qty=quantities)
    def test_fast_engine_deterministic(self, bar, qty):
        """Fast engine should produce the same result every time."""
        engine = FastExecutionEngine(base_slippage=0.01)
        p1 = engine.calculate_execution_price(bar["Close"], qty, "BUY", bar)
        p2 = engine.calculate_execution_price(bar["Close"], qty, "BUY", bar)
        assert p1 == p2

    @given(
        bar=ohlcv_bar(),
        qty=st.integers(min_value=1, max_value=1000),
    )
    def test_impact_is_sublinear(self, bar, qty):
        """Square Root Law predicts sublinear impact: 10x quantity < 10x impact."""
        assume(qty >= 1)
        engine = RealisticExecutionEngine(seed=42)
        impact_1 = abs(
            engine.calculate_execution_price(bar["Close"], qty, "BUY", bar)
            - bar["Close"]
        )
        impact_10 = abs(
            engine.calculate_execution_price(bar["Close"], qty * 10, "BUY", bar)
            - bar["Close"]
        )
        # sqrt(10) ≈ 3.16, so 10x quantity should give ~3.16x impact, not 10x
        if impact_1 > 0:
            assert impact_10 < impact_1 * 10 + 1e-10


# ---------------------------------------------------------------------------
# Account roundtrip invariant
# ---------------------------------------------------------------------------


class TestAccountInvariants:
    """Property-based tests for account state consistency."""

    @given(
        price=st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
        qty=st.integers(min_value=1, max_value=100),
    )
    def test_buy_sell_roundtrip_conserves_cash(self, price, qty):
        """Buy N shares then sell N at the same price should return cash minus fees."""
        account = Account(initial_cash=100_000.0)
        trade_date = datetime.date(2025, 6, 1)  # post-SEC-fee era

        # Buy
        account.execute_trade(
            symbol="TEST",
            quantity=qty,
            price=price,
            instruction="BUY",
            asset_type="EQUITY",
            trade_date=trade_date,
            buying_power_check=200_000.0,
        )

        # Sell at same price
        account.execute_trade(
            symbol="TEST",
            quantity=qty,
            price=price,
            instruction="SELL",
            asset_type="EQUITY",
            trade_date=trade_date,
            buying_power_check=200_000.0,
        )

        # Cash should be initial minus fees (TAF on sell)
        expected_fees = account.fee_calculator.calculate_total_regulatory_fees(
            trade_date, qty, price, "EQUITY"
        )
        assert abs(account.cash - (100_000.0 - expected_fees)) < 0.01

    @given(
        price=st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
        qty=st.integers(min_value=1, max_value=100),
    )
    def test_position_cleared_after_roundtrip(self, price, qty):
        """After buy+sell of same quantity, position should be empty."""
        account = Account(initial_cash=100_000.0)
        trade_date = datetime.date(2025, 6, 1)

        account.execute_trade(
            symbol="TEST",
            quantity=qty,
            price=price,
            instruction="BUY",
            asset_type="EQUITY",
            trade_date=trade_date,
            buying_power_check=200_000.0,
        )
        account.execute_trade(
            symbol="TEST",
            quantity=qty,
            price=price,
            instruction="SELL",
            asset_type="EQUITY",
            trade_date=trade_date,
            buying_power_check=200_000.0,
        )

        assert "TEST" not in account.positions

    @given(
        initial_cash=st.floats(
            min_value=1000.0, max_value=1_000_000.0, allow_nan=False
        ),
    )
    def test_reset_restores_initial_state(self, initial_cash):
        """Reset should return account to initial state."""
        account = Account(initial_cash=initial_cash)
        # Dirty the state
        account.cash = 0
        account.positions["FOO"] = {
            "quantity": 100,
            "avgPrice": 50.0,
            "assetType": "EQUITY",
        }
        account.is_pdt_flagged = True

        account.reset()

        assert account.cash == initial_cash
        assert len(account.positions) == 0
        assert not account.is_pdt_flagged
