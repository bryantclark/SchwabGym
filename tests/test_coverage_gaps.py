"""
Coverage Gap Tests
==================

Tests specifically designed to hit edge cases and error paths
in __init__.py and client.py that are missed by standard unit tests.
"""

import unittest.mock as mock

import pytest

from schwabgym import (
    MockClient,
    check_dependencies,
    get_info,
    get_version,
    print_banner,
)


class TestInitModule:
    """Tests for schwabgym/__init__.py functions."""

    def test_metadata_functions(self):
        """Test that metadata functions run without error."""
        assert isinstance(get_version(), str)
        assert isinstance(get_info(), dict)
        # Capture stdout to ensure banner prints
        with mock.patch("builtins.print") as mock_print:
            print_banner()
            mock_print.assert_called_once()

    def test_check_dependencies_success(self):
        """Test dependency check when everything is installed."""
        assert check_dependencies() is True

    def test_check_dependencies_missing_required(self):
        """Test dependency check when a required package is missing."""
        # Mock __import__ to fail for 'pandas'
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="Missing required dependencies"),
        ):
            check_dependencies()

    def test_check_dependencies_missing_optional(self):
        """Test dependency check when optional packages are missing (should warn)."""
        # Mock __import__ to fail for 'torch'
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=mock_import),
            pytest.warns(UserWarning, match="Optional dependencies not installed"),
        ):
            check_dependencies()


class TestClientAuthEdges:
    """Tests for MockClient authentication guard rails."""

    @pytest.fixture
    def client(self, sample_data):
        return MockClient(sample_data, initial_cash=10000.0)

    def test_cash_setter(self, client):
        """Test the cash property setter coverage."""
        client.cash = 50000.0
        assert client.cash == 50000.0
        assert client.account.cash == 50000.0

    def test_unauthorized_access_all_methods(self, client):
        """Ensure all secure methods return 401 on bad hash."""
        bad_hash = "BAD_HASH"

        # cancel_order
        resp = client.cancel_order(bad_hash, 123)
        assert resp.status_code == 401

        # replace_order
        resp = client.replace_order(bad_hash, 123, {})
        assert resp.status_code == 401

        # get_order
        resp = client.get_order(bad_hash, 123)
        assert resp.status_code == 401

        # get_orders_for_account
        resp = client.get_orders_for_account(bad_hash)
        assert resp.status_code == 401

    def test_internal_helpers(self, client):
        """Cover internal helper methods."""
        price = client._get_current_raw_price("TEST")
        assert isinstance(price, float)

        bp = client._calculate_buying_power(10000)
        assert bp == 20000
