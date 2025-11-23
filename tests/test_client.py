"""
Client Tests
============

Unit tests for the MockClient simulator.

Author: Bryant Clark
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from schwabgym import MockClient
from schwabgym.orders import MockEquities as eq

class TestMockClient:
    """Test MockClient functionality."""

    def test_initialization(self, sample_data):
        """Test client initialization."""
        client = MockClient(sample_data, initial_cash=50000.0)
        assert client.cash == 50000.0
        assert client.current_step == 0
        assert client.max_steps == len(sample_data) - 1
        assert len(client.positions) == 0

    def test_account_linked(self, client):
        """Test account linking."""
        resp = client.account_linked()
        assert resp.status_code == 200
        data = resp.json()
        assert 'accountNumber' in data
        assert 'hashValue' in data
        assert data['hashValue'] == client.account_hash

    def test_account_details(self, client):
        """Test account details retrieval."""
        resp = client.account_details(client.account_hash)
        assert resp.status_code == 200
        data = resp.json()
        
        acct = data['securitiesAccount']
        assert acct['currentBalances']['cashBalance'] == 10000.0
        assert acct['currentBalances']['liquidationValue'] == 10000.0
        assert acct['currentBalances']['buyingPower'] == 20000.0  # 2:1 margin
        assert len(acct['positions']) == 0

    def test_quote(self, client):
        """Test quote retrieval."""
        # Single symbol
        resp = client.quote('TEST')
        assert resp.status_code == 200
        data = resp.json()
        assert 'TEST' in data
        assert data['TEST']['quote']['symbol'] == 'TEST'
        assert data['TEST']['quote']['lastPrice'] > 0

        # Multiple symbols
        resp = client.quote(['TEST', 'OTHER'])
        data = resp.json()
        assert 'TEST' in data
        assert 'OTHER' in data

    def test_price_history(self, client):
        """Test price history retrieval."""
        resp = client.price_history('TEST')
        assert resp.status_code == 200
        data = resp.json()
        assert 'candles' in data
        assert len(data['candles']) > 0
        assert 'close' in data['candles'][0]

    def test_market_buy_order(self, client):
        """Test placing a market buy order."""
        # Get initial price
        quote = client.quote('TEST').json()['TEST']['quote']['lastPrice']
        qty = 10
        
        # Place order
        order = eq.equity_buy_market('TEST', qty)
        resp = client.place_order(client.account_hash, order)
        
        assert resp.status_code == 201
        
        # Verify position
        acct = client.account_details(client.account_hash).json()['securitiesAccount']
        positions = acct['positions']
        assert len(positions) == 1
        assert positions[0]['instrument']['symbol'] == 'TEST'
        assert positions[0]['longQuantity'] == qty
        
        # Verify cash deduction (approximate due to spread/slippage)
        expected_cost = qty * quote
        assert client.cash < 10000.0 - expected_cost * 0.99 

    def test_market_sell_order(self, client):
        """Test placing a market sell order (long exit)."""
        # Establish long position first
        client.place_order(client.account_hash, eq.equity_buy_market('TEST', 20))
        
        # Sell half
        order = eq.equity_sell_market('TEST', 10)
        resp = client.place_order(client.account_hash, order)
        
        assert resp.status_code == 201
        
        # Verify position reduced
        acct = client.account_details(client.account_hash).json()['securitiesAccount']
        pos = acct['positions'][0]
        assert pos['longQuantity'] == 10

    def test_short_selling(self, client):
        """Test short selling."""
        # Place short order
        order = eq.equity_sell_short_market('TEST', 10)
        resp = client.place_order(client.account_hash, order)
        
        assert resp.status_code == 201
        
        # Verify short position
        acct = client.account_details(client.account_hash).json()['securitiesAccount']
        pos = acct['positions'][0]
        assert pos['shortQuantity'] == 10
        assert pos['longQuantity'] == 0

    def test_limit_order_queuing(self, client):
        """Test that limit orders are queued."""
        # Place limit buy well below market
        current_price = client.quote('TEST').json()['TEST']['quote']['lastPrice']
        limit_price = current_price * 0.5
        
        order = eq.equity_buy_limit('TEST', 10, limit_price)
        resp = client.place_order(client.account_hash, order)
        
        assert resp.status_code == 201
        
        # Should be in working orders, not positions
        assert len(client.working_orders) == 1
        assert len(client.positions) == 0

    def test_limit_order_fill(self, fast_client):
        """Test that limit orders get filled when price crosses."""
        client = fast_client
        # Current price is around 100 (from sample_data fixture)
        # Place limit buy above market (should fill immediately in next step)
        limit_price = 1000.0 
        order = eq.equity_buy_limit('TEST', 10, limit_price)
        client.place_order(client.account_hash, order)
        
        assert len(client.working_orders) == 1
        
        # Advance time to trigger fill check
        client.advance_time()
        
        # Should be filled
        assert len(client.working_orders) == 0
        assert len(client.positions) == 1
        assert client.positions['TEST']['quantity'] == 10

    def test_insufficient_funds(self, client):
        """Test rejection on insufficient funds."""
        # Try to buy more than cash available
        qty = 1000000
        order = eq.equity_buy_market('TEST', qty)
        resp = client.place_order(client.account_hash, order)
        
        assert resp.status_code == 400
        assert "Insufficient Buying Power" in resp.json()['error']

    def test_sell_more_than_owned(self, client):
        """Test rejection when selling more than owned (without shorting)."""
        # Try to sell without owning
        order = eq.equity_sell_market('TEST', 10)
        resp = client.place_order(client.account_hash, order)
        
        assert resp.status_code == 400
        assert "Position not available" in resp.json()['error']

    def test_unsupported_order_type(self, client):
        """Test rejection of unsupported order types."""
        order = eq.equity_buy_market('TEST', 10)
        order['orderType'] = 'INVALID_TYPE'
        resp = client.place_order(client.account_hash, order)
        
        assert resp.status_code == 400
        assert "Unsupported order type" in resp.json()['error']

    def test_unauthorized_access(self, client):
        """Test unauthorized access."""
        resp = client.account_details("WRONG_HASH")
        assert resp.status_code == 401
        
        resp = client.place_order("WRONG_HASH", {})
        assert resp.status_code == 401

    def test_pdt_rule_flagging(self, client):
        """Test Pattern Day Trader flagging."""
        # Force account value below $25k (client fixture starts with 10k)
        assert client._calculate_equity() < 25000
        
        # Execute 4 day trades
        for _ in range(4):
            # Buy
            client.place_order(client.account_hash, eq.equity_buy_market('TEST', 10))
            # Sell (Day Trade)
            client.place_order(client.account_hash, eq.equity_sell_market('TEST', 10))
            
        # 5th trade should be blocked
        order = eq.equity_buy_market('TEST', 10)
        client.place_order(client.account_hash, order) # Open
        
        # Try to close (Day Trade #5)
        resp = client.place_order(client.account_hash, eq.equity_sell_market('TEST', 10))
        
        assert resp.status_code == 403
        assert "Pattern Day Trader" in resp.json()['error']
        
        # Verify subsequent orders are also blocked
        resp = client.place_order(client.account_hash, eq.equity_buy_market('TEST', 1))
        assert resp.status_code == 403

    def test_advance_time(self, client):
        """Test time advancement."""
        start_step = client.current_step
        res = client.advance_time()
        assert res is True
        assert client.current_step == start_step + 1

    def test_reset(self, client):
        """Test client reset."""
        client.advance_time()
        client.place_order(client.account_hash, eq.equity_buy_market('TEST', 10))
        
        client.reset()
        
        assert client.current_step == 0
        assert len(client.positions) == 0
        assert len(client.working_orders) == 0
