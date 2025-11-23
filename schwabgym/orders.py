"""
SchwabGym Order Builders
========================

Compatible order construction matching schwab.orders.equities and
schwab.orders.options.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

from typing import Optional, Dict, Any


class MockResponse:
    """
    Mock HTTP response object matching httpx/requests interface.
    
    This class replicates the `httpx.Response` interface returned by `schwab-py`.
    It allows bot code to call `.json()` and check `.status_code` identically
    in both simulation and production environments.
    """
    
    def __init__(self, json_data: Dict, status_code: int = 200, headers: Optional[Dict] = None):
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
    
    def json(self) -> Dict:
        """Return JSON response body."""
        return self._json_data
    
    def raise_for_status(self) -> None:
        """Raise exception if status code indicates error."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}: {self._json_data}")


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
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
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
        order = {
            "orderType": order_type,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": "EQUITY"
                }
            }]
        }
        
        if price is not None:
            order["price"] = f"{price:.4f}"  # Schwab expects string with 4 decimals
        
        if stop_price is not None:
            order["stopPrice"] = f"{stop_price:.4f}"
        
        return order
    
    @staticmethod
    def equity_buy_market(symbol: str, quantity: int) -> Dict[str, Any]:
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
    def equity_sell_market(symbol: str, quantity: int) -> Dict[str, Any]:
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
    def equity_sell_short_market(symbol: str, quantity: int) -> Dict[str, Any]:
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
    def equity_buy_to_cover_market(symbol: str, quantity: int) -> Dict[str, Any]:
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
    def equity_buy_limit(symbol: str, quantity: int, price: float) -> Dict[str, Any]:
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
    def equity_sell_limit(symbol: str, quantity: int, price: float) -> Dict[str, Any]:
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
    def equity_buy_stop(symbol: str, quantity: int, stop_price: float) -> Dict[str, Any]:
        """
        Stop buy order (buy when price rises above stop).
        
        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares
            stop_price (float): Stop trigger price
            
        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "BUY", "STOP", stop_price=stop_price)
    
    @staticmethod
    def equity_sell_stop(symbol: str, quantity: int, stop_price: float) -> Dict[str, Any]:
        """
        Stop sell order (sell when price falls below stop).
        
        Args:
            symbol (str): Ticker symbol
            quantity (int): Number of shares
            stop_price (float): Stop trigger price
            
        Returns:
            dict: Order specification
        """
        return MockEquities._base_order(symbol, quantity, "SELL", "STOP", stop_price=stop_price)


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
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Base option order template."""
        order = {
            "orderType": order_type,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": "OPTION"
                }
            }]
        }
        
        if price is not None:
            order["price"] = f"{price:.2f}"  # Options use 2 decimals
        
        return order
    
    @staticmethod
    def option_buy_to_open_market(symbol: str, quantity: int) -> Dict[str, Any]:
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
    def option_sell_to_close_market(symbol: str, quantity: int) -> Dict[str, Any]:
        """
        Sell to close option position (market order).
        
        Args:
            symbol (str): Option symbol
            quantity (int): Number of contracts
            
        Returns:
            dict: Order specification
        """
        return MockOptions._base_option_order(symbol, quantity, "SELL_TO_CLOSE", "MARKET")
    
    @staticmethod
    def option_sell_to_open_market(symbol: str, quantity: int) -> Dict[str, Any]:
        """
        Sell to open option position (write options).
        
        Args:
            symbol (str): Option symbol
            quantity (int): Number of contracts
            
        Returns:
            dict: Order specification
        """
        return MockOptions._base_option_order(symbol, quantity, "SELL_TO_OPEN", "MARKET")
    
    @staticmethod
    def option_buy_to_close_market(symbol: str, quantity: int) -> Dict[str, Any]:
        """
        Buy to close option position (close short).
        
        Args:
            symbol (str): Option symbol
            quantity (int): Number of contracts
            
        Returns:
            dict: Order specification
        """
        return MockOptions._base_option_order(symbol, quantity, "BUY_TO_CLOSE", "MARKET")


# Compatibility: Try to import real schwab-py, fall back to mocks
try:
    from schwab.orders import equities
    from schwab.orders import options
    
    # If schwab-py is installed, prefer it (for live trading)
    # But keep mocks available for testing
    pass
except ImportError:
    # schwab-py not installed, use mocks
    equities = MockEquities
    options = MockOptions
