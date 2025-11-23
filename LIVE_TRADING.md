# Live Trading Deployment Guide

This guide outlines the steps required to transition a strategy from the SchwabGym simulator to live trading using the `schwab-py` library.

## Prerequisites

- A Charles Schwab developer account.
- An approved application with API Key and Secret.
- The `schwab-py` library installed (`pip install schwab-py`).

## Code Migration

The `SchwabGym` API is designed to mirror `schwab-py`. Migration primarily involves updating import statements and initializing the authenticated client.

### 1. Update Imports

Replace the simulator classes with their `schwab-py` equivalents.

**Simulation:**
```python
from schwabgym import MockClient
from schwabgym.orders import MockEquities as eq
```

**Live Trading:**
```python
from schwab.client import Client
from schwab.orders import equities as eq
from schwab import auth
```

### 2. Authenticate Client

The `MockClient` requires only data and initial cash. The real `Client` requires OAuth authentication.

**Simulation:**
```python
client = MockClient(df, initial_cash=25000)
```

**Live Trading:**
```python
client = auth.easy_client(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='./token.json'
)
```

### 3. Execution Loop

In simulation, `client.advance_time()` manually steps through historical data. In live trading, the script must manage the timing of API calls (e.g., using `time.sleep()` or a scheduler).

**Simulation:**
```python
while client.advance_time():
    quote = client.get_quotes('AAPL')
    # ... strategy logic ...
```

**Live Trading:**
```python
import time

while True:
    quote = client.get_quotes('AAPL')
    # ... strategy logic ...
    time.sleep(60) # Wait for next candle
```

## Emergency Procedures

For live trading, it is recommended to have a standalone script ready to close all positions immediately in case of system failure or unexpected behavior.

```python
# emergency_close.py
from schwab.client import Client
from schwab.orders import equities as eq

# ... authenticate client ...

# Get all positions
acct = client.get_account(account_hash, fields='positions').json()
positions = acct['securitiesAccount'].get('positions', [])

for pos in positions:
    symbol = pos['instrument']['symbol']
    qty = pos['longQuantity'] if pos['longQuantity'] > 0 else pos['shortQuantity']

    # Place market order to close
    if pos['longQuantity'] > 0:
        order = eq.equity_sell_market(symbol, qty)
    else:
        order = eq.equity_buy_to_cover_market(symbol, abs(qty))

    client.place_order(account_hash, order)
    print(f"Closed {symbol}")
```

## Production Considerations

When moving to a live environment, consider the following:

-   **Error Handling:** Implement try/except blocks around API calls to handle network issues or API limits.
-   **Rate Limiting:** Ensure your request frequency complies with Schwab's API limits (typically 120 requests per minute).
-   **State Persistence:** The `MockClient` tracks state in memory. For live trading, consider saving position and order state to a database or file to recover from restarts.
-   **Risk Management:** Verify that your logic includes safety checks for position sizes and stop losses independent of the broker's validation.

## Disclaimer

Trading involves risk. Always test your code thoroughly in a controlled environment before deploying it with real capital. This guide provides technical instructions for library usage and does not constitute financial advice.
