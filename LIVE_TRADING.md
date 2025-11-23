# Live Trading Deployment Guide

This guide covers everything you need to switch from simulation to live trading with minimal code changes.

## ⚠️ Critical Safety Checklist

Before deploying to live trading, ensure you've completed:

- [ ] Thoroughly backtested on multiple symbols and time periods
- [ ] Validated strategy on out-of-sample data
- [ ] Tested with realistic transaction costs and slippage
- [ ] Verified Pattern Day Trader rules don't trigger unexpectedly
- [ ] Set up proper position sizing and risk management
- [ ] Created emergency stop-loss mechanisms
- [ ] Tested order execution in paper trading mode (if available)
- [ ] Set up monitoring and alerting
- [ ] Reviewed and understand all regulatory requirements
- [ ] Started with minimum capital for initial live testing

## 🔄 Code Changes Required

### 1. Import Statement Changes

**Before (Simulation):**
```python
from mock_client import MockClient as Client
from schwab_compat import MockEquities as eq
```

**After (Live):**
```python
from schwab.client import Client
from schwab.orders import equities as eq
```

### 2. Authentication Setup

**Simulation** doesn't require authentication. **Live trading** requires Schwab OAuth:

```python
from schwab import auth

# Initialize client with OAuth
client = auth.easy_client(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='./token.json'
)
```

### 3. Complete Example Comparison

#### Simulation Code
```python
from mock_client import MockClient
from data_loader import load_and_clean_data
from schwab_compat import MockEquities as eq

# Load historical data
df = load_and_clean_data('AAPL.csv')

# Initialize simulator
client = MockClient(df, initial_cash=25000)
account_hash = client.account_linked().json()['hashValue']

# Trading logic
while client.advance_time():
    quote = client.quote('AAPL')
    price = quote.json()['AAPL']['quote']['lastPrice']
    
    # Your strategy logic here
    if should_buy():
        order = eq.equity_buy_market('AAPL', 10)
        client.place_order(account_hash, order)
```

#### Live Trading Code
```python
from schwab import auth
from schwab.client import Client
from schwab.orders import equities as eq
import time

# Authenticate
client = auth.easy_client(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='./token.json'
)

# Get account
accounts = client.get_account_numbers().json()
account_hash = accounts[0]['hashValue']

# Trading loop (runs continuously)
while True:
    quote = client.get_quotes('AAPL')
    price = quote.json()['AAPL']['quote']['lastPrice']
    
    # Your strategy logic here (IDENTICAL to simulation)
    if should_buy():
        order = eq.equity_buy_market('AAPL', 10)
        client.place_order(account_hash, order)
    
    time.sleep(60)  # Wait before next iteration
```

## 🛡️ Production Best Practices

### 1. Environment Management

Use environment variables for sensitive data:

```python
import os
from dotenv import load_dotenv

load_dotenv()

client = auth.easy_client(
    api_key=os.getenv('SCHWAB_API_KEY'),
    app_secret=os.getenv('SCHWAB_APP_SECRET'),
    callback_url=os.getenv('SCHWAB_CALLBACK_URL'),
    token_path=os.getenv('SCHWAB_TOKEN_PATH')
)
```

Create a `.env` file:
```bash
SCHWAB_API_KEY=your_api_key_here
SCHWAB_APP_SECRET=your_app_secret_here
SCHWAB_CALLBACK_URL=https://127.0.0.1:8182/
SCHWAB_TOKEN_PATH=./token.json
```

**NEVER commit `.env` files to git!**

### 2. Error Handling

Add robust error handling for live trading:

```python
import time
import logging
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

def safe_place_order(client, account_hash, order, max_retries=3):
    """Place order with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.place_order(account_hash, order)
            response.raise_for_status()
            return response
        except RequestException as e:
            logger.error(f"Order failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

### 3. Position Management

Track positions carefully:

```python
def get_current_position(client, account_hash, symbol):
    """Get current position for a symbol."""
    acct = client.get_account(account_hash, fields='positions').json()
    
    for pos in acct['securitiesAccount'].get('positions', []):
        if pos['instrument']['symbol'] == symbol:
            return pos['longQuantity'] if pos['longQuantity'] > 0 else -pos['shortQuantity']
    
    return 0

# Use it in your strategy
current_shares = get_current_position(client, account_hash, 'AAPL')
if current_shares < target_shares:
    qty_to_buy = target_shares - current_shares
    order = eq.equity_buy_market('AAPL', qty_to_buy)
    safe_place_order(client, account_hash, order)
```

### 4. Risk Management

Implement position limits:

```python
class RiskManager:
    def __init__(self, max_position_size=10000, max_portfolio_risk=0.02):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
    
    def validate_order(self, client, account_hash, symbol, qty, price):
        """Check if order passes risk checks."""
        acct = client.get_account(account_hash).json()['securitiesAccount']
        nav = acct['currentBalances']['liquidationValue']
        
        # Check position size
        order_value = qty * price
        if order_value > self.max_position_size:
            logger.warning(f"Order exceeds max position size: ${order_value}")
            return False
        
        # Check portfolio risk
        if order_value / nav > self.max_portfolio_risk:
            logger.warning(f"Order exceeds max portfolio risk: {order_value/nav:.2%}")
            return False
        
        return True

# Use it
risk_mgr = RiskManager(max_position_size=10000, max_portfolio_risk=0.02)

if risk_mgr.validate_order(client, account_hash, 'AAPL', qty, price):
    order = eq.equity_buy_market('AAPL', qty)
    client.place_order(account_hash, order)
```

### 5. Monitoring & Alerts

Set up real-time monitoring:

```python
import smtplib
from email.mime.text import MIMEText

class AlertSystem:
    def __init__(self, email_config):
        self.email_config = email_config
    
    def send_alert(self, subject, message):
        """Send email alert."""
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']
        
        try:
            with smtplib.SMTP(self.email_config['smtp_server']) as server:
                server.send_message(msg)
            logger.info(f"Alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

# Use it
alerts = AlertSystem({
    'smtp_server': 'smtp.gmail.com',
    'from': 'your_bot@gmail.com',
    'to': 'your_email@gmail.com'
})

# Monitor account value
acct = client.get_account(account_hash).json()['securitiesAccount']
nav = acct['currentBalances']['liquidationValue']

if nav < 20000:  # Threshold
    alerts.send_alert(
        "⚠️ Low Account Value Alert",
        f"Account value dropped to ${nav:,.2f}"
    )
```

## 📊 Testing Checklist

Before going live, test these scenarios:

### Order Execution
- [ ] Market orders execute correctly
- [ ] Orders are reflected in account immediately
- [ ] Position updates are accurate
- [ ] Cash balance updates correctly with fees

### Error Handling
- [ ] Network timeouts are handled gracefully
- [ ] Invalid orders are rejected properly
- [ ] Rate limits don't cause crashes
- [ ] Token refresh works automatically

### Edge Cases
- [ ] Market open/close transitions
- [ ] Holiday handling
- [ ] Pre-market / after-hours (if applicable)
- [ ] Partial fills (if using limit orders)

### Performance
- [ ] Strategy completes within acceptable timeframe
- [ ] API calls don't exceed rate limits
- [ ] Memory usage is reasonable
- [ ] No memory leaks over extended runtime

## 🚨 Emergency Procedures

### Emergency Stop

Create a kill switch:

```python
# emergency_stop.py
from schwab import auth
from schwab.orders import equities as eq

client = auth.easy_client(...)  # Your auth
accounts = client.get_account_numbers().json()
account_hash = accounts[0]['hashValue']

# Get all positions
acct = client.get_account(account_hash, fields='positions').json()

# Close all positions
for pos in acct['securitiesAccount'].get('positions', []):
    symbol = pos['instrument']['symbol']
    qty = pos['longQuantity'] if pos['longQuantity'] > 0 else pos['shortQuantity']
    
    if qty > 0:
        order = eq.equity_sell_market(symbol, int(qty))
    else:
        order = eq.equity_buy_to_cover_market(symbol, int(abs(qty)))
    
    client.place_order(account_hash, order)
    print(f"Closed position: {symbol}")

print("All positions closed!")
```

Run this script manually if you need to immediately exit all positions.

## 📈 Gradual Rollout Strategy

1. **Week 1**: Paper trading with live data (if available)
2. **Week 2**: Live trading with $100 test capital
3. **Week 3**: Live trading with 10% of intended capital
4. **Week 4**: Live trading with 25% of intended capital
5. **Week 5+**: Gradually scale to full capital

Monitor closely at each stage before proceeding.

## 🔍 Monitoring Dashboard

Consider building a simple dashboard:

```python
# dashboard.py
import streamlit as st
from schwab import auth

st.title("Trading Bot Dashboard")

client = auth.easy_client(...)  # Your auth
account_hash = client.get_account_numbers().json()[0]['hashValue']

# Get account data
acct = client.get_account(account_hash, fields='positions').json()['securitiesAccount']

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Account Value", f"${acct['currentBalances']['liquidationValue']:,.2f}")
col2.metric("Cash", f"${acct['currentBalances']['cashBalance']:,.2f}")
col3.metric("Buying Power", f"${acct['currentBalances']['buyingPower']:,.2f}")

# Display positions
st.subheader("Current Positions")
for pos in acct.get('positions', []):
    st.write(f"{pos['instrument']['symbol']}: {pos['longQuantity']} shares")

# Run with: streamlit run dashboard.py
```

## 📞 Support & Resources

- **Schwab Developer Portal**: https://developer.schwab.com/
- **schwab-py Documentation**: https://schwab-py.readthedocs.io/
- **Discord Community**: (Join the schwab-py Discord)
- **Emergency**: Always have Schwab's customer support number handy

## ⚖️ Legal Disclaimer

**You are solely responsible for any trading decisions made using this software.**

- This simulator is not affiliated with Charles Schwab
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Consult with a financial advisor before trading
- Understand all applicable regulations and laws

---

**Remember**: Start small, monitor closely, and scale gradually. Good luck! 🚀