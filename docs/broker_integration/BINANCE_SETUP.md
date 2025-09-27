# Binance - Instrukcja Konfiguracji

## Przegląd

Binance to jedna z największych giełd kryptowalut na świecie. Nasza integracja wykorzystuje Binance REST API oraz WebSocket dla danych w czasie rzeczywistym.

## Wymagania

- Konto na Binance.com
- Aktywny API Key z uprawnieniami do tradingu
- Python packages: `python-binance`, `websocket-client`
- Weryfikacja tozżsamości (KYC) dla tradingu

## Konfiguracja krok po kroku

### 1. Utworzenie API Key na Binance

1. **Zaloguj się na Binance.com**
2. **Idź do Account → API Management**
3. **Kliknij "Create API"**
4. **Ustaw nazwę**: `AI-ML-Trader-Bot`
5. **Włącz uprawnienia**:
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ✅ Enable Futures Trading (opcjonalnie)
   - ❌ Disable Withdrawals (bezpieczeństwo)

6. **Zapisz API Key i Secret** (Secret pokazuje się tylko raz!)

### 2. Konfiguracja IP Whitelist (Zalecane)

```bash
# Sprawdź swoje publiczne IP
curl ifconfig.me

# Dodaj IP do whitelist na Binance:
# Account -> API Management -> Edit -> IP Access Restriction
```

### 3. Dodanie konta w systemie

1. **Otwórz panel Settings w Web Dashboard**
2. **Kliknij "+ Dodaj Konto"**
3. **Wybierz "Binance"**
4. **Uzupełnij dane**:
   ```
   API Key: [Twój API Key z Binance]
   API Secret: [Twój API Secret z Binance]
   ```

### 4. Testowanie połączenia

```python
from binance.client import Client

# Testowe połączenie
client = Client(api_key, api_secret, testnet=True)

# Sprawdzenie statusu konta
account = client.get_account()
print(f"Account Status: {account['accountType']}")
print(f"Can Trade: {account['canTrade']}")

# Sprawdzenie sald
balances = client.get_account()['balances']
for balance in balances:
    if float(balance['free']) > 0:
        print(f"{balance['asset']}: {balance['free']}")
```

## Obsługiwane funkcje

### ✅ Zaimplementowane
- Pobieranie informacji o koncie i saldach
- Spot Trading (BUY/SELL)
- Market i Limit orders
- Pobieranie historii transakcji
- Monitoring otwartych zleceń
- WebSocket dla danych real-time
- Klines/Candlestick data

### 🔄 W przygotowaniu
- Futures Trading
- Margin Trading
- OCO Orders (One-Cancels-Other)
- Advanced order types
- DCA (Dollar Cost Averaging) strategies

## Obsługiwane pary walutowe

### Major Crypto Pairs
- **BTCUSDT** - Bitcoin/Tether
- **ETHUSDT** - Ethereum/Tether  
- **BNBUSDT** - Binance Coin/Tether
- **ADAUSDT** - Cardano/Tether
- **DOTUSDT** - Polkadot/Tether
- **SOLUSDT** - Solana/Tether

### Trading Pairs
- **BTCETH** - Bitcoin/Ethereum
- **ETHBNB** - Ethereum/Binance Coin
- **ADABTC** - Cardano/Bitcoin

## Przykłady strategii dla Binance

### Crypto Momentum Strategy
```python
class CryptoMomentumBinance:
    def __init__(self, symbol="BTCUSDT"):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.quantity = 0.001  # BTC
    
    def get_klines(self, interval='1h', limit=100):
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=interval,
            limit=limit
        )
        return [[float(x) for x in line] for line in klines]
    
    def calculate_rsi(self, prices, period=14):
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_signals(self):
        klines = self.get_klines()
        close_prices = [k[4] for k in klines]
        rsi = self.calculate_rsi(close_prices)
        
        # Simple momentum strategy
        if rsi < 30:  # Oversold
            return self.place_buy_order()
        elif rsi > 70:  # Overbought
            return self.place_sell_order()
    
    def place_buy_order(self):
        order = self.client.order_market_buy(
            symbol=self.symbol,
            quantity=self.quantity
        )
        return order
```

### DCA Strategy Example
```python
class DCAbinance:
    def __init__(self, symbol="BTCUSDT", amount_usdt=50):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.amount = amount_usdt
    
    def execute_dca_buy(self):
        """Execute Dollar Cost Averaging buy"""
        try:
            order = self.client.order_market_buy(
                symbol=self.symbol,
                quoteOrderQty=self.amount
            )
            
            print(f"DCA Buy executed: {order['executedQty']} {self.symbol}")
            return order
        except Exception as e:
            print(f"DCA Error: {e}")
            return None
```

## Limity API i Rate Limiting

### Request Limits
- **Request Weight**: 1200 per minute
- **Orders**: 10 per second
- **Raw Requests**: 5000 per 5 minutes

### Najlepsze praktyki
```python
import time
from binance.exceptions import BinanceAPIException

def safe_api_call(func, *args, **kwargs):
    """Bezpieczne wywołanie API z retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except BinanceAPIException as e:
            if e.code == -1003:  # Rate limit
                print(f"Rate limit hit, waiting {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise e
    
    raise Exception("Max retries exceeded")
```

## Troubleshooting

### Problem: "Invalid API Key"
**Rozwiązanie:**
```bash
# Sprawdź API Key
curl -X GET 'https://api.binance.com/api/v3/account' \
     -H 'X-MBX-APIKEY: YOUR_API_KEY'

# Jeśli błąd 401, sprawdź:
# 1. Poprawność API Key
# 2. Czy API nie wygasł
# 3. IP Whitelist settings
```

### Problem: "Signature invalid"
**Rozwiązanie:**
- Sprawdź poprawność API Secret
- Sprawdź synchronizację czasu serwera
- Użyj biblioteki `python-binance` zamiast raw requests

### Problem: "Insufficient balance"
**Rozwiązanie:**
```python
# Sprawdź dostępne saldo
balances = client.get_account()['balances']
usdt_balance = next(b for b in balances if b['asset'] == 'USDT')
print(f"Available USDT: {usdt_balance['free']}")
```

### Problem: "Market is closed"
**Rozwiązanie:**
- Binance działa 24/7, sprawdź status:
```python
status = client.get_system_status()
print(f"System Status: {status['msg']}")
```

## Bezpieczeństwo

### Zabezpieczenia API
- **Nie udostępniaj API Keys**
- **Włącz IP Whitelist**
- **Wyłącz Withdrawal permissions**
- **Używaj 2FA na koncie**
- **Regularnie rotuj API Keys**

### Recommended API Permissions
```
✅ Enable Reading
✅ Enable Spot & Margin Trading
❌ Enable Futures Trading (tylko jeśli potrzebne)
❌ Enable Withdrawals (NIGDY dla botów)
```

### Environment Variables
```bash
# .env file
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=false
```

## WebSocket Real-time Data

### Price Streams
```python
from binance import ThreadedWebsocketManager

def handle_socket_message(msg):
    print(f"Price Update: {msg['s']} = {msg['c']}")

# Start WebSocket
twm = ThreadedWebsocketManager(
    api_key=api_key, 
    api_secret=api_secret
)
twm.start()

# Subscribe to price updates
twm.start_symbol_ticker_socket(
    callback=handle_socket_message,
    symbol='BTCUSDT'
)
```

## API Reference

### Podstawowe operacje
```python
from binance.client import Client

client = Client(api_key, api_secret)

# Informacje o koncie
account = client.get_account()

# Aktualny kurs
ticker = client.get_symbol_ticker(symbol="BTCUSDT")

# Złożenie zlecenia market
order = client.order_market_buy(
    symbol='BTCUSDT',
    quantity=0.001
)

# Złożenie zlecenia limit
order = client.order_limit_buy(
    symbol='BTCUSDT',
    quantity=0.001,
    price='30000.00'
)

# Anulowanie zlecenia
client.cancel_order(symbol='BTCUSDT', orderId=order['orderId'])

# Historia transakcji
trades = client.get_my_trades(symbol='BTCUSDT')
```

## Monitoring i alerty

```python
# Monitor balansu
def check_balance_alerts():
    account = client.get_account()
    balances = {b['asset']: float(b['free']) for b in account['balances']}
    
    if balances.get('USDT', 0) < 100:  # Alert gdy USDT < 100
        send_alert("Low USDT balance")
    
    if balances.get('BTC', 0) > 0.1:   # Alert gdy BTC > 0.1
        send_alert("High BTC exposure")
```

## Wsparcie

Jeśli masz problemy z konfiguracją Binance:

1. **Sprawdź logi**: `docker-compose logs backend | grep binance`
2. **Test API**: `curl localhost:5000/api/brokers | grep Binance`  
3. **Binance Status**: https://www.binance.com/en/support/announcement
4. **GitHub Issues**: [Zgłoś problem](https://github.com/szarastrefa/AI-ML-Trader-Bot/issues)

## Przydatne linki

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Python-Binance Library](https://python-binance.readthedocs.io/)
- [Binance API Limits](https://www.binance.com/en/support/faq/360004492232)
- [Binance Trading Rules](https://www.binance.com/en/trade-rule)