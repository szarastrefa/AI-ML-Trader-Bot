# MetaTrader 5 - Instrukcja Konfiguracji

## Przegląd

MetaTrader 5 jest jedną z najpopularniejszych platform do tradingu na rynku Forex i CFD. Nasza integracja wykorzystuje bibliotekę Python do komunikacji z terminalem MT5.

## Wymagania

- MetaTrader 5 Terminal zainstalowany na Windows/Wine
- Konto demo lub live u brokera obsługującego MT5
- Python package: `MetaTrader5`
- Włączone Expert Advisors i algorytmic trading

## Konfiguracja krok po kroku

### 1. Instalacja MetaTrader 5

```bash
# Pobierz MT5 z oficjalnej strony
# https://www.metatrader5.com/en/download

# Dla Linux (przez Wine)
sudo apt install wine
wine mt5setup.exe
```

### 2. Konfiguracja terminala

1. **Otwórz MT5 Terminal**
2. **Zaloguj się do swojego konta**
3. **Włącz Expert Advisors**:
   - Menu: `Tools` → `Options`
   - Zakładka: `Expert Advisors`
   - Zaznacz: `Allow algorithmic trading`
   - Zaznacz: `Allow WebRequest for listed URL`
   - Dodaj URL: `https://localhost:5000`

4. **Konfiguracja Auto Trading**:
   - Kliknij przycisk `Auto Trading` w toolbar
   - Powinien się podświetlić na zielono

### 3. Dodanie konta w systemie

1. **Otwórz panel Settings w Web Dashboard**
2. **Kliknij "+ Dodaj Konto"**
3. **Wybierz "MetaTrader 5"**
4. **Uzupełnij dane**:
   ```
   Login: [Twój numer konta MT5]
   Hasło: [Hasło do konta]
   Serwer: [Serwer brokera, np. "Demo-MetaQuotes"]
   ```

### 4. Testowanie połączenia

```python
import MetaTrader5 as mt5

# Inicjalizacja połączenia
if not mt5.initialize():
    print("Initialize failed")
    quit()

# Sprawdzenie wersji
print(f"MetaTrader5 version: {mt5.version()}")

# Informacje o koncie
account_info = mt5.account_info()
print(f"Account: {account_info.login}")
print(f"Balance: {account_info.balance}")
print(f"Equity: {account_info.equity}")
```

## Obsługiwane funkcje

### ✅ Zaimplementowane
- Pobieranie informacji o koncie (balance, equity, margin)
- Otwieranie pozycji (BUY/SELL)
- Zamykanie pozycji
- Pobieranie historii transakcji
- Monitoring otwartych pozycji
- Pobieranie danych cenowych (OHLCV)

### 🔄 W przygotowaniu
- Zaawansowane typy zleceń (Stop Loss, Take Profit)
- Trailing Stop
- Partial close positions
- Custom indicators integration

## Przykłady strategii dla MT5

### RSI Scalper Strategy
```python
class RSIScalperMT5:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.rsi_period = 14
        self.lot_size = 0.01
    
    def get_rsi(self):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100)
        close_prices = [rate[4] for rate in rates]
        return self.calculate_rsi(close_prices)
    
    def check_signals(self):
        rsi = self.get_rsi()
        
        if rsi < 30:  # Oversold
            return "BUY"
        elif rsi > 70:  # Overbought
            return "SELL"
        else:
            return "HOLD"
```

## Troubleshooting

### Problem: "Initialize failed"
**Rozwiązanie:**
```bash
# 1. Sprawdź czy MT5 jest uruchomiony
ps aux | grep terminal

# 2. Sprawdź uprawnienia
sudo chmod 777 /home/.wine/drive_c/Program\ Files/MetaTrader\ 5/

# 3. Restartuj MT5 Terminal
killall terminal.exe
wine ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/terminal.exe
```

### Problem: "Invalid account"
**Rozwiązanie:**
- Sprawdź poprawność danych logowania
- Upewnij się że konto jest aktywne
- Sprawdź czy serwer jest dostępny

### Problem: "Trade is disabled"
**Rozwiązanie:**
- Włącz Expert Advisors w terminalu
- Sprawdź czy AutoTrading jest aktywny
- Upewnij się że rynek jest otwarty

### Problem: "Not enough money"
**Rozwiązanie:**
- Sprawdź saldo konta
- Zmniejsz wielkość pozycji (lot size)
- Sprawdź wymagania margin

## Bezpieczeństwo

### Zabezpieczenia konta
- **Nigdy nie udostępniaj danych logowania**
- **Używaj kont demo do testowania**
- **Ustaw limity dzienne dla tradingu**
- **Regularnie sprawdzaj logi transakcji**

### Recommended settings
```python
# Maksymalne pozycje jednocześnie
MAX_POSITIONS = 3

# Maksymalna wielkość pozycji
MAX_LOT_SIZE = 0.1

# Stop Loss (w pipsach)
DEFAULT_STOP_LOSS = 50

# Take Profit (w pipsach) 
DEFAULT_TAKE_PROFIT = 100
```

## API Reference

### Główne funkcje

```python
# Połączenie
mt5.initialize(login=12345, password="password", server="Demo-Server")

# Informacje o koncie
account_info = mt5.account_info()

# Otwarcie pozycji
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "EURUSD",
    "volume": 0.1,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick("EURUSD").ask,
    "deviation": 20,
    "magic": 234000,
    "comment": "AI/ML Bot",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)

# Zamknięcie pozycji
mt5.Close(symbol="EURUSD", deviation=20)

# Pobranie danych cenowych
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1000)

# Rozłączenie
mt5.shutdown()
```

## Wsparcie

Jeśli masz problemy z konfiguracją MT5:

1. **Sprawdź logi systemu**: `docker-compose logs backend | grep MT5`
2. **Sprawdź status w panelu**: Settings → Connected Accounts
3. **Testuj połączenie**: `curl localhost:5000/api/brokers | grep MT5`
4. **GitHub Issues**: [Zgłoś problem](https://github.com/szarastrefa/AI-ML-Trader-Bot/issues)

## Przydatne linki

- [MetaTrader 5 Official](https://www.metatrader5.com/)
- [MQL5 Documentation](https://www.mql5.com/en/docs)
- [Python MT5 Package](https://pypi.org/project/MetaTrader5/)
- [MT5 Python Examples](https://github.com/nickmccullum/algorithmic-trading-python)