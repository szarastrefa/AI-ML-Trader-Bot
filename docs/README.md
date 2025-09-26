# AI/ML Trader Bot - Dokumentacja

Zaawansowany system tradingu algorytmicznego z wykorzystaniem sztucznej inteligencji i uczenia maszynowego, wspierający wielu brokerów i platformy handlowe.

## 🎯 Przegląd Systemu

AI/ML Trader Bot to kompleksowe rozwiązanie do automatycznego handlu finansowego, które łączy:

- **Multi-Broker Support** - Obsługa MetaTrader 5, Interactive Brokers, giełd kryptowalut
- **Strategie AI/ML** - Smart Money Concept, Depth of Market Analysis, Machine Learning
- **Real-time Monitoring** - Web GUI z live tracking i portfolio visualization
- **Risk Management** - Zaawansowane zarządzanie ryzykiem z automated controls
- **Containerization** - Pełna konteneryzacja Docker z production-ready deployment

## 🏗️ Architektura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Databases     │
│   (React)       │◄──►│   (Python)      │◄──►│ PostgreSQL/Redis│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────►│  WebSocket      │◄─────────────┘
                        │  Real-time      │
                        └─────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Broker Integrations                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  MetaTrader 5   │  CCXT Crypto    │  Interactive Brokers        │
│  • XM           │  • Binance      │  • Stocks, Options          │
│  • IC Markets   │  • Coinbase Pro │  • Futures, Forex           │
│  • RoboForex    │  • Kraken       │  • Multi-currency           │
│  • InstaForex   │  • Bitfinex     │  • TWS API                  │
│  • FBS, XTB     │  • Huobi, OKX   │  • Market data              │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 🚀 Szybki Start

### Wymagania

- **Docker & Docker Compose** (zalecane)
- **Python 3.11+** (dla lokalnej instalacji)
- **Node.js 18+** (dla lokalnej instalacji frontendu)
- **PostgreSQL 15+** (jeśli nie używasz Docker)
- **Redis 7+** (jeśli nie używasz Docker)

### Instalacja z Docker (Zalecane)

```bash
# 1. Klonowanie repozytorium
git clone https://github.com/szarastrefa/AI-ML-Trader-Bot.git
cd AI-ML-Trader-Bot

# 2. Konfiguracja zmiennych środowiskowych
cp config/examples/.env.example .env
nano .env  # Edytuj konfigurację

# 3. Uruchomienie całego systemu
docker-compose up --build

# 4. Dostęp do aplikacji
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000
# PostgreSQL: localhost:5432
# Redis: localhost:6379
```

### Lokalna Instalacja

#### Backend (Python)

```bash
# 1. Przejście do katalogu backend
cd backend

# 2. Utworzenie wirtualnego środowiska
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
venv\Scripts\activate  # Windows

# 3. Instalacja zależności
pip install -r requirements.txt

# 4. Konfiguracja bazy danych
export DATABASE_URL="postgresql://user:pass@localhost:5432/trader_db"
export REDIS_URL="redis://localhost:6379/0"

# 5. Uruchomienie backendu
python main.py
```

#### Frontend (React)

```bash
# 1. Przejście do katalogu frontend
cd frontend

# 2. Instalacja zależności
npm install

# 3. Konfiguracja API URL
echo "REACT_APP_API_URL=http://localhost:5000" > .env.local
echo "REACT_APP_WS_URL=ws://localhost:5000" >> .env.local

# 4. Uruchomienie frontendu
npm start
```

## 🔌 Konfiguracja Brokerów

### MetaTrader 5

```yaml
# config/brokers/mt5.yaml
mt5:
  login: "12345678"
  password: "your_password"
  server: "MetaQuotes-Demo"
  path: "/path/to/terminal64.exe"  # Opcjonalne
```

**Wspierani brokerzy:**
- XM (server: "XMGlobal-Real", "XMGlobal-Demo")
- IC Markets (server: "ICMarkets-Live", "ICMarkets-Demo")
- RoboForex (server: "RoboForex-ECN", "RoboForex-Demo")
- InstaForex (server: "InstaForex-Server", "InstaForex-Demo")
- FBS (server: "FBS-Real", "FBS-Demo")
- XTB (server: "XTB-Real", "XTB-Demo")
- Admiral Markets (server: "AdmiralMarkets-Live", "AdmiralMarkets-Demo")

### Giełdy Kryptowalut (CCXT)

```yaml
# config/brokers/crypto.yaml
binance:
  api_key: "your_api_key"
  secret: "your_secret_key"
  sandbox: true  # false dla live trading

coinbase_pro:
  api_key: "your_api_key"
  secret: "your_secret_key"
  passphrase: "your_passphrase"
  sandbox: true

kraken:
  api_key: "your_api_key"
  secret: "your_secret_key"
```

**Wspierane giełdy:**
- Binance, Coinbase Pro, Kraken
- Bitfinex, Huobi, OKX
- Bybit, KuCoin, Bittrex
- Bitstamp, Gemini

### Interactive Brokers

```yaml
# config/brokers/ibkr.yaml
ibkr:
  host: "127.0.0.1"
  port: 7497  # 7496 dla live, 7497 dla paper trading
  client_id: 1
  account: "DU123456"  # Paper trading account
```

## 🧠 Strategie AI/ML

### Smart Money Concept (SMC)

```python
# Konfiguracja strategii SMC
smc_config = {
    'lookback_period': 20,
    'min_structure_points': 3,
    'bos_confirmation_candles': 2,
    'order_block_strength': 3,
    'fvg_min_size': 0.0005,
    'risk_reward_ratio': 2.0,
    'timeframes': ['1h', '4h', '1d'],
    'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
    'volume_threshold': 1.5,
    'confidence_threshold': 0.7
}
```

**Funkcjonalności SMC:**
- Break of Structure (BOS) detection
- Change of Character (ChoCh) analysis
- Order Blocks identification
- Fair Value Gaps (FVG) detection
- Supply/Demand zone mapping

### Depth of Market Analysis

```python
# Konfiguracja DOM analysis
dom_config = {
    'order_book_levels': 20,
    'imbalance_threshold': 2.0,
    'volume_profile_periods': 100,
    'iceberg_detection': True,
    'flow_analysis_window': 10
}
```

### Machine Learning Models

```python
# Import/Export modeli
from ml import ModelManager

model_manager = ModelManager()

# Import modelu
model_manager.import_model('model.pkl', 'my_strategy')

# Export modelu
model_manager.export_model('my_strategy', 'exported_model.pkl')
```

**Wspierane formaty:**
- `.pkl` (scikit-learn, joblib)
- `.onnx` (PyTorch, TensorFlow)
- `.json` (konfiguracje i metadata)

## 🛡️ Risk Management

### Position Sizing

```python
# Algorytmy position sizing
risk_config = {
    'max_risk_per_trade': 0.02,  # 2% kapitału na trade
    'max_portfolio_risk': 0.10,  # 10% całkowitego ryzyka
    'kelly_criterion': True,
    'volatility_adjustment': True,
    'correlation_limit': 0.7
}
```

### Automated Controls

- **Circuit Breakers** - Zatrzymanie przy przekroczeniu strat
- **Dynamic Stop-Loss** - Dostosowanie do volatility
- **Emergency Shutdown** - Natychmiastowe zamknięcie pozycji
- **Portfolio Monitoring** - Real-time risk metrics

## 📊 API Reference

### Authentication

```bash
# Login
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Response
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {"id": 1, "username": "admin"}
}
```

### Portfolio Management

```bash
# Get portfolio summary
curl -X GET http://localhost:5000/api/summary \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response
{
  "total_equity": 10000.00,
  "total_pnl": 250.50,
  "accounts": [
    {
      "account_id": "mt5_demo",
      "broker": "mt5",
      "equity": 5000.00,
      "pnl": 125.25,
      "positions_count": 3
    }
  ],
  "risk_metrics": {
    "var_1d": 150.00,
    "max_drawdown": 0.05,
    "sharpe_ratio": 1.85
  }
}
```

### Account Management

```bash
# Add broker account
curl -X POST http://localhost:5000/api/accounts \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "broker": "mt5",
    "config": {
      "login": "12345678",
      "password": "password",
      "server": "MetaQuotes-Demo"
    }
  }'

# Get all accounts
curl -X GET http://localhost:5000/api/accounts \
  -H "Authorization: Bearer YOUR_TOKEN"

# Remove account
curl -X DELETE http://localhost:5000/api/accounts/mt5_demo \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Strategy Management

```bash
# Get available strategies
curl -X GET http://localhost:5000/api/strategies \
  -H "Authorization: Bearer YOUR_TOKEN"

# Import ML model
curl -X POST http://localhost:5000/api/strategies/import \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@model.pkl" \
  -F "name=my_ml_strategy"

# Export strategy
curl -X GET http://localhost:5000/api/strategies/my_strategy/export \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o exported_strategy.pkl
```

### Real-time WebSocket

```javascript
// JavaScript WebSocket client
const socket = io('http://localhost:5000', {
  auth: {
    token: 'YOUR_JWT_TOKEN'
  }
});

// Subscribe to portfolio updates
socket.emit('subscribe_updates', { type: 'portfolio' });

// Listen for real-time data
socket.on('portfolio_update', (data) => {
  console.log('Portfolio update:', data);
});

socket.on('trading_status', (data) => {
  console.log('Trading status:', data.status);
});

socket.on('account_added', (data) => {
  console.log('New account:', data.account_id);
});
```

## 🧪 Testing

### Unit Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Coverage report
pytest --cov=backend tests/

# Frontend tests
cd frontend
npm test

# E2E tests
npm run test:e2e
```

### Integration Tests

```bash
# Test broker connections
pytest tests/integration/test_brokers.py -v

# Test strategy execution
pytest tests/integration/test_strategies.py -v

# Test API endpoints
pytest tests/integration/test_api.py -v
```

## 🔧 Troubleshooting

### Częste Problemy

#### MetaTrader 5 Connection Issues

```bash
# Check if MT5 terminal is running
ps aux | grep terminal64

# Verify API settings in MT5
# Tools -> Options -> Expert Advisors
# ✓ Allow automated trading
# ✓ Allow DLL imports
# ✓ Allow import of external experts
```

#### Docker Issues

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs frontend

# Restart services
docker-compose restart backend

# Clean rebuild
docker-compose down
docker-compose up --build
```

#### Database Connection

```bash
# Test PostgreSQL connection
psql -h localhost -p 5432 -U trader_user -d ai_trader_db

# Check Redis connection
redis-cli ping

# Reset database
docker-compose down -v
docker-compose up -d db
```

#### WebSocket Issues

```bash
# Check WebSocket endpoint
wscat -c ws://localhost:5000/socket.io/?EIO=4&transport=websocket

# Verify CORS settings
# Backend: CORS(app, cors_allowed_origins="*")

# Check firewall settings
sudo ufw status
sudo ufw allow 5000
```

## 📈 Monitoring i Logging

### Log Levels

```python
# config/logging.yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - console
    - file
    - rotate
```

### Metrics Collection

```bash
# Health check
curl http://localhost:5000/health

# System metrics
curl http://localhost:5000/api/metrics

# Trading performance
curl http://localhost:5000/api/performance
```

## 🔐 Security

### Best Practices

1. **API Keys** - Przechowuj w zmiennych środowiskowych
2. **Passwords** - Używaj silnych haseł i 2FA gdzie możliwe
3. **Network** - Ogranicz dostęp do portów (firewall)
4. **Updates** - Regularnie aktualizuj dependencies
5. **Backups** - Automatyczne kopie strategii i konfiguracji

### Production Deployment

```bash
# Use production environment
export FLASK_ENV=production
export DEBUG=False

# Configure secure secrets
export SECRET_KEY="very-long-random-secret-key"
export JWT_SECRET="another-secure-secret"

# Enable HTTPS
export USE_SSL=true
export SSL_CERT="/path/to/cert.pem"
export SSL_KEY="/path/to/key.pem"

# Configure production database
export DATABASE_URL="postgresql://user:pass@prod-db:5432/trader_db"
```

## 📚 Dodatkowe Zasoby

- [GitHub Issues](https://github.com/szarastrefa/AI-ML-Trader-Bot/issues)
- [Wiki](https://github.com/szarastrefa/AI-ML-Trader-Bot/wiki)
- [Discussions](https://github.com/szarastrefa/AI-ML-Trader-Bot/discussions)
- [Release Notes](https://github.com/szarastrefa/AI-ML-Trader-Bot/releases)

## ⚠️ Disclaimer

Trading niesie ryzyko strat finansowych. System jest dostarczany "as-is" bez gwarancji. Zawsze:

- Testuj strategie na kontach demo przed wdrożeniem live
- Nie inwestuj więcej niż możesz stracić
- Monitoruj pozycje i zarządzaj ryzykiem
- Poznaj regulacje prawne w swojej jurysdykcji

## 📄 Licencja

MIT License - Zobacz [LICENSE](../LICENSE) dla szczegółów.

---

*AI/ML Trader Bot - Zaawansowany system tradingu algorytmicznego*