# 🤖 AI/ML Trader Bot - Advanced Algorithmic Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)

Zaawansowany system tradingu algorytmicznego z wykorzystaniem sztucznej inteligencji i uczenia maszynowego. System obsługuje wielu brokerów, implementuje strategie AI/ML oraz oferuje intuicyjny Web GUI dla kompletnego zarządzania portfelem.

## 🎯 Kluczowe Funkcjonalności

### 📊 **Multi-Broker Support**
- **MetaTrader 5** - Forex, CFD, commodities (XM, IC Markets, RoboForex, InstaForex, FBS, XTB, Admiral Markets)
- **Giełdy Kryptowalut (CCXT)** - 15+ exchanges (Binance, Coinbase Pro, Kraken, Bitfinex, Huobi, OKX, Bybit, KuCoin)
- **Interactive Brokers** - Akcje, opcje, futures (TWS API ready)
- **Unified Interface** - Jednolity interfejs dla wszystkich platform

### 🧠 **AI/ML Trading Strategies**
- **Smart Money Concept** - BOS, ChoCh, Order Blocks, Fair Value Gaps
- **Depth of Market Analysis** - Real-time order flow, iceberg detection, volume profile
- **Machine Learning Models** - Support for scikit-learn, PyTorch, TensorFlow, ONNX
- **Custom Strategy Framework** - BaseStrategy dla własnych implementacji

### 🌐 **Professional Web GUI**
- **React SPA** - Nowoczesny single-page application
- **Real-time Monitoring** - WebSocket updates, live charts
- **Portfolio Dashboard** - Complete overview, P&L tracking
- **Strategy Management** - Drag-and-drop assignment, real-time control

### 🛡️ **Advanced Risk Management**
- **Position Sizing** - Kelly Criterion, volatility-based, correlation adjustment
- **Portfolio Protection** - VaR calculation, drawdown monitoring, circuit breakers
- **Automated Controls** - Emergency stop, daily loss limits, margin monitoring
- **Dynamic Adjustments** - Real-time risk parameter updates

### 🐳 **Production-Ready Infrastructure**
- **Docker Containerization** - Scalable, portable deployment
- **PostgreSQL Database** - Persistent data storage, performance tracking
- **Redis Caching** - Real-time data, session management
- **Celery Workers** - Background tasks, async processing
- **Comprehensive Logging** - Multi-handler, structured logging

## 🚀 Szybki Start

### Wymagania
- **Docker** i **Docker Compose**
- **4GB RAM** (minimum), **8GB RAM** (recommended)
- **2 CPU cores** (minimum)
- **10GB disk space**

### 1. Klonowanie Repozytorium
```bash
git clone https://github.com/szarastrefa/AI-ML-Trader-Bot.git
cd AI-ML-Trader-Bot
```

### 2. Konfiguracja (Opcjonalnie)
```bash
# Skopiuj przykładowy plik konfiguracyjny
cp config/examples/.env.example .env

# Edytuj konfigurację (API keys, broker credentials)
nano .env
```

### 3. Uruchomienie Systemu
```bash
# Uruchomienie całego stacku
docker-compose up --build

# Lub w tle
docker-compose up --build -d
```

### 4. Dostęp do Aplikacji
- **Frontend (Web GUI)**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs (jeśli włączone)
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### 5. Domyślne Dane Logowania
- **Username**: `admin`
- **Password**: `password`
- ⚠️ **Zmień natychmiast po pierwszym logowaniu!**

## 📁 Struktura Projektu

```
AI-ML-Trader-Bot/
├── 📁 backend/                 # Python backend (Flask + WebSocket)
│   ├── 📄 main.py             # Główna aplikacja (11,506 bytes)
│   ├── 📄 requirements.txt    # Zależności Python (1,394 bytes)
│   ├── 📁 connectors/         # Integracje brokerów
│   │   ├── 📄 __init__.py     # Connector manager (2,228 bytes)
│   │   ├── 📄 base_connector.py # Abstract base (8,054 bytes)
│   │   └── 📄 ccxt_connector.py # CCXT crypto (19,801 bytes)
│   ├── 📁 strategies/         # Strategie handlowe
│   │   ├── 📄 __init__.py     # Strategy framework (10,774 bytes)
│   │   ├── 📄 smc_strategy.py # Smart Money Concept (25,792 bytes)
│   │   ├── 📄 dom_analysis.py # DOM analysis (30,475 bytes)
│   │   └── 📄 ml_strategy.py  # ML strategy (30,831 bytes)
│   ├── 📁 ml/                 # Machine Learning framework
│   │   └── 📄 __init__.py     # ML models & features (22,921 bytes)
│   ├── 📁 risk/               # Risk management
│   │   └── 📄 __init__.py     # Risk manager (29,913 bytes)
│   ├── 📁 data/               # Data pipeline
│   │   └── 📄 __init__.py     # Real-time data (26,848 bytes)
│   ├── 📁 api/                # REST API endpoints
│   │   └── 📄 __init__.py     # Flask blueprint (20,976 bytes)
│   └── 📁 utils/              # Narzędzia pomocnicze
│       ├── 📄 __init__.py     # Utils init (311 bytes)
│       ├── 📄 config.py       # Configuration (11,985 bytes)
│       ├── 📄 logger.py       # Logging system (10,450 bytes)
│       └── 📄 database.py     # Database models (15,324 bytes)
├── 📁 frontend/               # React frontend
│   ├── 📄 package.json       # NPM dependencies (1,945 bytes)
│   └── 📁 src/
│       └── 📄 App.js          # Main React app (7,302 bytes)
├── 📁 docker/                 # Docker configurations
│   ├── 📄 backend.Dockerfile  # Backend image (1,829 bytes)
│   ├── 📄 frontend.Dockerfile # Frontend image (1,266 bytes)
│   └── 📄 nginx.conf          # Nginx config (3,962 bytes)
├── 📁 config/                 # Konfiguracje
│   └── 📁 examples/
│       └── 📄 .env.example    # Environment variables (6,548 bytes)
├── 📁 docs/                   # Dokumentacja
│   └── 📄 README.md           # Szczegółowa dokumentacja (13,596 bytes)
├── 📁 tests/                  # Unit tests
│   └── 📄 test_strategies.py  # Strategy tests (18,413 bytes)
├── 📄 docker-compose.yml      # Services orchestration (3,249 bytes)
├── 📄 .gitignore             # Git ignore rules (6,173 bytes)
└── 📄 README.md              # Ten plik
```

## 🎛️ Konfiguracja Brokerów

### MetaTrader 5
```bash
# W pliku .env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo  # lub serwer twojego brokera

# Popularne serwery brokerów:
# XM: XMGlobal-Real, XMGlobal-Demo
# IC Markets: ICMarkets-Live, ICMarkets-Demo  
# RoboForex: RoboForex-ECN, RoboForex-Demo
```

### Giełdy Kryptowalut
```bash
# Binance
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret
BINANCE_SANDBOX=true

# Coinbase Pro
COINBASE_PRO_API_KEY=your_api_key
COINBASE_PRO_SECRET=your_secret  
COINBASE_PRO_PASSPHRASE=your_passphrase
COINBASE_PRO_SANDBOX=true
```

### Interactive Brokers
```bash
# TWS/Gateway konfiguracja
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497 for paper, 7496 for live
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=DU123456
```

## 📈 Strategie Handlowe

### Smart Money Concept (SMC)
```python
# Konfiguracja strategii SMC
SMC_LOOKBACK_PERIOD=20
SMC_BOS_CONFIRMATION_CANDLES=2
SMC_ORDER_BLOCK_STRENGTH=3
SMC_FVG_MIN_SIZE=0.0005
SMC_RISK_REWARD_RATIO=2.0
SMC_CONFIDENCE_THRESHOLD=0.7
```

**Funkcjonalności:**
- ✅ Break of Structure (BOS) detection
- ✅ Change of Character (ChoCh) analysis  
- ✅ Order Blocks identification
- ✅ Fair Value Gaps (FVG) detection
- ✅ Supply/Demand zones mapping
- ✅ Multi-timeframe analysis

### Depth of Market Analysis
```python
# Konfiguracja analizy DOM
DOM_ORDER_BOOK_LEVELS=20
DOM_IMBALANCE_THRESHOLD=2.0
DOM_LARGE_ORDER_THRESHOLD=10000
DOM_ICEBERG_DETECTION=true
```

**Funkcjonalności:**
- ✅ Order flow analysis
- ✅ Volume imbalance detection
- ✅ Iceberg orders identification
- ✅ Market maker activity tracking
- ✅ Volume profile calculation
- ✅ Support/resistance from DOM

### Machine Learning Strategy
```python
# Konfiguracja strategii ML
ML_MODEL_UPDATE_INTERVAL=3600
ML_RETRAIN_INTERVAL=86400
ML_FEATURE_HISTORY_DAYS=30
ML_MIN_TRAINING_SAMPLES=1000
ML_CONFIDENCE_THRESHOLD=0.6
```

**Obsługiwane formaty modeli:**
- 📦 **Scikit-learn** (.pkl, .joblib)
- 🔥 **PyTorch** (.pth)
- 🧠 **TensorFlow** (.h5, .savedmodel)
- ⚡ **ONNX** (.onnx)
- 📊 **Custom JSON** (.json)

## 🛡️ Risk Management

### Position Sizing Algorithms
- **Kelly Criterion** - Optimal position sizing based on win rate
- **Fixed Fractional** - Percentage of capital per trade
- **Volatility-based** - Dynamic sizing based on market volatility
- **Correlation Adjustment** - Reduce size for correlated positions

### Portfolio Protection
- **Value at Risk (VaR)** - 1-day and 1-week calculations
- **Maximum Drawdown** - Real-time monitoring and alerts
- **Circuit Breakers** - Automatic trading suspension
- **Daily Loss Limits** - Prevent excessive losses
- **Margin Monitoring** - Real-time margin level tracking

## 📊 API Endpoints

### Authentication
```bash
# Login
POST /api/auth/login
{
  "username": "admin",
  "password": "password"
}
```

### Portfolio Management
```bash
# Get portfolio summary
GET /api/summary

# Get all accounts
GET /api/accounts

# Add new account
POST /api/accounts
{
  "broker": "binance",
  "config": {
    "api_key": "your_key",
    "secret": "your_secret",
    "sandbox": true
  }
}
```

### Trading Control
```bash
# Start trading
POST /api/trading/start

# Stop trading  
POST /api/trading/stop

# Get trading status
GET /api/trading/status
```

### Strategy Management
```bash
# Get all strategies
GET /api/strategies

# Create new strategy
POST /api/strategies
{
  "type": "smc",
  "name": "SMC_EURUSD", 
  "config": {
    "lookback_period": 20,
    "confidence_threshold": 0.7
  }
}

# Activate strategy
POST /api/strategies/{strategy_name}/activate
```

## 🐳 Docker Deployment

### Produkcyjne Wdrożenie
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with custom environment
DOCKER_BUILDKIT=1 docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up --scale backend=3 -d
```

### Monitoring
```bash
# Check container health
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Performance metrics
docker stats
```

### Backup & Restore
```bash
# Backup database
docker-compose exec db pg_dump -U trader ai_trader_db > backup.sql

# Restore database
docker-compose exec -T db psql -U trader ai_trader_db < backup.sql
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_strategies.py -v

# Run with coverage
python -m pytest tests/ --cov=backend --cov-report=html
```

### Integration Tests
```bash
# Test broker connections
python tests/test_connectors.py

# Test strategy performance
python tests/test_strategy_performance.py

# Load testing
python tests/load_test.py
```

## 📚 Dokumentacja

Szczegółowa dokumentacja znajduje się w folderze `docs/`:

- **[Setup Guide](docs/README.md)** - Kompletny przewodnik instalacji
- **[API Reference](docs/api.md)** - Dokumentacja REST API
- **[Strategy Development](docs/strategies.md)** - Tworzenie własnych strategii
- **[Broker Integration](docs/brokers.md)** - Integracja z brokerami
- **[Risk Management](docs/risk.md)** - System zarządzania ryzykiem
- **[Troubleshooting](docs/troubleshooting.md)** - Rozwiązywanie problemów

## 🤝 Wkład w Rozwój

### Development Setup
```bash
# Clone repository
git clone https://github.com/szarastrefa/AI-ML-Trader-Bot.git
cd AI-ML-Trader-Bot

# Create development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r backend/requirements.txt
npm install --prefix frontend/

# Run in development mode
python backend/main.py  # Backend
npm start --prefix frontend/  # Frontend
```

### Contributing Guidelines
1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`) 
5. Open a **Pull Request**

## 📄 Licencja

Ten projekt jest objęty licencją MIT - zobacz plik [LICENSE](LICENSE) dla szczegółów.

## ⚠️ Disclaimer

**RYZYKO INWESTYCYJNE:** Trading na rynkach finansowych niesie ze sobą znaczne ryzyko straty kapitału. Ten system jest narzędziem pomocniczym i nie gwarantuje zysków. Zawsze:

- 🔴 **Testuj na kontach demo** przed użyciem środków rzeczywistych
- 🔴 **Inwestuj tylko środki**, na których stratę możesz sobie pozwolić
- 🔴 **Monitoruj pozycje** i zarządzaj ryzykiem
- 🔴 **Konsultuj się z doradcą finansowym** przed podejmowaniem decyzji

## 📞 Support

- **GitHub Issues**: [Zgłoś problem](https://github.com/szarastrefa/AI-ML-Trader-Bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/szarastrefa/AI-ML-Trader-Bot/discussions)
- **Email**: support@tradingbot.dev
- **Documentation**: [docs.tradingbot.dev](https://docs.tradingbot.dev)

---

<div align="center">
  <strong>🚀 Zaawansowany Trading Bot z AI/ML - Production Ready 🚀</strong><br>
  <em>Stworzony przez szarastrefa | 2025</em>
</div>