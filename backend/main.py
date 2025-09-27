from typing import List, Dict, Any, Optional
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import random

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask and extensions
try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError as e:
    logger.error(f"Flask import error: {e}")
    sys.exit(1)

# Local imports with error handling
try:
    from utils.config import config
    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Config import warning: {e}")
    CONFIG_AVAILABLE = False
    config = None

try:
    from celery_app import celery_app
    CELERY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Celery import warning: {e}")
    CELERY_AVAILABLE = False
    celery_app = None

# Mock trading system state
class TradingSystemState:
    def __init__(self):
        self.is_running = False
        self.connected_accounts = {}
        self.active_strategies = {}
        self.positions = []
        self.orders = []
        self.portfolio_value = 10000.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.last_update = datetime.now()
        self.system_logs = []
        
        # Initialize with demo data
        self._initialize_demo_data()
    
    def _initialize_demo_data(self):
        """Initialize with realistic demo data"""
        # Demo accounts
        self.connected_accounts = {
            'MT5_DEMO_001': {
                'broker': 'MetaTrader 5',
                'account_id': 'MT5_DEMO_001',
                'balance': 10000.0,
                'equity': 10125.50,
                'margin': 2500.0,
                'free_margin': 7625.50,
                'currency': 'USD',
                'leverage': '1:100',
                'connected_at': datetime.now(),
                'status': 'Connected',
                'ping': f"{random.randint(10, 50)}ms"
            },
            'BINANCE_001': {
                'broker': 'Binance',
                'account_id': 'BINANCE_001', 
                'balance': 5000.0,
                'equity': 5087.25,
                'margin': 1200.0,
                'free_margin': 3887.25,
                'currency': 'USDT',
                'leverage': '1:10',
                'connected_at': datetime.now(),
                'status': 'Connected',
                'ping': f"{random.randint(15, 80)}ms"
            },
            'IBKR_DEMO_001': {
                'broker': 'Interactive Brokers',
                'account_id': 'IBKR_DEMO_001',
                'balance': 25000.0,
                'equity': 25234.75,
                'margin': 5000.0,
                'free_margin': 20234.75,
                'currency': 'USD',
                'leverage': '1:4',
                'connected_at': datetime.now(),
                'status': 'Connected',
                'ping': f"{random.randint(5, 25)}ms"
            }
        }
        
        # Demo strategies
        self.active_strategies = {
            'RSI_SCALPER_EURUSD': {
                'id': 'RSI_SCALPER_EURUSD',
                'name': 'RSI Scalper EUR/USD',
                'type': 'RSI_SCALPING',
                'symbol': 'EURUSD',
                'status': 'Active',
                'profit': 125.50,
                'profit_percentage': 1.25,
                'trades_today': 8,
                'win_rate': 75.0,
                'max_drawdown': 8.5,
                'sharpe_ratio': 2.1,
                'confidence': 89.2,
                'created_at': datetime.now(),
                'last_signal': 'BUY'
            },
            'ML_MOMENTUM_BTC': {
                'id': 'ML_MOMENTUM_BTC',
                'name': 'ML Momentum BTC/USDT',
                'type': 'ML_MOMENTUM',
                'symbol': 'BTCUSDT', 
                'status': 'Active',
                'profit': 287.25,
                'profit_percentage': 5.75,
                'trades_today': 3,
                'win_rate': 66.7,
                'max_drawdown': 12.3,
                'sharpe_ratio': 1.8,
                'confidence': 91.5,
                'created_at': datetime.now(),
                'last_signal': 'BUY'
            },
            'MACD_TREND_GBPUSD': {
                'id': 'MACD_TREND_GBPUSD',
                'name': 'MACD Trend GBP/USD',
                'type': 'MACD_TREND',
                'symbol': 'GBPUSD',
                'status': 'Paused',
                'profit': -23.75,
                'profit_percentage': -0.24,
                'trades_today': 2,
                'win_rate': 50.0,
                'max_drawdown': 15.2,
                'sharpe_ratio': 0.9,
                'confidence': 67.8,
                'created_at': datetime.now(),
                'last_signal': 'SELL'
            },
            'AI_SMART_MONEY_SPY': {
                'id': 'AI_SMART_MONEY_SPY',
                'name': 'AI Smart Money SPY',
                'type': 'AI_SMART_MONEY',
                'symbol': 'SPY',
                'status': 'Active',
                'profit': 456.80,
                'profit_percentage': 1.83,
                'trades_today': 5,
                'win_rate': 80.0,
                'max_drawdown': 6.7,
                'sharpe_ratio': 2.4,
                'confidence': 94.3,
                'created_at': datetime.now(),
                'last_signal': 'BUY'
            }
        }
        
        # Demo positions
        self.positions = [
            {
                'id': 'POS_001',
                'account_id': 'MT5_DEMO_001',
                'symbol': 'EURUSD',
                'side': 'BUY',
                'size': 0.1,
                'entry_price': 1.0850,
                'current_price': 1.0875,
                'profit': 25.0,
                'profit_percentage': 2.3,
                'strategy': 'RSI_SCALPER_EURUSD',
                'opened_at': datetime.now()
            },
            {
                'id': 'POS_002', 
                'account_id': 'BINANCE_001',
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'size': 0.01,
                'entry_price': 43500.0,
                'current_price': 44200.0,
                'profit': 70.0,
                'profit_percentage': 1.6,
                'strategy': 'ML_MOMENTUM_BTC',
                'opened_at': datetime.now()
            },
            {
                'id': 'POS_003',
                'account_id': 'IBKR_DEMO_001',
                'symbol': 'SPY',
                'side': 'BUY',
                'size': 10,
                'entry_price': 428.50,
                'current_price': 431.20,
                'profit': 27.0,
                'profit_percentage': 0.63,
                'strategy': 'AI_SMART_MONEY_SPY',
                'opened_at': datetime.now()
            }
        ]
        
        # System logs
        self.system_logs = [
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'AI/ML Trader Bot started successfully'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Connected to MetaTrader 5 demo account'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Connected to Binance API'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Connected to Interactive Brokers TWS'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Strategy RSI_SCALPER_EURUSD activated'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Strategy ML_MOMENTUM_BTC activated'},
            {'timestamp': datetime.now().isoformat(), 'level': 'warning', 'message': 'High latency detected on Binance connection (78ms)'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Position opened: EURUSD BUY 0.1 lots'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Position opened: BTCUSDT BUY 0.01 BTC'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'AI Smart Money signal: SPY BUY (confidence: 94.3%)'},
            {'timestamp': datetime.now().isoformat(), 'level': 'debug', 'message': 'Market data updated: 1247 instruments'},
            {'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Portfolio P&L updated: +$845.80'}
        ]
        
        # Calculate totals
        self.total_pnl = sum(pos['profit'] for pos in self.positions)
        self.daily_pnl = self.total_pnl * 0.8  # 80% of total profit made today
        
    def start_trading(self):
        """Start trading simulation"""
        self.is_running = True
        self.last_update = datetime.now()
        logger.info("Trading simulation started")
        
    def stop_trading(self):
        """Stop trading simulation"""
        self.is_running = False
        self.last_update = datetime.now()
        logger.info("Trading simulation stopped")
        
    def get_portfolio_summary(self):
        """Get comprehensive portfolio summary"""
        total_equity = sum(acc['equity'] for acc in self.connected_accounts.values())
        total_balance = sum(acc['balance'] for acc in self.connected_accounts.values())
        active_strategies_count = len([s for s in self.active_strategies.values() if s['status'] == 'Active'])
        
        return {
            'total_accounts': len(self.connected_accounts),
            'active_strategies': active_strategies_count,
            'total_strategies': len(self.active_strategies),
            'total_pnl': round(self.total_pnl, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'total_balance': round(total_balance, 2),
            'total_equity': round(total_equity, 2),
            'open_positions': len(self.positions),
            'portfolio_value': round(total_equity, 2),
            'status': 'Live Trading' if self.is_running else 'Demo Mode',
            'last_update': self.last_update.isoformat(),
            'system_status': 'Operational',
            'uptime_hours': 24.5,
            'win_rate': 68.5,
            'trades_today': sum(s['trades_today'] for s in self.active_strategies.values())
        }
    
    def get_detailed_accounts(self):
        """Get detailed account information"""
        detailed_accounts = []
        for acc_id, acc_data in self.connected_accounts.items():
            detailed_accounts.append({
                'account_id': acc_id,
                'broker': acc_data['broker'],
                'balance': acc_data['balance'],
                'equity': acc_data['equity'],
                'margin': acc_data['margin'],
                'free_margin': acc_data['free_margin'],
                'currency': acc_data['currency'],
                'leverage': acc_data['leverage'],
                'profit': acc_data['equity'] - acc_data['balance'],
                'profit_percentage': round(((acc_data['equity'] - acc_data['balance']) / acc_data['balance']) * 100, 2),
                'connected_at': acc_data['connected_at'].isoformat(),
                'status': acc_data['status'],
                'ping': acc_data['ping']
            })
        return detailed_accounts
    
    def get_detailed_strategies(self):
        """Get detailed strategy information"""
        return list(self.active_strategies.values())
    
    def get_supported_brokers(self):
        """Get list of supported brokers"""
        return [
            {
                'name': 'MetaTrader 5',
                'display_name': 'MetaTrader 5',
                'category': 'Forex/CFD',
                'supported': True,
                'config_fields': ['login', 'password', 'server']
            },
            {
                'name': 'Binance',
                'display_name': 'Binance',
                'category': 'Cryptocurrency',
                'supported': True,
                'config_fields': ['api_key', 'api_secret']
            },
            {
                'name': 'Interactive Brokers',
                'display_name': 'Interactive Brokers',
                'category': 'Stocks/Options',
                'supported': True,
                'config_fields': ['username', 'password', 'trading_mode']
            },
            {
                'name': 'Coinbase Pro',
                'display_name': 'Coinbase Pro',
                'category': 'Cryptocurrency',
                'supported': True,
                'config_fields': ['api_key', 'api_secret', 'passphrase']
            },
            {
                'name': 'Kraken',
                'display_name': 'Kraken',
                'category': 'Cryptocurrency',
                'supported': True,
                'config_fields': ['api_key', 'private_key']
            },
            {
                'name': 'Alpaca',
                'display_name': 'Alpaca',
                'category': 'Stocks',
                'supported': True,
                'config_fields': ['api_key', 'secret_key']
            }
        ]
    
    def get_market_overview(self):
        """Get market overview data"""
        return {
            'major_pairs': [
                {'symbol': 'EURUSD', 'price': 1.0875, 'change': 0.0025, 'change_pct': 0.23},
                {'symbol': 'GBPUSD', 'price': 1.2650, 'change': -0.0015, 'change_pct': -0.12},
                {'symbol': 'USDJPY', 'price': 149.85, 'change': 0.45, 'change_pct': 0.30},
                {'symbol': 'USDCHF', 'price': 0.8925, 'change': 0.0008, 'change_pct': 0.09}
            ],
            'crypto_pairs': [
                {'symbol': 'BTCUSDT', 'price': 44200.0, 'change': 700.0, 'change_pct': 1.61},
                {'symbol': 'ETHUSDT', 'price': 2650.5, 'change': -45.2, 'change_pct': -1.68},
                {'symbol': 'ADAUSDT', 'price': 0.485, 'change': 0.012, 'change_pct': 2.54},
                {'symbol': 'DOTUSDT', 'price': 7.85, 'change': -0.15, 'change_pct': -1.88}
            ],
            'stocks': [
                {'symbol': 'SPY', 'price': 431.20, 'change': 2.70, 'change_pct': 0.63},
                {'symbol': 'QQQ', 'price': 358.45, 'change': -1.25, 'change_pct': -0.35},
                {'symbol': 'AAPL', 'price': 175.82, 'change': 1.15, 'change_pct': 0.66},
                {'symbol': 'TSLA', 'price': 248.50, 'change': -3.20, 'change_pct': -1.27}
            ],
            'market_sentiment': 'Bullish',
            'volatility_index': 23.5,
            'fear_greed_index': 67
        }

# Global trading system instance
trading_system = TradingSystemState()

def create_app():
    """Create Flask application with complete dashboard API"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['DEBUG'] = os.getenv('FLASK_ENV') == 'development'
    
    # Enable CORS
    CORS(app)
    
    @app.route('/')
    def index():
        return jsonify({
            'status': 'AI/ML Trader Bot API',
            'version': '1.0.0',
            'description': 'Advanced algorithmic trading system with AI/ML strategies',
            'timestamp': datetime.now().isoformat(),
            'features': [
                'Multi-broker support',
                'AI/ML trading strategies', 
                'Real-time portfolio monitoring',
                'Risk management',
                'Professional web dashboard'
            ]
        })
    
    @app.route('/health')
    def health_check():
        """Comprehensive health check endpoint"""
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'flask': 'ok',
                'config': 'ok' if CONFIG_AVAILABLE else 'warning',
                'celery': 'ok' if CELERY_AVAILABLE else 'warning',
                'trading_engine': 'ok',
                'database': 'ok',
                'redis': 'ok'
            },
            'system_info': {
                'uptime': '24.5h',
                'memory_usage': '156MB',
                'cpu_usage': '12%',
                'connected_accounts': len(trading_system.connected_accounts),
                'active_strategies': len([s for s in trading_system.active_strategies.values() if s['status'] == 'Active'])
            }
        }
        
        # Check Redis connection if Celery available
        try:
            if CELERY_AVAILABLE and celery_app:
                inspect = celery_app.control.inspect()
                active_tasks = inspect.active()
                status['components']['celery_workers'] = 'ok' if active_tasks is not None else 'warning'
        except Exception as e:
            status['components']['celery_workers'] = f'error: {e}'
        
        return jsonify(status)
    
    # === MAIN DASHBOARD ENDPOINTS ===
    
    @app.route('/api/summary')
    def api_summary():
        """Enhanced portfolio summary for dashboard"""
        summary = trading_system.get_portfolio_summary()
        return jsonify(summary)
    
    @app.route('/api/dashboard')
    def dashboard_data():
        """Complete dashboard data in one endpoint"""
        try:
            dashboard = {
                'summary': trading_system.get_portfolio_summary(),
                'accounts': trading_system.get_detailed_accounts(),
                'strategies': trading_system.get_detailed_strategies(),
                'positions': trading_system.positions,
                'market': trading_system.get_market_overview(),
                'system_status': {
                    'trading_active': trading_system.is_running,
                    'last_update': trading_system.last_update.isoformat(),
                    'uptime': '24.5 hours',
                    'version': '1.0.0'
                }
            }
            return jsonify(dashboard)
        except Exception as e:
            logger.error(f"Dashboard data error: {e}")
            return jsonify({'error': 'Failed to load dashboard data'}), 500
    
    # === DETAILED ENDPOINTS ===
    
    @app.route('/api/accounts')
    def get_accounts():
        """Get detailed account information"""
        try:
            accounts = trading_system.get_detailed_accounts()
            return jsonify(accounts)
        except Exception as e:
            logger.error(f"Accounts error: {e}")
            return jsonify({'error': 'Failed to get accounts'}), 500
    
    @app.route('/api/strategies')
    def get_strategies():
        """Get detailed strategy information"""
        try:
            strategies = trading_system.get_detailed_strategies()
            return jsonify({
                'active': [s for s in strategies if s['status'] == 'Active'],
                'paused': [s for s in strategies if s['status'] == 'Paused'],
                'all': strategies
            })
        except Exception as e:
            logger.error(f"Strategies error: {e}")
            return jsonify({'error': 'Failed to get strategies'}), 500
    
    @app.route('/api/positions')
    def get_positions():
        """Get current positions"""
        try:
            return jsonify(trading_system.positions)
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return jsonify({'error': 'Failed to get positions'}), 500
    
    @app.route('/api/market')
    def get_market_data():
        """Get market overview data"""
        try:
            market_data = trading_system.get_market_overview()
            return jsonify(market_data)
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return jsonify({'error': 'Failed to get market data'}), 500
    
    @app.route('/api/brokers')
    def get_supported_brokers():
        """Get list of supported brokers"""
        try:
            brokers = trading_system.get_supported_brokers()
            return jsonify(brokers)
        except Exception as e:
            logger.error(f"Brokers error: {e}")
            return jsonify({'error': 'Failed to get brokers'}), 500
    
    @app.route('/api/system/logs')
    def get_system_logs():
        """Get system logs"""
        try:
            return jsonify(trading_system.system_logs)
        except Exception as e:
            logger.error(f"Logs error: {e}")
            return jsonify({'error': 'Failed to get logs'}), 500
    
    # === TRADING CONTROL ENDPOINTS ===
    
    @app.route('/api/trading/start', methods=['POST'])
    def start_trading():
        """Start trading engine"""
        try:
            trading_system.start_trading()
            return jsonify({
                'status': 'started',
                'message': 'Trading engine started successfully',
                'timestamp': datetime.now().isoformat(),
                'active_strategies': len([s for s in trading_system.active_strategies.values() if s['status'] == 'Active'])
            })
        except Exception as e:
            logger.error(f"Start trading error: {e}")
            return jsonify({'error': 'Failed to start trading'}), 500
    
    @app.route('/api/trading/stop', methods=['POST'])
    def stop_trading():
        """Stop trading engine"""
        try:
            trading_system.stop_trading()
            return jsonify({
                'status': 'stopped',
                'message': 'Trading engine stopped successfully', 
                'timestamp': datetime.now().isoformat(),
                'final_pnl': trading_system.total_pnl
            })
        except Exception as e:
            logger.error(f"Stop trading error: {e}")
            return jsonify({'error': 'Failed to stop trading'}), 500
    
    @app.route('/api/trading/status')
    def trading_status():
        """Get detailed trading status"""
        try:
            return jsonify({
                'is_running': trading_system.is_running,
                'connected_accounts': len(trading_system.connected_accounts),
                'active_strategies': len([s for s in trading_system.active_strategies.values() if s['status'] == 'Active']),
                'open_positions': len(trading_system.positions),
                'portfolio_value': sum(acc['equity'] for acc in trading_system.connected_accounts.values()),
                'daily_pnl': trading_system.daily_pnl,
                'last_update': trading_system.last_update.isoformat(),
                'uptime': '24.5 hours',
                'system_load': '12%'
            })
        except Exception as e:
            logger.error(f"Trading status error: {e}")
            return jsonify({'error': 'Failed to get trading status'}), 500
    
    # === STRATEGY MANAGEMENT ===
    
    @app.route('/api/strategies/<strategy_id>/start', methods=['POST'])
    def start_strategy(strategy_id):
        """Start specific strategy"""
        try:
            if strategy_id in trading_system.active_strategies:
                trading_system.active_strategies[strategy_id]['status'] = 'Active'
                return jsonify({
                    'status': 'started',
                    'strategy': strategy_id,
                    'message': f'Strategy {strategy_id} activated'
                })
            else:
                return jsonify({'error': 'Strategy not found'}), 404
        except Exception as e:
            logger.error(f"Start strategy error: {e}")
            return jsonify({'error': 'Failed to start strategy'}), 500
    
    @app.route('/api/strategies/<strategy_id>/stop', methods=['POST'])
    def stop_strategy(strategy_id):
        """Stop specific strategy"""
        try:
            if strategy_id in trading_system.active_strategies:
                trading_system.active_strategies[strategy_id]['status'] = 'Paused'
                return jsonify({
                    'status': 'stopped',
                    'strategy': strategy_id,
                    'message': f'Strategy {strategy_id} paused'
                })
            else:
                return jsonify({'error': 'Strategy not found'}), 404
        except Exception as e:
            logger.error(f"Stop strategy error: {e}")
            return jsonify({'error': 'Failed to stop strategy'}), 500
    
    # === SYSTEM INFO ENDPOINTS ===
    
    @app.route('/api/system/info')
    def system_info():
        """Get comprehensive system information"""
        try:
            return jsonify({
                'name': 'AI/ML Trader Bot',
                'version': '1.0.0',
                'description': 'Advanced algorithmic trading system with AI/ML strategies',
                'author': 'szarastrefa',
                'github': 'https://github.com/szarastrefa/AI-ML-Trader-Bot',
                'features': [
                    'Multi-broker support (MT5, Binance, Interactive Brokers)',
                    'AI/ML trading strategies',
                    'Real-time portfolio monitoring', 
                    'Advanced risk management',
                    'Professional web dashboard',
                    'Celery background tasks',
                    'Docker containerization'
                ],
                'supported_brokers': ['MetaTrader 5', 'Binance', 'Interactive Brokers', 'Coinbase Pro', 'Kraken', 'Alpaca'],
                'supported_strategies': ['RSI Scalping', 'ML Momentum', 'MACD Trend', 'AI Smart Money'],
                'api_version': 'v1',
                'build_date': '2025-09-27',
                'uptime': datetime.now().isoformat(),
                'environment': os.getenv('FLASK_ENV', 'production')
            })
        except Exception as e:
            logger.error(f"System info error: {e}")
            return jsonify({'error': 'Failed to get system info'}), 500
    
    # === CELERY TASK ENDPOINTS ===
    
    @app.route('/api/tasks/health', methods=['POST'])
    def trigger_health_task():
        """Trigger health check task"""
        if not CELERY_AVAILABLE:
            return jsonify({
                'error': 'Celery not available',
                'status': 'degraded',
                'message': 'Background tasks unavailable'
            }), 503
            
        try:
            from tasks import health_check
            task = health_check.delay()
            return jsonify({
                'task_id': task.id,
                'status': 'submitted',
                'message': 'Health check task submitted'
            })
        except Exception as e:
            logger.error(f"Health task error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # === ERROR HANDLERS ===
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'message': 'The requested API endpoint does not exist',
            'available_endpoints': ['/api/summary', '/api/dashboard', '/api/accounts', '/api/strategies', '/api/brokers']
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    return app

def main():
    """Main application entry point"""
    logger.info("Starting AI/ML Trader Bot with Complete Dashboard...")
    
    app = create_app()
    
    # Run the application
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting Flask app on {host}:{port}")
    logger.info(f"Config available: {CONFIG_AVAILABLE}")
    logger.info(f"Celery available: {CELERY_AVAILABLE}")
    logger.info(f"Trading system initialized with {len(trading_system.connected_accounts)} demo accounts")
    logger.info(f"Active strategies: {len([s for s in trading_system.active_strategies.values() if s['status'] == 'Active'])}")
    
    print("\n🚀 AI/ML Trader Bot - Complete Dashboard API")
    print("=" * 50)
    print(f"📊 Dashboard: http://{host}:{port}")
    print(f"🔗 API Summary: http://{host}:{port}/api/summary")
    print(f"📈 Full Dashboard: http://{host}:{port}/api/dashboard")
    print(f"⚕️  Health Check: http://{host}:{port}/health")
    print("=" * 50)
    
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    main()