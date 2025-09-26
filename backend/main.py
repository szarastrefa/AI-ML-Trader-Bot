#!/usr/bin/env python3
"""
Main backend application for AI/ML Trading Bot
Central orchestrator for trading operations, broker connections, and ML strategies
"""

import os
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
    from flask_sqlalchemy import SQLAlchemy
except ImportError as e:
    print(f"Flask dependencies missing: {e}")
    sys.exit(1)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    print("APScheduler not available - using basic scheduling")
    SCHEDULER_AVAILABLE = False
    BackgroundScheduler = None

try:
    from connectors import BrokerManager
except ImportError:
    print("BrokerManager not available - using mock")
    BrokerManager = None

try:
    from strategies import StrategyManager
except ImportError:
    print("StrategyManager not available - using mock")
    StrategyManager = None

try:
    from utils.config import Config
except ImportError:
    print("Config module not available - using defaults")
    Config = None

try:
    from utils.logger import setup_logger
except ImportError:
    print("Logger module not available - using basic logging")
    setup_logger = None

try:
    from utils.database import init_db
except ImportError:
    print("Database module not available - using basic setup")
    init_db = None

# Try to import Celery - make it optional
try:
    from tasks import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    print("Celery not available - background tasks will run in-process")
    CELERY_AVAILABLE = False
    celery_app = None

import atexit

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
if Config:
    config = Config()
else:
    # Fallback configuration
    class MockConfig:
        def get(self, key, default=None):
            return os.environ.get(key, default)
    config = MockConfig()

app.config['SQLALCHEMY_DATABASE_URI'] = config.get('DATABASE_URL', 'sqlite:///trader.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = config.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize database
db = SQLAlchemy(app)
if init_db:
    init_db(app)

# Initialize managers
if BrokerManager:
    broker_manager = BrokerManager()
else:
    # Mock broker manager
    class MockBrokerManager:
        def get_connector(self, name):
            return None
    broker_manager = MockBrokerManager()

if StrategyManager:
    strategy_manager = StrategyManager()
else:
    # Mock strategy manager
    class MockStrategyManager:
        def list_strategies(self):
            return []
        def import_model(self, file, name):
            return False
        def export_model(self, strategy_id):
            return None
    strategy_manager = MockStrategyManager()

# Setup logging
if setup_logger:
    logger = setup_logger('main', config.get('LOG_LEVEL', 'INFO'))
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Background scheduler for periodic tasks
if SCHEDULER_AVAILABLE:
    scheduler = BackgroundScheduler()
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
else:
    scheduler = None


class TradingOrchestrator:
    """Main orchestrator for trading operations"""
    
    def __init__(self):
        self.is_running = False
        self.connected_accounts = {}
        self.active_strategies = {}
        
    def start_trading(self):
        """Start the trading engine"""
        self.is_running = True
        logger.info("Trading orchestrator started")
        
    def stop_trading(self):
        """Stop the trading engine"""
        self.is_running = False
        logger.info("Trading orchestrator stopped")
        
    def add_account(self, broker_name: str, account_config: Dict) -> bool:
        """Add a new trading account"""
        try:
            connector = broker_manager.get_connector(broker_name)
            if connector and hasattr(connector, 'connect') and connector.connect(account_config):
                account_id = f"{broker_name}_{account_config.get('account_id', 'default')}"
                self.connected_accounts[account_id] = {
                    'broker': broker_name,
                    'connector': connector,
                    'config': account_config,
                    'connected_at': datetime.now()
                }
                logger.info(f"Added account: {account_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to add account {broker_name}: {e}")
        return False
        
    def remove_account(self, account_id: str) -> bool:
        """Remove a trading account"""
        if account_id in self.connected_accounts:
            connector = self.connected_accounts[account_id]['connector']
            if hasattr(connector, 'disconnect'):
                connector.disconnect()
            del self.connected_accounts[account_id]
            logger.info(f"Removed account: {account_id}")
            return True
        return False
        
    def get_portfolio_summary(self) -> Dict:
        """Get aggregated portfolio summary"""
        total_equity = 0.0
        total_pnl = 0.0
        account_summaries = []
        
        for account_id, account_data in self.connected_accounts.items():
            try:
                connector = account_data['connector']
                balance = connector.get_balance() if hasattr(connector, 'get_balance') else {'equity': 0.0}
                positions = connector.get_positions() if hasattr(connector, 'get_positions') else []
                
                account_equity = balance.get('equity', 0.0)
                account_pnl = sum(pos.get('profit', 0.0) for pos in positions)
                
                total_equity += account_equity
                total_pnl += account_pnl
                
                account_summaries.append({
                    'account_id': account_id,
                    'broker': account_data['broker'],
                    'equity': account_equity,
                    'pnl': account_pnl,
                    'positions_count': len(positions)
                })
                
            except Exception as e:
                logger.error(f"Error getting summary for {account_id}: {e}")
                
        return {
            'total_equity': total_equity,
            'total_pnl': total_pnl,
            'accounts': account_summaries,
            'last_updated': datetime.now().isoformat()
        }


# Initialize orchestrator
orchestrator = TradingOrchestrator()


# API Routes
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'trading_active': orchestrator.is_running,
        'connected_accounts': len(orchestrator.connected_accounts),
        'components': {
            'broker_manager': BrokerManager is not None,
            'strategy_manager': StrategyManager is not None,
            'scheduler': SCHEDULER_AVAILABLE,
            'celery': CELERY_AVAILABLE
        }
    })


@app.route('/api/summary')
def get_summary():
    """Get portfolio summary"""
    try:
        summary = orchestrator.get_portfolio_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/accounts', methods=['GET'])
def get_accounts():
    """Get all connected accounts"""
    accounts = []
    for account_id, account_data in orchestrator.connected_accounts.items():
        accounts.append({
            'account_id': account_id,
            'broker': account_data['broker'],
            'connected_at': account_data['connected_at'].isoformat(),
            'status': 'connected'
        })
    return jsonify({'accounts': accounts})


@app.route('/api/accounts', methods=['POST'])
def add_account():
    """Add a new trading account"""
    try:
        data = request.get_json()
        broker_name = data.get('broker')
        account_config = data.get('config', {})
        
        if not broker_name:
            return jsonify({'error': 'Broker name is required'}), 400
            
        success = orchestrator.add_account(broker_name, account_config)
        if success:
            return jsonify({'message': 'Account added successfully'})
        else:
            return jsonify({'error': 'Failed to add account'}), 400
            
    except Exception as e:
        logger.error(f"Error adding account: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/accounts/<account_id>', methods=['DELETE'])
def remove_account(account_id):
    """Remove a trading account"""
    try:
        success = orchestrator.remove_account(account_id)
        if success:
            return jsonify({'message': 'Account removed successfully'})
        else:
            return jsonify({'error': 'Account not found'}), 404
            
    except Exception as e:
        logger.error(f"Error removing account: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get available strategies"""
    try:
        strategies = strategy_manager.list_strategies()
        return jsonify({'strategies': strategies})
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies/import', methods=['POST'])
def import_strategy():
    """Import a new strategy model"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        strategy_name = request.form.get('name', file.filename)
        
        success = strategy_manager.import_model(file, strategy_name)
        if success:
            return jsonify({'message': 'Strategy imported successfully'})
        else:
            return jsonify({'error': 'Failed to import strategy'}), 400
            
    except Exception as e:
        logger.error(f"Error importing strategy: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategies/<strategy_id>/export', methods=['GET'])
def export_strategy(strategy_id):
    """Export a strategy model"""
    try:
        file_path = strategy_manager.export_model(strategy_id)
        if file_path:
            return send_from_directory(
                os.path.dirname(file_path),
                os.path.basename(file_path),
                as_attachment=True
            )
        else:
            return jsonify({'error': 'Strategy not found'}), 404
            
    except Exception as e:
        logger.error(f"Error exporting strategy: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """Start the trading engine"""
    try:
        orchestrator.start_trading()
        return jsonify({'message': 'Trading started'})
    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    """Stop the trading engine"""
    try:
        orchestrator.stop_trading()
        return jsonify({'message': 'Trading stopped'})
    except Exception as e:
        logger.error(f"Error stopping trading: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent log entries"""
    try:
        limit = request.args.get('limit', 100, type=int)
        level = request.args.get('level', 'INFO')
        
        # This is a simplified implementation
        # In production, you'd read from log files or database
        logs = [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': 'System initialized successfully',
                'module': 'main'
            }
        ]
        return jsonify({'logs': logs})
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify({'error': str(e)}), 500


# Periodic tasks
def update_market_data():
    """Update market data from all connected brokers"""
    if not orchestrator.is_running:
        return
        
    logger.info("Updating market data...")
    for account_id, account_data in orchestrator.connected_accounts.items():
        try:
            connector = account_data['connector']
            # Update market data for this broker
            if hasattr(connector, 'update_market_data'):
                connector.update_market_data()
        except Exception as e:
            logger.error(f"Error updating market data for {account_id}: {e}")


def execute_strategies():
    """Execute trading strategies"""
    if not orchestrator.is_running:
        return
        
    logger.info("Executing strategies...")
    # This would be implemented based on active strategies
    # and their assigned accounts


# Schedule periodic tasks
if scheduler:
    try:
        scheduler.add_job(
            func=update_market_data,
            trigger="interval",
            seconds=60,  # Update every minute
            id='update_market_data'
        )

        scheduler.add_job(
            func=execute_strategies,
            trigger="interval",
            seconds=300,  # Execute every 5 minutes
            id='execute_strategies'
        )
        logger.info("Scheduled periodic tasks")
    except Exception as e:
        logger.error(f"Error scheduling tasks: {e}")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    orchestrator.stop_trading()
    if scheduler:
        scheduler.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting AI/ML Trading Bot backend on port {port}")
    logger.info(f"Available components:")
    logger.info(f"  - Broker Manager: {BrokerManager is not None}")
    logger.info(f"  - Strategy Manager: {StrategyManager is not None}")
    logger.info(f"  - Background Scheduler: {SCHEDULER_AVAILABLE}")
    logger.info(f"  - Celery Tasks: {CELERY_AVAILABLE}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)