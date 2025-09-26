#!/usr/bin/env python3
"""
API Module
Flask blueprints and REST API endpoints for the trading bot
"""

from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import jwt
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)


def create_api_blueprint(orchestrator, broker_manager, strategy_manager, model_manager, risk_manager):
    """Create API blueprint with all endpoints"""
    
    api_bp = Blueprint('api', __name__)
    
    def token_required(f):
        """Decorator for JWT token authentication"""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
            
            try:
                if token.startswith('Bearer '):
                    token = token[7:]
                
                data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
                current_user_id = data['user_id']
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token has expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Token is invalid'}), 401
            
            return f(*args, **kwargs)
        return decorated
    
    # Health check endpoint
    @api_bp.route('/health', methods=['GET'])
    def health_check():
        """System health check"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'trading_active': orchestrator.is_running,
            'connected_accounts': len(orchestrator.connected_accounts),
            'supported_brokers': broker_manager.list_supported_brokers(),
            'version': '1.0.0'
        })
    
    # Authentication endpoints
    @api_bp.route('/auth/login', methods=['POST'])
    @limiter.limit("5 per minute")
    def login():
        """User authentication"""
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # Simple authentication for demo (replace with proper auth)
            if username == 'admin' and password == 'password':
                token = jwt.encode(
                    {
                        'user_id': 1,
                        'username': username,
                        'exp': datetime.utcnow().timestamp() + 3600  # 1 hour
                    },
                    current_app.config['SECRET_KEY'],
                    algorithm='HS256'
                )
                
                return jsonify({
                    'token': token,
                    'user': {'id': 1, 'username': username}
                })
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'error': 'Authentication failed'}), 500
    
    # Portfolio endpoints
    @api_bp.route('/summary', methods=['GET'])
    @token_required
    def get_portfolio_summary():
        """Get portfolio summary"""
        try:
            summary = orchestrator.get_portfolio_summary()
            return jsonify(summary)
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return jsonify({'error': 'Failed to get portfolio summary'}), 500
    
    @api_bp.route('/accounts', methods=['GET'])
    @token_required
    def get_accounts():
        """Get all connected accounts"""
        try:
            accounts = []
            for account_id, account_data in orchestrator.connected_accounts.items():
                connector = account_data['connector']
                balance = connector.get_balance()
                positions = connector.get_positions()
                
                accounts.append({
                    'account_id': account_id,
                    'broker': account_data['broker'],
                    'balance': {
                        'total': float(balance.total),
                        'available': float(balance.available),
                        'currency': balance.currency,
                        'equity': float(balance.equity),
                        'margin': float(balance.margin)
                    },
                    'positions_count': len(positions),
                    'connected_at': account_data['connected_at'].isoformat(),
                    'is_active': True
                })
            
            return jsonify(accounts)
            
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return jsonify({'error': 'Failed to get accounts'}), 500
    
    @api_bp.route('/accounts', methods=['POST'])
    @token_required
    def add_account():
        """Add new broker account"""
        try:
            data = request.get_json()
            broker_name = data.get('broker')
            account_config = data.get('config', {})
            
            if not broker_name:
                return jsonify({'error': 'Broker name is required'}), 400
            
            if not broker_manager.is_broker_supported(broker_name):
                return jsonify({
                    'error': f'Broker {broker_name} is not supported',
                    'supported_brokers': broker_manager.list_supported_brokers()
                }), 400
            
            success = orchestrator.add_account(broker_name, account_config)
            
            if success:
                return jsonify({'message': f'Account {broker_name} added successfully'})
            else:
                return jsonify({'error': f'Failed to add account {broker_name}'}), 500
                
        except Exception as e:
            logger.error(f"Error adding account: {e}")
            return jsonify({'error': 'Failed to add account'}), 500
    
    @api_bp.route('/accounts/<account_id>', methods=['DELETE'])
    @token_required
    def remove_account(account_id):
        """Remove broker account"""
        try:
            success = orchestrator.remove_account(account_id)
            
            if success:
                return jsonify({'message': f'Account {account_id} removed successfully'})
            else:
                return jsonify({'error': f'Account {account_id} not found'}), 404
                
        except Exception as e:
            logger.error(f"Error removing account: {e}")
            return jsonify({'error': 'Failed to remove account'}), 500
    
    # Strategy endpoints
    @api_bp.route('/strategies', methods=['GET'])
    @token_required
    def get_strategies():
        """Get all strategies"""
        try:
            available_strategies = strategy_manager.list_strategies()
            active_strategies = strategy_manager.list_active_strategies()
            
            return jsonify({
                'available': available_strategies,
                'active': active_strategies
            })
            
        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return jsonify({'error': 'Failed to get strategies'}), 500
    
    @api_bp.route('/strategies', methods=['POST'])
    @token_required
    def create_strategy():
        """Create new strategy instance"""
        try:
            data = request.get_json()
            strategy_type = data.get('type')
            strategy_name = data.get('name')
            config = data.get('config', {})
            
            if not strategy_type or not strategy_name:
                return jsonify({'error': 'Strategy type and name are required'}), 400
            
            strategy = strategy_manager.create_strategy(strategy_type, strategy_name, config)
            
            if strategy:
                success = strategy_manager.add_strategy(strategy)
                if success:
                    return jsonify({
                        'message': f'Strategy {strategy_name} created successfully',
                        'strategy': {
                            'name': strategy.name,
                            'type': strategy.__class__.__name__,
                            'config': strategy.get_config()
                        }
                    })
                else:
                    return jsonify({'error': f'Strategy {strategy_name} already exists'}), 409
            else:
                return jsonify({'error': f'Strategy type {strategy_type} not supported'}), 400
                
        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            return jsonify({'error': 'Failed to create strategy'}), 500
    
    @api_bp.route('/strategies/<strategy_name>', methods=['DELETE'])
    @token_required
    def remove_strategy(strategy_name):
        """Remove strategy"""
        try:
            success = strategy_manager.remove_strategy(strategy_name)
            
            if success:
                return jsonify({'message': f'Strategy {strategy_name} removed successfully'})
            else:
                return jsonify({'error': f'Strategy {strategy_name} not found'}), 404
                
        except Exception as e:
            logger.error(f"Error removing strategy: {e}")
            return jsonify({'error': 'Failed to remove strategy'}), 500
    
    @api_bp.route('/strategies/<strategy_name>/activate', methods=['POST'])
    @token_required
    def activate_strategy(strategy_name):
        """Activate strategy"""
        try:
            strategy = strategy_manager.get_strategy(strategy_name)
            
            if strategy:
                strategy.activate()
                return jsonify({'message': f'Strategy {strategy_name} activated'})
            else:
                return jsonify({'error': f'Strategy {strategy_name} not found'}), 404
                
        except Exception as e:
            logger.error(f"Error activating strategy: {e}")
            return jsonify({'error': 'Failed to activate strategy'}), 500
    
    @api_bp.route('/strategies/<strategy_name>/deactivate', methods=['POST'])
    @token_required
    def deactivate_strategy(strategy_name):
        """Deactivate strategy"""
        try:
            strategy = strategy_manager.get_strategy(strategy_name)
            
            if strategy:
                strategy.deactivate()
                return jsonify({'message': f'Strategy {strategy_name} deactivated'})
            else:
                return jsonify({'error': f'Strategy {strategy_name} not found'}), 404
                
        except Exception as e:
            logger.error(f"Error deactivating strategy: {e}")
            return jsonify({'error': 'Failed to deactivate strategy'}), 500
    
    # Trading control endpoints
    @api_bp.route('/trading/start', methods=['POST'])
    @token_required
    def start_trading():
        """Start trading engine"""
        try:
            orchestrator.start_trading()
            return jsonify({'message': 'Trading started', 'timestamp': datetime.now().isoformat()})
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            return jsonify({'error': 'Failed to start trading'}), 500
    
    @api_bp.route('/trading/stop', methods=['POST'])
    @token_required
    def stop_trading():
        """Stop trading engine"""
        try:
            orchestrator.stop_trading()
            return jsonify({'message': 'Trading stopped', 'timestamp': datetime.now().isoformat()})
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            return jsonify({'error': 'Failed to stop trading'}), 500
    
    @api_bp.route('/trading/status', methods=['GET'])
    @token_required
    def get_trading_status():
        """Get current trading status"""
        try:
            return jsonify({
                'is_running': orchestrator.is_running,
                'connected_accounts': len(orchestrator.connected_accounts),
                'active_strategies': len(orchestrator.active_strategies),
                'last_update': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return jsonify({'error': 'Failed to get trading status'}), 500
    
    # Positions and orders endpoints
    @api_bp.route('/positions', methods=['GET'])
    @token_required
    def get_all_positions():
        """Get all positions across all accounts"""
        try:
            all_positions = []
            
            for account_id, account_data in orchestrator.connected_accounts.items():
                connector = account_data['connector']
                positions = connector.get_positions()
                
                for pos in positions:
                    all_positions.append({
                        'account_id': account_id,
                        'broker': account_data['broker'],
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': float(pos.size),
                        'entry_price': float(pos.entry_price),
                        'current_price': float(pos.current_price),
                        'profit': float(pos.profit),
                        'profit_percentage': float(pos.profit_percentage),
                        'timestamp': pos.timestamp.isoformat()
                    })
            
            return jsonify(all_positions)
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return jsonify({'error': 'Failed to get positions'}), 500
    
    @api_bp.route('/orders', methods=['GET'])
    @token_required
    def get_all_orders():
        """Get all open orders across all accounts"""
        try:
            all_orders = []
            
            for account_id, account_data in orchestrator.connected_accounts.items():
                connector = account_data['connector']
                orders = connector.get_orders()
                
                for order in orders:
                    all_orders.append({
                        'account_id': account_id,
                        'broker': account_data['broker'],
                        'symbol': order.symbol,
                        'side': order.side.value,
                        'order_type': order.order_type.value,
                        'amount': float(order.amount),
                        'price': float(order.price) if order.price else None,
                        'status': order.status.value,
                        'filled_amount': float(order.filled_amount),
                        'order_id': order.order_id,
                        'timestamp': order.timestamp.isoformat() if order.timestamp else None
                    })
            
            return jsonify(all_orders)
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return jsonify({'error': 'Failed to get orders'}), 500
    
    # Market data endpoints
    @api_bp.route('/market/ticker/<symbol>', methods=['GET'])
    @token_required
    def get_ticker(symbol):
        """Get ticker data for symbol"""
        try:
            broker = request.args.get('broker', 'mt5')
            
            if not orchestrator.connected_accounts:
                return jsonify({'error': 'No connected accounts'}), 400
            
            # Find account for the specified broker
            account_data = None
            for acc_id, acc_data in orchestrator.connected_accounts.items():
                if acc_data['broker'] == broker:
                    account_data = acc_data
                    break
            
            if not account_data:
                return jsonify({'error': f'No connected account for broker {broker}'}), 400
            
            connector = account_data['connector']
            ticker = connector.get_ticker(symbol)
            
            return jsonify({
                'symbol': ticker.symbol,
                'bid': float(ticker.bid),
                'ask': float(ticker.ask),
                'last': float(ticker.last),
                'volume': float(ticker.volume),
                'high': float(ticker.high),
                'low': float(ticker.low),
                'change': float(ticker.change),
                'change_percentage': float(ticker.change_percentage),
                'timestamp': ticker.timestamp.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return jsonify({'error': f'Failed to get ticker for {symbol}'}), 500
    
    @api_bp.route('/market/symbols', methods=['GET'])
    @token_required
    def get_available_symbols():
        """Get available symbols from all brokers"""
        try:
            all_symbols = {}
            
            for account_id, account_data in orchestrator.connected_accounts.items():
                connector = account_data['connector']
                symbols = connector.get_available_symbols()
                all_symbols[account_id] = {
                    'broker': account_data['broker'],
                    'symbols': symbols[:100] if len(symbols) > 100 else symbols  # Limit to 100
                }
            
            return jsonify(all_symbols)
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return jsonify({'error': 'Failed to get symbols'}), 500
    
    # Risk management endpoints
    @api_bp.route('/risk/metrics', methods=['GET'])
    @token_required
    def get_risk_metrics():
        """Get current risk metrics"""
        try:
            metrics = risk_manager.get_portfolio_risk_metrics()
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return jsonify({'error': 'Failed to get risk metrics'}), 500
    
    # Broker information endpoints
    @api_bp.route('/brokers', methods=['GET'])
    def get_supported_brokers():
        """Get list of supported brokers"""
        try:
            brokers = broker_manager.list_supported_brokers()
            broker_info = []
            
            for broker in brokers:
                broker_info.append({
                    'name': broker,
                    'display_name': broker.replace('_', ' ').title(),
                    'category': self._get_broker_category(broker),
                    'supported': True
                })
            
            return jsonify(broker_info)
            
        except Exception as e:
            logger.error(f"Error getting brokers: {e}")
            return jsonify({'error': 'Failed to get brokers'}), 500
    
    def _get_broker_category(self, broker_name: str) -> str:
        """Get broker category"""
        if broker_name in ['mt5', 'metatrader5']:
            return 'Forex/CFD'
        elif broker_name in ['ibkr', 'interactive_brokers']:
            return 'Stocks/Options'
        elif broker_name in ['binance', 'coinbase_pro', 'kraken', 'bitfinex', 'huobi', 'okx', 'bybit', 'kucoin', 'bittrex']:
            return 'Cryptocurrency'
        else:
            return 'Other'
    
    # System endpoints
    @api_bp.route('/system/info', methods=['GET'])
    def get_system_info():
        """Get system information"""
        try:
            return jsonify({
                'name': 'AI/ML Trader Bot',
                'version': '1.0.0',
                'description': 'Advanced algorithmic trading system with AI/ML strategies',
                'author': 'szarastrefa',
                'supported_brokers': broker_manager.list_supported_brokers(),
                'supported_strategies': [s['type'] for s in strategy_manager.list_strategies()],
                'uptime': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return jsonify({'error': 'Failed to get system info'}), 500
    
    # Error handlers
    @api_bp.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @api_bp.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({'error': 'Method not allowed'}), 405
    
    @api_bp.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return api_bp


__all__ = ['create_api_blueprint', 'limiter']