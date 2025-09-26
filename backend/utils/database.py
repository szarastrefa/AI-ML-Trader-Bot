#!/usr/bin/env python3
"""
Database Initialization and Models
SQLAlchemy models and database setup for trading data
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON

logger = logging.getLogger(__name__)

# Global database instance
db = None


def init_db(app: Flask) -> SQLAlchemy:
    """Initialize database with Flask app
    
    Args:
        app: Flask application instance
        
    Returns:
        SQLAlchemy: Database instance
    """
    global db
    
    if db is None:
        db = SQLAlchemy(app)
    
    # Import models to ensure they're registered
    from .models import *
    
    # Create tables
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    return db


def get_db() -> SQLAlchemy:
    """Get database instance
    
    Returns:
        SQLAlchemy: Database instance
    """
    global db
    return db


class BaseModel(db.Model):
    """Base model with common fields"""
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
    
    def update(self, **kwargs):
        """Update model attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()


class Account(BaseModel):
    """Trading account model"""
    __tablename__ = 'accounts'
    
    account_id = db.Column(db.String(100), unique=True, nullable=False)
    broker_name = db.Column(db.String(50), nullable=False)
    account_type = db.Column(db.String(20), nullable=False)  # 'live' or 'demo'
    currency = db.Column(db.String(10), nullable=False, default='USD')
    initial_balance = db.Column(db.Numeric(15, 2), default=0.0)
    current_balance = db.Column(db.Numeric(15, 2), default=0.0)
    equity = db.Column(db.Numeric(15, 2), default=0.0)
    margin = db.Column(db.Numeric(15, 2), default=0.0)
    free_margin = db.Column(db.Numeric(15, 2), default=0.0)
    is_active = db.Column(db.Boolean, default=True)
    last_update = db.Column(db.DateTime, default=datetime.utcnow)
    config = db.Column(JSON)  # Store broker-specific configuration
    
    # Relationships
    positions = db.relationship('Position', backref='account', lazy=True)
    orders = db.relationship('Order', backref='account', lazy=True)
    trades = db.relationship('Trade', backref='account', lazy=True)
    
    def __repr__(self):
        return f'<Account {self.account_id} ({self.broker_name})>'


class Position(BaseModel):
    """Trading position model"""
    __tablename__ = 'positions'
    
    account_id = db.Column(db.String(100), db.ForeignKey('accounts.account_id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    side = db.Column(db.String(10), nullable=False)  # 'long' or 'short'
    size = db.Column(db.Numeric(15, 8), nullable=False)
    entry_price = db.Column(db.Numeric(15, 8), nullable=False)
    current_price = db.Column(db.Numeric(15, 8), default=0.0)
    profit = db.Column(db.Numeric(15, 2), default=0.0)
    profit_percentage = db.Column(db.Numeric(8, 4), default=0.0)
    unrealized_pnl = db.Column(db.Numeric(15, 2), default=0.0)
    realized_pnl = db.Column(db.Numeric(15, 2), default=0.0)
    broker_position_id = db.Column(db.String(100))
    is_open = db.Column(db.Boolean, default=True)
    opened_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime)
    metadata = db.Column(JSON)  # Additional position data
    
    def __repr__(self):
        return f'<Position {self.symbol} {self.side} {self.size}>'


class Order(BaseModel):
    """Trading order model"""
    __tablename__ = 'orders'
    
    account_id = db.Column(db.String(100), db.ForeignKey('accounts.account_id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    side = db.Column(db.String(10), nullable=False)  # 'buy' or 'sell'
    order_type = db.Column(db.String(20), nullable=False)  # 'market', 'limit', 'stop', etc.
    amount = db.Column(db.Numeric(15, 8), nullable=False)
    price = db.Column(db.Numeric(15, 8))
    stop_price = db.Column(db.Numeric(15, 8))
    take_profit = db.Column(db.Numeric(15, 8))
    stop_loss = db.Column(db.Numeric(15, 8))
    filled_amount = db.Column(db.Numeric(15, 8), default=0.0)
    status = db.Column(db.String(20), default='pending')
    broker_order_id = db.Column(db.String(100))
    commission = db.Column(db.Numeric(10, 4), default=0.0)
    placed_at = db.Column(db.DateTime, default=datetime.utcnow)
    filled_at = db.Column(db.DateTime)
    cancelled_at = db.Column(db.DateTime)
    metadata = db.Column(JSON)
    
    def __repr__(self):
        return f'<Order {self.symbol} {self.side} {self.amount} @ {self.price}>'


class Trade(BaseModel):
    """Executed trade model"""
    __tablename__ = 'trades'
    
    account_id = db.Column(db.String(100), db.ForeignKey('accounts.account_id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    side = db.Column(db.String(10), nullable=False)
    amount = db.Column(db.Numeric(15, 8), nullable=False)
    price = db.Column(db.Numeric(15, 8), nullable=False)
    commission = db.Column(db.Numeric(10, 4), default=0.0)
    pnl = db.Column(db.Numeric(15, 2), default=0.0)
    strategy_name = db.Column(db.String(100))
    signal_id = db.Column(db.String(100))  # Reference to signal that caused trade
    broker_trade_id = db.Column(db.String(100))
    executed_at = db.Column(db.DateTime, default=datetime.utcnow)
    metadata = db.Column(JSON)
    
    def __repr__(self):
        return f'<Trade {self.symbol} {self.side} {self.amount} @ {self.price}>'


class Strategy(BaseModel):
    """Trading strategy model"""
    __tablename__ = 'strategies'
    
    name = db.Column(db.String(100), unique=True, nullable=False)
    strategy_type = db.Column(db.String(50), nullable=False)  # 'smc', 'ml', 'dom', etc.
    description = db.Column(db.Text)
    config = db.Column(JSON)  # Strategy configuration
    is_active = db.Column(db.Boolean, default=False)
    assigned_accounts = db.Column(JSON)  # List of account IDs
    performance_metrics = db.Column(JSON)  # Performance data
    model_path = db.Column(db.String(255))  # Path to ML model file
    model_metadata = db.Column(JSON)  # Model metadata
    last_signal_at = db.Column(db.DateTime)
    
    # Relationships
    signals = db.relationship('Signal', backref='strategy', lazy=True)
    
    def __repr__(self):
        return f'<Strategy {self.name} ({self.strategy_type})>'


class Signal(BaseModel):
    """Trading signal model"""
    __tablename__ = 'signals'
    
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    action = db.Column(db.String(10), nullable=False)  # 'buy', 'sell', 'hold'
    confidence = db.Column(db.Numeric(5, 4), nullable=False)  # 0.0000 to 1.0000
    price = db.Column(db.Numeric(15, 8))
    stop_loss = db.Column(db.Numeric(15, 8))
    take_profit = db.Column(db.Numeric(15, 8))
    signal_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_executed = db.Column(db.Boolean, default=False)
    executed_at = db.Column(db.DateTime)
    execution_price = db.Column(db.Numeric(15, 8))
    pnl = db.Column(db.Numeric(15, 2))
    metadata = db.Column(JSON)  # Signal-specific data
    
    def __repr__(self):
        return f'<Signal {self.symbol} {self.action} (confidence: {self.confidence})>'


class MarketData(BaseModel):
    """Market data storage model"""
    __tablename__ = 'market_data'
    
    symbol = db.Column(db.String(20), nullable=False)
    broker = db.Column(db.String(50), nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)  # '1m', '5m', '1h', '1d', etc.
    timestamp = db.Column(db.DateTime, nullable=False)
    open_price = db.Column(db.Numeric(15, 8), nullable=False)
    high_price = db.Column(db.Numeric(15, 8), nullable=False)
    low_price = db.Column(db.Numeric(15, 8), nullable=False)
    close_price = db.Column(db.Numeric(15, 8), nullable=False)
    volume = db.Column(db.Numeric(20, 8), default=0.0)
    tick_volume = db.Column(db.Integer, default=0)
    spread = db.Column(db.Numeric(10, 8), default=0.0)
    
    # Create compound index for efficient queries
    __table_args__ = (
        db.Index('idx_market_data_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        db.Index('idx_market_data_broker_timestamp', 'broker', 'timestamp'),
    )
    
    def __repr__(self):
        return f'<MarketData {self.symbol} {self.timeframe} @ {self.timestamp}>'


class RiskMetric(BaseModel):
    """Risk metrics storage model"""
    __tablename__ = 'risk_metrics'
    
    account_id = db.Column(db.String(100), db.ForeignKey('accounts.account_id'))
    metric_type = db.Column(db.String(50), nullable=False)  # 'var', 'drawdown', 'sharpe', etc.
    metric_value = db.Column(db.Numeric(15, 6), nullable=False)
    calculation_period = db.Column(db.String(20))  # '1d', '1w', '1m', etc.
    calculation_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    metadata = db.Column(JSON)  # Additional metric data
    
    def __repr__(self):
        return f'<RiskMetric {self.metric_type}: {self.metric_value}>'


class SystemLog(BaseModel):
    """System log entries model"""
    __tablename__ = 'system_logs'
    
    level = db.Column(db.String(20), nullable=False)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    logger_name = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    module = db.Column(db.String(100))
    function = db.Column(db.String(100))
    line_number = db.Column(db.Integer)
    account_id = db.Column(db.String(100))  # Optional account context
    symbol = db.Column(db.String(20))  # Optional symbol context
    strategy_name = db.Column(db.String(100))  # Optional strategy context
    exception_info = db.Column(db.Text)  # Exception traceback if present
    metadata = db.Column(JSON)  # Additional log data
    
    # Index for efficient log queries
    __table_args__ = (
        db.Index('idx_system_logs_timestamp_level', 'created_at', 'level'),
        db.Index('idx_system_logs_logger_timestamp', 'logger_name', 'created_at'),
    )
    
    def __repr__(self):
        return f'<SystemLog {self.level} {self.logger_name}: {self.message[:50]}...>'


class ModelRegistry(BaseModel):
    """ML model registry"""
    __tablename__ = 'model_registry'
    
    model_name = db.Column(db.String(100), unique=True, nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # 'sklearn', 'pytorch', 'tensorflow', etc.
    model_version = db.Column(db.String(20), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    file_hash = db.Column(db.String(64))  # SHA-256 hash
    training_data_hash = db.Column(db.String(64))  # Hash of training data
    performance_metrics = db.Column(JSON)  # Model performance data
    hyperparameters = db.Column(JSON)  # Model configuration
    feature_names = db.Column(JSON)  # List of feature names
    is_deployed = db.Column(db.Boolean, default=False)
    deployed_at = db.Column(db.DateTime)
    notes = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ModelRegistry {self.model_name} v{self.model_version}>'


class BacktestResult(BaseModel):
    """Backtest results model"""
    __tablename__ = 'backtest_results'
    
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'), nullable=False)
    test_name = db.Column(db.String(100), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    initial_capital = db.Column(db.Numeric(15, 2), nullable=False)
    final_capital = db.Column(db.Numeric(15, 2), nullable=False)
    total_return = db.Column(db.Numeric(10, 4))  # Percentage return
    total_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    losing_trades = db.Column(db.Integer, default=0)
    win_rate = db.Column(db.Numeric(5, 4), default=0.0)  # Percentage
    max_drawdown = db.Column(db.Numeric(10, 4), default=0.0)
    sharpe_ratio = db.Column(db.Numeric(10, 6))
    sortino_ratio = db.Column(db.Numeric(10, 6))
    profit_factor = db.Column(db.Numeric(10, 4))
    avg_trade_duration = db.Column(db.Integer)  # Minutes
    detailed_results = db.Column(JSON)  # Detailed backtest data
    
    def __repr__(self):
        return f'<BacktestResult {self.test_name} {self.symbol} ({self.total_return:.2f}%)>'


class AlertRule(BaseModel):
    """Alert rules model"""
    __tablename__ = 'alert_rules'
    
    name = db.Column(db.String(100), nullable=False)
    rule_type = db.Column(db.String(50), nullable=False)  # 'price', 'pnl', 'risk', 'system'
    condition = db.Column(db.String(200), nullable=False)  # Alert condition
    threshold_value = db.Column(db.Numeric(15, 6))
    comparison_operator = db.Column(db.String(10))  # '>', '<', '>=', '<=', '==', '!='
    target_accounts = db.Column(JSON)  # List of account IDs to monitor
    target_symbols = db.Column(JSON)  # List of symbols to monitor
    notification_channels = db.Column(JSON)  # List of notification methods
    is_active = db.Column(db.Boolean, default=True)
    last_triggered = db.Column(db.DateTime)
    trigger_count = db.Column(db.Integer, default=0)
    cooldown_minutes = db.Column(db.Integer, default=60)  # Cooldown between alerts
    
    def __repr__(self):
        return f'<AlertRule {self.name} ({self.rule_type})>'


class Notification(BaseModel):
    """Notification history model"""
    __tablename__ = 'notifications'
    
    alert_rule_id = db.Column(db.Integer, db.ForeignKey('alert_rules.id'))
    notification_type = db.Column(db.String(50), nullable=False)  # 'email', 'sms', 'webhook', etc.
    recipient = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(200))
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # 'pending', 'sent', 'failed'
    sent_at = db.Column(db.DateTime)
    delivery_attempts = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    metadata = db.Column(JSON)
    
    def __repr__(self):
        return f'<Notification {self.notification_type} to {self.recipient} ({self.status})>'