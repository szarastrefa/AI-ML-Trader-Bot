#!/usr/bin/env python3
"""
Database Models
SQLAlchemy models for trading data storage
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON

logger = logging.getLogger(__name__)

# This will be set by database.py
db = None

def set_db_instance(database_instance):
    """Set the database instance from database.py"""
    global db
    db = database_instance


class BaseModel:
    """Base model with common fields"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        if not hasattr(self, '__table__'):
            return {}
            
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name, None)
            if isinstance(value, datetime):
                value = value.isoformat() if value else None
            result[column.name] = value
        return result
    
    def update(self, **kwargs):
        """Update model attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()


class Account(BaseModel):
    """Trading account model"""
    pass  # Will be defined when db is available


class Position(BaseModel):
    """Trading position model"""
    pass


class Order(BaseModel):
    """Trading order model"""
    pass


class Trade(BaseModel):
    """Executed trade model"""
    pass


class Strategy(BaseModel):
    """Trading strategy model"""
    pass


class Signal(BaseModel):
    """Trading signal model"""
    pass


class MarketData(BaseModel):
    """Market data storage model"""
    pass


class RiskMetric(BaseModel):
    """Risk metrics storage model"""
    pass


class SystemLog(BaseModel):
    """System log entries model"""
    pass


class ModelRegistry(BaseModel):
    """ML model registry"""
    pass


class BacktestResult(BaseModel):
    """Backtest results model"""
    pass


class AlertRule(BaseModel):
    """Alert rules model"""
    pass


class Notification(BaseModel):
    """Notification history model"""
    pass


# When database is available, define actual models
def initialize_models(database_instance):
    """Initialize models with database instance"""
    global db, Account, Position, Order, Trade, Strategy, Signal
    global MarketData, RiskMetric, SystemLog, ModelRegistry
    global BacktestResult, AlertRule, Notification
    
    db = database_instance
    
    class BaseModelDB(db.Model):
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
    
    class Account(BaseModelDB):
        """Trading account model"""
        __tablename__ = 'accounts'
        
        account_id = db.Column(db.String(100), unique=True, nullable=False)
        broker_name = db.Column(db.String(50), nullable=False)
        account_type = db.Column(db.String(20), nullable=False)
        currency = db.Column(db.String(10), nullable=False, default='USD')
        initial_balance = db.Column(db.Numeric(15, 2), default=0.0)
        current_balance = db.Column(db.Numeric(15, 2), default=0.0)
        equity = db.Column(db.Numeric(15, 2), default=0.0)
        margin = db.Column(db.Numeric(15, 2), default=0.0)
        free_margin = db.Column(db.Numeric(15, 2), default=0.0)
        is_active = db.Column(db.Boolean, default=True)
        last_update = db.Column(db.DateTime, default=datetime.utcnow)
        config = db.Column(JSON)
        
        def __repr__(self):
            return f'<Account {self.account_id} ({self.broker_name})>'
    
    class Position(BaseModelDB):
        """Trading position model"""
        __tablename__ = 'positions'
        
        account_id = db.Column(db.String(100), db.ForeignKey('accounts.account_id'), nullable=False)
        symbol = db.Column(db.String(20), nullable=False)
        side = db.Column(db.String(10), nullable=False)
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
        metadata = db.Column(JSON)
        
        def __repr__(self):
            return f'<Position {self.symbol} {self.side} {self.size}>'
    
    class Order(BaseModelDB):
        """Trading order model"""
        __tablename__ = 'orders'
        
        account_id = db.Column(db.String(100), db.ForeignKey('accounts.account_id'), nullable=False)
        symbol = db.Column(db.String(20), nullable=False)
        side = db.Column(db.String(10), nullable=False)
        order_type = db.Column(db.String(20), nullable=False)
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
    
    class Trade(BaseModelDB):
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
        signal_id = db.Column(db.String(100))
        broker_trade_id = db.Column(db.String(100))
        executed_at = db.Column(db.DateTime, default=datetime.utcnow)
        metadata = db.Column(JSON)
        
        def __repr__(self):
            return f'<Trade {self.symbol} {self.side} {self.amount} @ {self.price}>'
    
    class Strategy(BaseModelDB):
        """Trading strategy model"""
        __tablename__ = 'strategies'
        
        name = db.Column(db.String(100), unique=True, nullable=False)
        strategy_type = db.Column(db.String(50), nullable=False)
        description = db.Column(db.Text)
        config = db.Column(JSON)
        is_active = db.Column(db.Boolean, default=False)
        assigned_accounts = db.Column(JSON)
        performance_metrics = db.Column(JSON)
        model_path = db.Column(db.String(255))
        model_metadata = db.Column(JSON)
        last_signal_at = db.Column(db.DateTime)
        
        def __repr__(self):
            return f'<Strategy {self.name} ({self.strategy_type})>'
    
    class Signal(BaseModelDB):
        """Trading signal model"""
        __tablename__ = 'signals'
        
        strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'), nullable=False)
        symbol = db.Column(db.String(20), nullable=False)
        action = db.Column(db.String(10), nullable=False)
        confidence = db.Column(db.Numeric(5, 4), nullable=False)
        price = db.Column(db.Numeric(15, 8))
        stop_loss = db.Column(db.Numeric(15, 8))
        take_profit = db.Column(db.Numeric(15, 8))
        signal_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
        is_executed = db.Column(db.Boolean, default=False)
        executed_at = db.Column(db.DateTime)
        execution_price = db.Column(db.Numeric(15, 8))
        pnl = db.Column(db.Numeric(15, 2))
        metadata = db.Column(JSON)
        
        def __repr__(self):
            return f'<Signal {self.symbol} {self.action} (confidence: {self.confidence})>'
    
    # Create simple placeholder models for other classes
    class MarketData(BaseModelDB):
        __tablename__ = 'market_data'
        symbol = db.Column(db.String(20), nullable=False)
        timestamp = db.Column(db.DateTime, nullable=False)
        price = db.Column(db.Numeric(15, 8), nullable=False)
        volume = db.Column(db.Numeric(20, 8), default=0.0)
    
    class RiskMetric(BaseModelDB):
        __tablename__ = 'risk_metrics'
        account_id = db.Column(db.String(100))
        metric_type = db.Column(db.String(50), nullable=False)
        metric_value = db.Column(db.Numeric(15, 6), nullable=False)
    
    class SystemLog(BaseModelDB):
        __tablename__ = 'system_logs'
        level = db.Column(db.String(20), nullable=False)
        logger_name = db.Column(db.String(100), nullable=False)
        message = db.Column(db.Text, nullable=False)
    
    class ModelRegistry(BaseModelDB):
        __tablename__ = 'model_registry'
        model_name = db.Column(db.String(100), unique=True, nullable=False)
        model_type = db.Column(db.String(50), nullable=False)
        file_path = db.Column(db.String(500), nullable=False)
    
    class BacktestResult(BaseModelDB):
        __tablename__ = 'backtest_results'
        strategy_id = db.Column(db.Integer, nullable=False)
        test_name = db.Column(db.String(100), nullable=False)
        total_return = db.Column(db.Numeric(10, 4))
    
    class AlertRule(BaseModelDB):
        __tablename__ = 'alert_rules'
        name = db.Column(db.String(100), nullable=False)
        rule_type = db.Column(db.String(50), nullable=False)
        is_active = db.Column(db.Boolean, default=True)
    
    class Notification(BaseModelDB):
        __tablename__ = 'notifications'
        notification_type = db.Column(db.String(50), nullable=False)
        recipient = db.Column(db.String(200), nullable=False)
        message = db.Column(db.Text, nullable=False)
        status = db.Column(db.String(20), default='pending')
    
    # Update module globals
    globals()['Account'] = Account
    globals()['Position'] = Position
    globals()['Order'] = Order
    globals()['Trade'] = Trade
    globals()['Strategy'] = Strategy
    globals()['Signal'] = Signal
    globals()['MarketData'] = MarketData
    globals()['RiskMetric'] = RiskMetric
    globals()['SystemLog'] = SystemLog
    globals()['ModelRegistry'] = ModelRegistry
    globals()['BacktestResult'] = BacktestResult
    globals()['AlertRule'] = AlertRule
    globals()['Notification'] = Notification
    
    return {
        'Account': Account,
        'Position': Position,
        'Order': Order,
        'Trade': Trade,
        'Strategy': Strategy,
        'Signal': Signal,
        'MarketData': MarketData,
        'RiskMetric': RiskMetric,
        'SystemLog': SystemLog,
        'ModelRegistry': ModelRegistry,
        'BacktestResult': BacktestResult,
        'AlertRule': AlertRule,
        'Notification': Notification
    }