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
    
    try:
        if db is None:
            db = SQLAlchemy(app)
        
        # Initialize models with database instance
        from . import models
        models.initialize_models(db)
        
        # Create tables
        with app.app_context():
            try:
                db.create_all()
                logger.info("Database tables created successfully")
            except Exception as e:
                logger.error(f"Error creating database tables: {e}")
        
        return db
        
    except ImportError as e:
        logger.warning(f"Models not available: {e}")
        # Create minimal database without models
        if db is None:
            db = SQLAlchemy(app)
        return db
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Return basic database instance
        if db is None:
            db = SQLAlchemy(app)
        return db


def get_db() -> SQLAlchemy:
    """Get database instance
    
    Returns:
        SQLAlchemy: Database instance
    """
    global db
    return db


# Import models with graceful fallback
try:
    from .models import (
        BaseModel, Account, Position, Order, Trade, Strategy, Signal,
        MarketData, RiskMetric, SystemLog, ModelRegistry, BacktestResult,
        AlertRule, Notification
    )
    MODELS_AVAILABLE = True
except ImportError:
    # Create placeholder classes if models not available
    class BaseModel:
        def to_dict(self):
            return {}
        
        def update(self, **kwargs):
            pass
    
    class Account(BaseModel):
        pass
    
    class Position(BaseModel):
        pass
    
    class Order(BaseModel):
        pass
    
    class Trade(BaseModel):
        pass
    
    class Strategy(BaseModel):
        pass
    
    class Signal(BaseModel):
        pass
    
    class MarketData(BaseModel):
        pass
    
    class RiskMetric(BaseModel):
        pass
    
    class SystemLog(BaseModel):
        pass
    
    class ModelRegistry(BaseModel):
        pass
    
    class BacktestResult(BaseModel):
        pass
    
    class AlertRule(BaseModel):
        pass
    
    class Notification(BaseModel):
        pass
    
    MODELS_AVAILABLE = False
    logger.warning("Database models not available - using placeholders")