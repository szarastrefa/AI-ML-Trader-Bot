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

# Import models at module level
from .models import (
    BaseModel, Account, Position, Order, Trade, Strategy, Signal,
    MarketData, RiskMetric, SystemLog, ModelRegistry, BacktestResult,
    AlertRule, Notification
)


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