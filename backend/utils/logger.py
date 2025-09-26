#!/usr/bin/env python3
"""
Logging Configuration
Comprehensive logging system with multiple handlers, formatters, and log levels
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional
from datetime import datetime
import json


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Add color to logger name
        record.name = f"\033[94m{record.name}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(name: str, level: str = 'INFO', log_file: Optional[str] = None, 
                enable_json: bool = False, enable_console: bool = True) -> logging.Logger:
    """Setup logger with multiple handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_json: Enable JSON formatting for file output
        enable_console: Enable console output
        
    Returns:
        logging.Logger: Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Use colored formatter for console
        if sys.stdout.isatty():  # If running in terminal
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Rotating file handler (10MB max, 5 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Use JSON formatter for file if enabled
        if enable_json:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    if log_file:
        error_file = log_file.replace('.log', '_errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        if enable_json:
            error_formatter = JsonFormatter()
        else:
            error_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s\n'
                    'Exception: %(pathname)s:%(lineno)d in %(funcName)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        error_handler.setFormatter(error_formatter)
        logger.addHandler(error_handler)
    
    return logger


def setup_trading_logger(config: dict = None) -> logging.Logger:
    """Setup specific logger for trading operations
    
    Args:
        config: Configuration dictionary
        
    Returns:
        logging.Logger: Trading logger
    """
    config = config or {}
    
    log_level = config.get('LOG_LEVEL', 'INFO')
    log_file = config.get('LOG_FILE', 'logs/trading.log')
    enable_json = config.get('LOG_JSON', False)
    
    return setup_logger(
        'trading',
        level=log_level,
        log_file=log_file,
        enable_json=enable_json
    )


def setup_strategy_logger(strategy_name: str, config: dict = None) -> logging.Logger:
    """Setup logger for specific strategy
    
    Args:
        strategy_name: Name of the strategy
        config: Configuration dictionary
        
    Returns:
        logging.Logger: Strategy logger
    """
    config = config or {}
    
    log_level = config.get('LOG_LEVEL', 'INFO')
    log_file = config.get('STRATEGY_LOG_FILE', f'logs/strategies/{strategy_name}.log')
    enable_json = config.get('LOG_JSON', False)
    
    return setup_logger(
        f'strategy.{strategy_name}',
        level=log_level,
        log_file=log_file,
        enable_json=enable_json
    )


def setup_broker_logger(broker_name: str, config: dict = None) -> logging.Logger:
    """Setup logger for specific broker
    
    Args:
        broker_name: Name of the broker
        config: Configuration dictionary
        
    Returns:
        logging.Logger: Broker logger
    """
    config = config or {}
    
    log_level = config.get('LOG_LEVEL', 'INFO')
    log_file = config.get('BROKER_LOG_FILE', f'logs/brokers/{broker_name}.log')
    enable_json = config.get('LOG_JSON', False)
    
    return setup_logger(
        f'broker.{broker_name}',
        level=log_level,
        log_file=log_file,
        enable_json=enable_json
    )


class TradingLogFilter(logging.Filter):
    """Custom filter for trading-specific log records"""
    
    def filter(self, record):
        # Add trading context if available
        if hasattr(record, 'symbol'):
            record.msg = f"[{record.symbol}] {record.msg}"
        
        if hasattr(record, 'account_id'):
            record.msg = f"[{record.account_id}] {record.msg}"
        
        return True


def log_trade_execution(logger: logging.Logger, symbol: str, action: str, 
                       amount: float, price: float, account_id: str = None):
    """Log trade execution with structured data
    
    Args:
        logger: Logger instance
        symbol: Trading symbol
        action: Trade action (buy/sell)
        amount: Trade amount
        price: Execution price
        account_id: Optional account ID
    """
    extra_data = {
        'symbol': symbol,
        'action': action,
        'amount': amount,
        'price': price,
        'trade_timestamp': datetime.now().isoformat()
    }
    
    if account_id:
        extra_data['account_id'] = account_id
    
    logger.info(
        f"Trade executed: {action.upper()} {amount} {symbol} @ {price}",
        extra=extra_data
    )


def log_strategy_signal(logger: logging.Logger, strategy_name: str, symbol: str,
                       signal_type: str, confidence: float, metadata: dict = None):
    """Log strategy signal generation
    
    Args:
        logger: Logger instance
        strategy_name: Name of the strategy
        symbol: Trading symbol
        signal_type: Type of signal (buy/sell/hold)
        confidence: Signal confidence (0-1)
        metadata: Additional signal metadata
    """
    extra_data = {
        'strategy_name': strategy_name,
        'symbol': symbol,
        'signal_type': signal_type,
        'confidence': confidence,
        'signal_timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        extra_data['metadata'] = metadata
    
    logger.info(
        f"Signal generated: {strategy_name} -> {signal_type.upper()} {symbol} (confidence: {confidence:.2f})",
        extra=extra_data
    )


def log_risk_event(logger: logging.Logger, event_type: str, description: str,
                  severity: str = 'INFO', account_id: str = None, metadata: dict = None):
    """Log risk management events
    
    Args:
        logger: Logger instance
        event_type: Type of risk event
        description: Event description
        severity: Event severity
        account_id: Optional account ID
        metadata: Additional metadata
    """
    extra_data = {
        'event_type': event_type,
        'severity': severity,
        'risk_timestamp': datetime.now().isoformat()
    }
    
    if account_id:
        extra_data['account_id'] = account_id
        
    if metadata:
        extra_data['metadata'] = metadata
    
    log_method = getattr(logger, severity.lower(), logger.info)
    log_method(f"Risk event: {event_type} - {description}", extra=extra_data)


# Configure root logger to avoid duplicate logs
logging.getLogger().handlers.clear()
logging.getLogger().addHandler(logging.NullHandler())