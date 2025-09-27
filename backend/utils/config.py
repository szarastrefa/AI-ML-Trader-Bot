#!/usr/bin/env python3
"""
Configuration Management
Handles application configuration from environment variables, YAML files, and defaults
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the trading bot"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_data = {}
        self.config_file = config_file
        
        # Load default configuration
        self._load_defaults()
        
        # Load from file if specified
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
        
    def _load_defaults(self):
        """Load default configuration values"""
        self.config_data = {
            # Server configuration
            'HOST': '0.0.0.0',
            'PORT': 5000,
            'DEBUG': False,
            'SECRET_KEY': 'dev-secret-key-change-in-production',
            
            # Database configuration
            'DATABASE_URL': 'sqlite:///trader.db',
            'REDIS_URL': 'redis://redis:6379/0',
            
            # Logging configuration
            'LOG_LEVEL': 'INFO',
            'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'LOG_FILE': 'logs/trader.log',
            'LOG_MAX_BYTES': 10485760,  # 10MB
            'LOG_BACKUP_COUNT': 5,
            
            # Trading configuration
            'DEFAULT_TIMEFRAME': '1h',
            'MAX_POSITIONS': 10,
            'DEFAULT_RISK_PERCENT': 0.02,  # 2% per trade
            'MAX_PORTFOLIO_RISK': 0.10,    # 10% total portfolio risk
            
            # Strategy configuration
            'STRATEGY_UPDATE_INTERVAL': 300,  # 5 minutes
            'MARKET_DATA_UPDATE_INTERVAL': 60,  # 1 minute
            'PORTFOLIO_UPDATE_INTERVAL': 30,   # 30 seconds
            
            # Risk management
            'ENABLE_RISK_MANAGEMENT': True,
            'MAX_DRAWDOWN': 0.15,  # 15%
            'DAILY_LOSS_LIMIT': 0.05,  # 5% daily
            'ENABLE_CIRCUIT_BREAKER': True,
            
            # API configuration
            'API_RATE_LIMIT': '1000 per hour',
            'API_TIMEOUT': 30,
            'ENABLE_API_DOCS': True,
            
            # WebSocket configuration
            'WS_PING_INTERVAL': 25,
            'WS_PING_TIMEOUT': 60,
            'WS_MAX_MESSAGE_SIZE': 1048576,  # 1MB
            
            # Celery configuration
            'CELERY_BROKER_URL': 'redis://redis:6379/0',
            'CELERY_RESULT_BACKEND': 'redis://redis:6379/0',
            'CELERY_TASK_SERIALIZER': 'json',
            'CELERY_ACCEPT_CONTENT': ['json'],
            
            # Security configuration
            'JWT_SECRET_KEY': 'jwt-secret-change-in-production',
            'JWT_ACCESS_TOKEN_EXPIRES': 3600,  # 1 hour
            'JWT_REFRESH_TOKEN_EXPIRES': 2592000,  # 30 days
            'BCRYPT_LOG_ROUNDS': 12,
            
            # Broker default settings
            'MT5_TIMEOUT': 60000,
            'CCXT_TIMEOUT': 30000,
            'IBKR_TIMEOUT': 30000,
            'DEFAULT_SLIPPAGE': 0.001,  # 0.1%
            'DEFAULT_COMMISSION': 0.0005,  # 0.05%
            
            # ML model configuration
            'MODEL_UPDATE_INTERVAL': 3600,  # 1 hour
            'MODEL_RETRAIN_INTERVAL': 86400,  # 24 hours
            'FEATURE_HISTORY_DAYS': 30,
            'MIN_TRAINING_SAMPLES': 1000,
            'MODEL_VALIDATION_SPLIT': 0.2,
            
            # Data storage
            'DATA_RETENTION_DAYS': 365,
            'BACKUP_INTERVAL_HOURS': 6,
            'COMPRESS_OLD_DATA': True,
            
            # Performance monitoring
            'ENABLE_PERFORMANCE_MONITORING': True,
            'METRICS_RETENTION_DAYS': 30,
            'ALERT_EMAIL_ENABLED': False,
            'ALERT_EMAIL_SMTP': '',
            'ALERT_EMAIL_FROM': '',
            'ALERT_EMAIL_TO': '',
        }
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self.config_data.update(file_config)
                    logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mapping = {
            # Server
            'HOST': 'HOST',
            'PORT': 'PORT',
            'DEBUG': 'DEBUG',
            'SECRET_KEY': 'SECRET_KEY',
            
            # Database
            'DATABASE_URL': 'DATABASE_URL',
            'REDIS_URL': 'REDIS_URL',
            
            # Logging
            'LOG_LEVEL': 'LOG_LEVEL',
            'LOG_FILE': 'LOG_FILE',
            
            # JWT
            'JWT_SECRET_KEY': 'JWT_SECRET_KEY',
            'JWT_ACCESS_TOKEN_EXPIRES': 'JWT_ACCESS_TOKEN_EXPIRES',
            
            # Trading
            'DEFAULT_RISK_PERCENT': 'DEFAULT_RISK_PERCENT',
            'MAX_PORTFOLIO_RISK': 'MAX_PORTFOLIO_RISK',
            'MAX_DRAWDOWN': 'MAX_DRAWDOWN',
            
            # Celery
            'CELERY_BROKER_URL': 'CELERY_BROKER_URL',
            'CELERY_RESULT_BACKEND': 'CELERY_RESULT_BACKEND',
        }
        
        for config_key, env_key in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Convert string values to appropriate types
                self.config_data[config_key] = self._convert_value(env_value)
    
    def _convert_value(self, value: str) -> Any:
        """Convert string environment variable to appropriate type"""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config_data[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values
        
        Returns:
            Dict containing all configuration
        """
        return self.config_data.copy()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary
        
        Args:
            config_dict: Dictionary of configuration values
        """
        self.config_data.update(config_dict)
    
    def save_to_file(self, output_file: str) -> bool:
        """Save current configuration to YAML file
        
        Args:
            output_file: Output file path
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {output_file}: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Required configuration checks
        required_keys = ['SECRET_KEY', 'DATABASE_URL', 'REDIS_URL']
        for key in required_keys:
            if not self.get(key) or self.get(key) == f'dev-{key.lower()}-change-in-production':
                errors.append(f"Configuration '{key}' must be set for production")
        
        # Validate numeric ranges
        numeric_validations = {
            'DEFAULT_RISK_PERCENT': (0.0, 0.1),  # 0-10%
            'MAX_PORTFOLIO_RISK': (0.0, 0.5),    # 0-50%
            'MAX_DRAWDOWN': (0.0, 1.0),          # 0-100%
            'PORT': (1, 65535),                  # Valid port range
        }
        
        for key, (min_val, max_val) in numeric_validations.items():
            value = self.get(key)
            if value is not None and not (min_val <= float(value) <= max_val):
                errors.append(f"Configuration '{key}' must be between {min_val} and {max_val}")
        
        # Validate file paths
        log_file = self.get('LOG_FILE')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create log directory {log_dir}: {e}")
        
        return errors
    
    def get_broker_config(self, broker_name: str) -> Dict[str, Any]:
        """Get broker-specific configuration
        
        Args:
            broker_name: Name of the broker
            
        Returns:
            Dict containing broker configuration
        """
        broker_config = self.config_data.get('brokers', {}).get(broker_name, {})
        
        # Add environment variable overrides
        env_prefix = f"{broker_name.upper()}_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                broker_config[config_key] = self._convert_value(value)
        
        return broker_config
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy-specific configuration
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dict containing strategy configuration
        """
        return self.config_data.get('strategies', {}).get(strategy_name, {})
    
    def is_production(self) -> bool:
        """Check if running in production mode
        
        Returns:
            bool: True if production mode
        """
        return os.getenv('FLASK_ENV', 'development') == 'production'
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled
        
        Returns:
            bool: True if debug mode enabled
        """
        return self.get('DEBUG', False)
    
    def __repr__(self) -> str:
        return f"Config(file={self.config_file}, keys={len(self.config_data)})"


# Global configuration instance
config = Config()


def load_config(config_file: str = None) -> Config:
    """Load configuration from file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Config: Configuration instance
    """
    global config
    config = Config(config_file)
    return config


def get_config() -> Config:
    """Get current configuration instance
    
    Returns:
        Config: Current configuration
    """
    return config