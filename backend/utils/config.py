from typing import List, Dict, Any, Optional
import os
import logging
import yaml
from pathlib import Path

class Config:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_data = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment and files"""
        try:
            # Environment variables
            self.config_data = {
                'database_url': os.getenv('DATABASE_URL', 'postgresql://trader:trader_password@localhost:5432/ai_trader'),
                'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                'flask_env': os.getenv('FLASK_ENV', 'development'),
                'secret_key': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
            }
            
            # Try to load from config file
            config_file = Path(__file__).parent.parent / 'config' / 'config.yaml'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    self.config_data.update(file_config)
        except Exception as e:
            self.logger.warning(f"Config loading error: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config_data.get(key, default)
    
    def validate_config(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        required_keys = ['database_url', 'redis_url']
        for key in required_keys:
            if not self.config_data.get(key):
                errors.append(f"Missing required config: {key}")
        
        return errors

# Global config instance
config = Config()
