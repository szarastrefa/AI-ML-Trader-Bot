from typing import List, Dict, Any, Optional
import sys
import os
import logging
from pathlib import Path

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

def create_app():
    """Create Flask application"""
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
            'timestamp': '2024-01-01T00:00:00Z'
        })
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        status = {
            'status': 'healthy',
            'components': {
                'flask': 'ok',
                'config': 'ok' if CONFIG_AVAILABLE else 'warning',
                'celery': 'ok' if CELERY_AVAILABLE else 'warning'
            }
        }
        
        # Check Redis connection
        try:
            if CELERY_AVAILABLE and celery_app:
                inspect = celery_app.control.inspect()
                active_tasks = inspect.active()
                status['components']['celery_workers'] = 'ok' if active_tasks is not None else 'warning'
        except Exception as e:
            status['components']['celery_workers'] = f'error: {e}'
        
        return jsonify(status)
    
    @app.route('/api/summary')
    def api_summary():
        """API summary endpoint"""
        return jsonify({
            'total_accounts': 0,
            'active_strategies': 0,
            'total_pnl': 0.0,
            'status': 'Demo Mode'
        })
    
    @app.route('/api/trading/start', methods=['POST'])
    def start_trading():
        """Start trading endpoint"""
        return jsonify({
            'status': 'started',
            'message': 'Trading simulation started'
        })
    
    @app.route('/api/trading/stop', methods=['POST'])
    def stop_trading():
        """Stop trading endpoint"""
        return jsonify({
            'status': 'stopped', 
            'message': 'Trading simulation stopped'
        })
    
    @app.route('/api/tasks/health', methods=['POST'])
    def trigger_health_task():
        """Trigger health check task"""
        if not CELERY_AVAILABLE:
            return jsonify({'error': 'Celery not available'}), 503
            
        try:
            from tasks import health_check
            task = health_check.delay()
            return jsonify({
                'task_id': task.id,
                'status': 'submitted'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def main():
    """Main application entry point"""
    logger.info("Starting AI/ML Trader Bot...")
    
    app = create_app()
    
    # Run the application
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting Flask app on {host}:{port}")
    logger.info(f"Config available: {CONFIG_AVAILABLE}")
    logger.info(f"Celery available: {CELERY_AVAILABLE}")
    
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    main()
