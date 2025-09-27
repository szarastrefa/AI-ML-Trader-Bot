from celery_app import celery_app
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.health_check')
def health_check(self):
    """Health check task"""
    try:
        task_info = {
            'status': 'healthy',
            'task_id': self.request.id,
            'worker': 'active',
            'timestamp': datetime.utcnow().isoformat(),
            'hostname': self.request.hostname
        }
        logger.info(f"Health check completed: {task_info}")
        return task_info
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {
            'status': 'error',
            'error': str(exc),
            'task_id': self.request.id,
            'timestamp': datetime.utcnow().isoformat()
        }

@celery_app.task(bind=True, name='tasks.process_trading_data')
def process_trading_data(self):
    """Process trading data task"""
    try:
        # Simulate data processing
        logger.info("Processing trading data...")
        time.sleep(2)  # Simulate work
        
        result = {
            'status': 'processed',
            'task_id': self.request.id,
            'data_processed': True,
            'timestamp': datetime.utcnow().isoformat(),
            'records_processed': 100  # Mock data
        }
        logger.info(f"Trading data processed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Trading data processing failed: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)
