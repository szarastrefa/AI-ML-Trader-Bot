from celery import Celery
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_celery():
    """Create and configure Celery app"""
    celery = Celery('ai_trader')
    
    # Configuration
    celery.conf.update(
        # Broker settings
        broker_url=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
        result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
        
        # Task settings
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # Beat scheduler settings
        beat_schedule_filename=os.getenv('CELERY_BEAT_SCHEDULE_FILENAME', '/tmp/celerybeat-schedule'),
        beat_scheduler='celery.beat:PersistentScheduler',
        
        # Worker settings
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=1000,
        
        # Retry settings
        task_default_retry_delay=60,
        task_max_retries=3,
        
        # Schedule tasks
        beat_schedule={
            'health-check': {
                'task': 'tasks.health_check',
                'schedule': 30.0,  # Every 30 seconds
            },
            'process-trading-data': {
                'task': 'tasks.process_trading_data',
                'schedule': 60.0,  # Every minute
            },
        },
    )
    
    logger.info("Celery app configured successfully")
    return celery

# Create celery app instance
celery_app = create_celery()

# Auto-discover tasks
try:
    celery_app.autodiscover_tasks(['tasks'])
except Exception as e:
    logger.warning(f"Task autodiscovery warning: {e}")

if __name__ == '__main__':
    celery_app.start()
