"""
CI/CD Scheduler for AQI Forecasting
Orchestrates:
- Real-time data fetching every 10 minutes
- Model retraining every 24 hours
"""
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Install APScheduler if not present
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    print("📦 Installing APScheduler...")
    os.system(f"{sys.executable} -m pip install apscheduler --break-system-packages -q")
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import job modules
FETCH_SCRIPT = PROJECT_ROOT / "src" / "ingestion" / "fetch_realtime.py"
RETRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "retrain.py"

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== JOB FUNCTIONS ====================
def fetch_data_job():
    """Job to fetch real-time data"""
    logger.info("="*60)
    logger.info("🌍 Starting data fetch job...")
    logger.info("="*60)
    
    try:
        # Execute fetch script
        import subprocess
        result = subprocess.run(
            [sys.executable, str(FETCH_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            logger.info("✅ Data fetch completed successfully")
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"   {line}")
        else:
            logger.error(f"❌ Data fetch failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"   Error: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Data fetch timed out (>120s)")
    except Exception as e:
        logger.error(f"❌ Data fetch job failed: {e}")


def retrain_model_job():
    """Job to retrain model"""
    logger.info("="*60)
    logger.info("🤖 Starting model retraining job...")
    logger.info("="*60)
    
    try:
        # Execute retrain script
        import subprocess
        result = subprocess.run(
            [sys.executable, str(RETRAIN_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max
        )
        
        if result.returncode == 0:
            logger.info("✅ Model retraining completed successfully")
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"   {line}")
        else:
            logger.error(f"❌ Model retraining failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"   Error: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Model retraining timed out (>10 min)")
    except Exception as e:
        logger.error(f"❌ Model retraining job failed: {e}")


# ==================== SCHEDULER SETUP ====================
def setup_scheduler():
    """Configure and start the scheduler"""
    scheduler = BlockingScheduler()
    
    # Job 1: Fetch data every 10 minutes
    scheduler.add_job(
        fetch_data_job,
        trigger=IntervalTrigger(minutes=10),
        id='fetch_data',
        name='Fetch Real-time AQI Data',
        max_instances=1,
        misfire_grace_time=300  # 5 minutes grace period
    )
    logger.info("📅 Scheduled: Data fetching every 10 minutes")
    
    # Job 2: Retrain model every 24 hours at 2 AM UTC
    scheduler.add_job(
        retrain_model_job,
        trigger=CronTrigger(hour=2, minute=0),  # 2 AM UTC daily
        id='retrain_model',
        name='Retrain AQI Model',
        max_instances=1,
        misfire_grace_time=3600  # 1 hour grace period
    )
    logger.info("📅 Scheduled: Model retraining daily at 2:00 AM UTC")
    
    return scheduler


# ==================== MAIN ====================
def main():
    """Main scheduler execution"""
    logger.info("="*60)
    logger.info("🚀 AQI FORECAST CI/CD SCHEDULER")
    logger.info("="*60)
    logger.info(f"   Start time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"   Log file: {LOG_FILE}")
    logger.info("")
    logger.info("   Schedule:")
    logger.info("   - Data fetch: Every 10 minutes")
    logger.info("   - Model retrain: Daily at 2:00 AM UTC")
    logger.info("="*60)
    
    # Run initial fetch immediately
    logger.info("\n🔥 Running initial data fetch...")
    fetch_data_job()
    
    # Setup and start scheduler
    logger.info("\n📡 Starting scheduler (Press Ctrl+C to stop)...")
    scheduler = setup_scheduler()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("\n⏹️  Scheduler stopped by user")
        scheduler.shutdown()
    except Exception as e:
        logger.error(f"\n❌ Scheduler crashed: {e}")
        scheduler.shutdown()


if __name__ == "__main__":
    main()