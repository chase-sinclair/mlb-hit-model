import subprocess
import sys
import time
from pathlib import Path

import schedule
from loguru import logger

ROOT = Path(__file__).parent
LOG_DIR = ROOT / 'outputs' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(LOG_DIR / 'scheduler.log', rotation='7 days', retention='30 days')


def job():
    logger.info('Scheduler: starting daily run...')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'main.py')],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        logger.error(f'main.py exited with code {result.returncode}')
    else:
        logger.info('Scheduler: daily run complete')


schedule.every().day.at('11:30').do(job)
logger.info('Scheduler armed — running main.py daily at 11:30 AM ET')
logger.info('Press Ctrl+C to stop')

while True:
    schedule.run_pending()
    time.sleep(30)
