import os
import sys
from loguru import logger

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Logs
logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
