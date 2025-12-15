"""
Logging Module for Steel Defect Detection System
"""
import logging
import os
from datetime import datetime

# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("SteelDefectDetection")

# Also add console handler for visibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
