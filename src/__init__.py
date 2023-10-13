"""defining logger"""
import os
import sys
import logging

MODEL_KEY_CD = "CD"
MODEL_KEY_DD = "DD"
MODEL_KEY_DS = "DS"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(LOG_DIR, "running_logs.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("logger")
