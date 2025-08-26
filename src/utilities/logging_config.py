import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger('rotating')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("logs/rotate.log", maxBytes=5*1024, backupCount=3)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

__all__ = ['logger']