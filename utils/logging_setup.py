import logging
import sys
from config_system import FILE_LOG_LEVEL

def setup_logging():
    """Centralized logging configuration for StockAgenticAI."""
    formatter = logging.Formatter('[%(asctime)s [%(levelname)s] %(name)s: %(message)s]')
    root_logger = logging.getLogger()
    root_logger.setLevel(FILE_LOG_LEVEL)
    
    # File handler
    file_handler = logging.FileHandler('trading.log', mode='a', encoding='utf-8')
    file_handler.setLevel(FILE_LOG_LEVEL)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel('INFO')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
