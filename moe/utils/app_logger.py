import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class AppLogger:
    def __init__(self, name, log_file='app.log', log_level=logging.INFO, max_file_size=1024*1024, backup_count=5):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (with rotation)
        file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)

