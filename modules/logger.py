import logging
import os
import config.config as cfg

class ProctorLogger:
    """
    ProctorLogger provides standardized, multi-level file and console loggers 
    for proctoring operations, hardware telemetry, and surveillance security warnings.
    """
    def __init__(self, log_dir: str = cfg.LOGS_DIR):
        pass

    def info(self, msg: str) -> None:
        """
        Logs an informational message.
        """
        pass

    def warn(self, msg: str) -> None:
        """
        Logs a warning message.
        """
        pass

    def error(self, msg: str) -> None:
        """
        Logs an error message.
        """
        pass
