"""
Logging configuration for the Star Wars RAG application.

This module provides a centralized logging system with structured logging
and proper formatting for debugging and monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_path: Optional[Path] = None,
    enable_structured_logging: bool = True
) -> None:
    """Setup application logging with proper configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format string for log messages
        file_path: Optional file path for logging to file
        enable_structured_logging: Whether to enable structured logging
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    if enable_structured_logging:
        setup_structured_logging(numeric_level, file_path)
    else:
        setup_standard_logging(numeric_level, format_string, file_path)


def setup_standard_logging(
    level: int,
    format_string: str,
    file_path: Optional[Path] = None
) -> None:
    """Setup standard Python logging.
    
    Args:
        level: Logging level
        format_string: Format string for log messages
        file_path: Optional file path for logging to file
    """
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if file_path:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def setup_structured_logging(
    level: int,
    file_path: Optional[Path] = None
) -> None:
    """Setup structured logging using structlog.
    
    Args:
        level: Logging level
        file_path: Optional file path for logging to file
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging handlers
    setup_standard_logging(level, "", file_path)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def log_service_event(
    logger: structlog.BoundLogger,
    service_name: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
    level: str = "info"
) -> None:
    """Log a service event with structured data.
    
    Args:
        logger: Logger instance
        service_name: Name of the service
        event: Event name/type
        details: Optional event details
        level: Log level
    """
    log_data = {
        "service": service_name,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if details:
        log_data.update(details)
    
    getattr(logger, level)(event, **log_data)


def log_performance_metric(
    logger: structlog.BoundLogger,
    metric_name: str,
    value: float,
    unit: str,
    service: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log a performance metric.
    
    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        service: Optional service name
        details: Optional additional details
    """
    log_data = {
        "metric": metric_name,
        "value": value,
        "unit": unit,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if service:
        log_data["service"] = service
    
    if details:
        log_data.update(details)
    
    logger.info("performance_metric", **log_data)


def log_error_with_context(
    logger: structlog.BoundLogger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    service: Optional[str] = None
) -> None:
    """Log an error with context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Optional context information
        service: Optional service name
    """
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if service:
        log_data["service"] = service
    
    if context:
        log_data["context"] = context
    
    logger.error("error_occurred", **log_data, exc_info=True)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get a logger instance for this class.
        
        Returns:
            Structured logger instance
        """
        return get_logger(self.__class__.__name__)
    
    def log_method_call(self, method_name: str, **kwargs) -> None:
        """Log a method call with parameters.
        
        Args:
            method_name: Name of the method being called
            **kwargs: Method parameters
        """
        self.logger.debug("method_call", method=method_name, params=kwargs)
    
    def log_method_result(self, method_name: str, result: Any) -> None:
        """Log a method result.
        
        Args:
            method_name: Name of the method
            result: Method result
        """
        self.logger.debug("method_result", method=method_name, result=str(result)[:200])


# Convenience functions for common logging patterns
def log_api_request(logger: structlog.BoundLogger, endpoint: str, method: str, **kwargs) -> None:
    """Log an API request.
    
    Args:
        logger: Logger instance
        endpoint: API endpoint
        method: HTTP method
        **kwargs: Additional request details
    """
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "timestamp": datetime.utcnow().isoformat()
    }
    log_data.update(kwargs)
    logger.info("api_request", **log_data)


def log_api_response(logger: structlog.BoundLogger, endpoint: str, status_code: int, duration: float, **kwargs) -> None:
    """Log an API response.
    
    Args:
        logger: Logger instance
        endpoint: API endpoint
        status_code: HTTP status code
        duration: Request duration in seconds
        **kwargs: Additional response details
    """
    log_data = {
        "endpoint": endpoint,
        "status_code": status_code,
        "duration": duration,
        "timestamp": datetime.utcnow().isoformat()
    }
    log_data.update(kwargs)
    logger.info("api_response", **log_data)
