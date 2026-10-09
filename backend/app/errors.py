"""Exception hierarchy shared across the package."""


class AppError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(AppError):
    """Required settings are missing or invalid."""


class DataValidationError(AppError):
    """Incoming data failed a quality gate."""


class ExternalServiceError(AppError):
    """An upstream API failed after all retries."""
