"""Sanitized application errors and their CLI classifications."""


class RedactVideoError(Exception):
    """Base class for expected processing failures."""


class UnsupportedInputError(RedactVideoError):
    """The input cannot be safely processed."""


class MediaProbeError(RedactVideoError):
    """Media metadata could not be inspected."""


class DecodeError(RedactVideoError):
    """Video frames could not be decoded."""


class EncodeError(RedactVideoError):
    """The processed media could not be encoded or muxed."""


class ConfigurationError(RedactVideoError):
    """Configuration or command arguments are invalid."""


class OcrUnavailableError(RedactVideoError):
    """The configured local OCR runtime is not installed or cannot initialize."""


class OcrProcessingError(RedactVideoError):
    """Local OCR inference failed or returned an unsafe result shape."""


class ReportWriteError(RedactVideoError):
    """A sanitized report could not be persisted atomically."""
