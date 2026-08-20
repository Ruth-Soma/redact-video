"""Local OCR abstraction and PaddleOCR implementation."""

from redact_video.ocr.apple_vision import AppleVisionOcrEngine
from redact_video.ocr.base import OcrEngine
from redact_video.ocr.paddle import PaddleOcrEngine

__all__ = ["AppleVisionOcrEngine", "OcrEngine", "PaddleOcrEngine"]
