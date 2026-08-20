# Changelog

All notable changes to Video Redactor are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-20

Initial public beta.

### Added

- Local payment-card detection using PP-OCRv6 tiny ONNX, with optional Apple
  Vision acceleration on macOS.
- PAN, expiry, CVV/CVC/CSC/CID, and saved-card field detection.
- Optical-flow tracking, sticky masks, scene-change reacquisition, and
  deterministic mask timelines.
- H.264 MP4 rendering with audio preservation and automatic output validation.
- Mandatory rendered-output verification, bounded repairs, and fail-closed
  `NEEDS_REVIEW` results.
- CLI, local asynchronous HTTP API, YAML configuration, sanitized reports, and
  an anonymous acceptance runner.
- Generated privacy corpus, release benchmark, Python 3.12/3.13 CI, Docker
  checks, and an opt-in real OCR smoke workflow.

### Performance

- The accepted 30.259-second, 1,815-frame reference recording completes in
  187.272 seconds on the development machine while preserving audio and
  reporting zero remaining leaks.
