# Video Redactor

[![CI](https://github.com/Ruth-Soma/redact-video/actions/workflows/ci.yml/badge.svg)](https://github.com/Ruth-Soma/redact-video/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Video Redactor is a local, headless Python engine that detects and removes
payment-card details from MP4 and MOV screen recordings. It combines OCR,
payment-field detection, temporal tracking, privacy masking, rendered-output
verification, and bounded automatic repair in one command.

The project is an Apache-2.0-licensed **public beta**. It is designed for local
CLI use and controlled self-hosting; it is not a hosted service or a compliance
certification product.

## Why this project exists

Screen recordings can reveal a card number, expiry date, or security code for
only a few frames while a user types, scrolls, or opens a saved-card form.
Single-frame OCR and static blur boxes are not enough. Video Redactor therefore:

1. scans locally for payment-card fields and value patterns;
2. tracks sensitive regions between OCR scans;
3. retains sticky masks through temporary detection loss;
4. renders a completed timeline against a fresh decode;
5. rescans the rendered video and repairs any remaining readable findings;
6. reports `COMPLETED` only after verification passes.

No cloud OCR service is used. Frames and recognized text remain local, and OCR
tokens are excluded from persisted reports and model representations.

## Current capabilities

- Full and partially typed PAN/card-number detection with local Luhn validation
- Expiry, CVV/CVC/CSC/CID, masked last-four, and saved-card detection
- Payment-context and semantic-field analysis to limit unrelated-number matches
- Lucas–Kanade optical flow, velocity prediction, template fallback, scene
  changes, same-frame reacquisition, and mandatory sticky masks
- Strong pixelation plus Gaussian-equivalent blur with resolution-aware padding
- H.264/yuv420p MP4 output with source-audio preservation and AAC fallback
- Every-frame rendered-output inspection with bounded opaque repairs
- Strict YAML configuration and sanitized, value-free JSON reports
- Local CLI, optional asynchronous HTTP API, and anonymous acceptance runner
- Generated privacy corpus, performance benchmark, CI, and non-root Docker image

## Requirements

- Python **3.12 or 3.13** for the full OCR pipeline
- FFmpeg and ffprobe available on `PATH`
- macOS or Linux

Python 3.14 can run the media-only manual mode, but current PaddlePaddle wheels
do not support the complete OCR runtime on Python 3.14.

Install FFmpeg with your platform package manager, for example:

```bash
# macOS
brew install ffmpeg

# Debian/Ubuntu
sudo apt-get install ffmpeg
```

## Install from source

After cloning this repository, the recommended setup uses
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.13 --extra ocr
uv run redact-video --version
```

An ordinary virtual environment also works:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install ".[ocr]"
redact-video --version
```

The first OCR run may download official model weights. Air-gapped deployments
can configure local model directories; see
[configuration](docs/CONFIGURATION.md).

## Quick start

Process a recording automatically and write a sanitized report:

```bash
uv run redact-video input.mp4 \
  --output input.redacted.mp4 \
  --report input.report.json \
  --verbose
```

The command exits with status `0` only when automatic verification finds no
remaining readable card details. Always inspect the status and review important
outputs before sharing them.

Use the higher privacy-oriented scan profile when additional runtime is
acceptable:

```bash
uv run redact-video input.mov \
  --output safe.mp4 \
  --safe \
  --report safe.report.json
```

Apply a strict YAML configuration:

```bash
uv run redact-video input.mp4 \
  --config config.example.yaml \
  --output input.redacted.mp4 \
  --report input.report.json
```

`--fast` lowers OCR scan rates for development and is not intended as a privacy
profile. It cannot be combined with `--safe`.

## How verification behaves

Automatic output can finish in one of two user-relevant states:

- `COMPLETED`: the final rendered-output pass found no remaining value-bearing
  card detections.
- `NEEDS_REVIEW`: an output exists, but verification could not establish a
  clean result within the configured pass limit.

`NEEDS_REVIEW` is intentionally not treated as success. Reports contain counts,
categories, timings, and timestamp-only issue windows—not OCR text, values,
frames, masks, or private media paths.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Automatic processing and verification completed |
| 1 | Processing failed |
| 2 | Input is corrupt or unsupported |
| 3 | Output was rendered but needs review |
| 4 | Arguments or configuration are invalid |

Manual-region mode is available for media-pipeline development:

```bash
uv run redact-video input.mp4 --manual-box 50,450,500,50
```

Manual coordinates use display-oriented `x,y,width,height` pixels. Manual mode
always exits with code `3` because it does not enter automatic verification.

## Platform behavior

PP-OCRv6 tiny through ONNX Runtime is the authoritative local OCR engine. On
supported macOS hosts, Apple Vision accelerates routine scans while ONNX still
performs bounded audits and takes over if the native helper fails. Linux uses
the ONNX path directly.

Constant-frame-rate video uses an FFmpeg raw-frame pipe. Variable-frame-rate
inputs use the timestamp-aware OpenCV path and are normalized to deterministic
constant-frame-rate output.

## Performance reference

On the August 2026 Apple-arm64 development machine, the accepted default
pipeline processed a controlled 30.259-second, 1260×2736, approximately 60 fps
recording in **187.272 seconds (3:07.272)**. It processed 1,815 frames, preserved
audio, performed three repair windows and two verification passes, and reported
zero remaining leaks. Every decoded output frame matched the previously
approved result.

This is a local observation, not a portable guarantee. Runtime depends on CPU,
resolution, frame rate, motion, OCR findings, and verification work. Privacy
verification takes priority over speed.

## Local HTTP API

Install the API and OCR extras, then start the loopback service:

```bash
uv sync --python 3.13 --extra api --extra ocr
uv run redact-video-api
```

The API provides job creation, status, output download, and sanitized reports
under `/v1/redactions`. It uses a bounded local process pool, upload limits,
private job directories, terminal raw-upload deletion, and output retention.

The API has no built-in authentication or TLS and binds to loopback by default.
Do not expose it directly to the internet. See [HTTP API](docs/API.md) and the
[security policy](SECURITY.md).

## Docker

Build the non-root local image:

```bash
docker build -t video-redactor:0.1.0 .
docker run --rm video-redactor:0.1.0 --help
```

The image installs the OCR and API extras and runs as the unprivileged
`redactor` user. Provide explicit, least-privilege input and output mounts when
processing media.

## Development

```bash
uv sync --python 3.13 --extra dev --extra api --extra ocr
uv run ruff check .
uv run pytest
```

Run the generated privacy/performance benchmark:

```bash
uv run python -m benchmarks.synthetic \
  --output synthetic-benchmark.json \
  --max-realtime-factor 20
```

Run an external controlled acceptance corpus:

```bash
uv run redact-video-acceptance \
  /private/non-production-inputs \
  /private/reviewed-outputs \
  --report /private/reviewed-outputs/acceptance.json
```

Never commit or attach real payment-card recordings, unredacted frames, OCR
transcripts, recognized values, or private reports. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Safety and limitations

- Automated verification reduces risk; it does not guarantee that every
  possible recording, UI, language, codec, or adversarial input is safe.
- Review outputs before publication when disclosure would have legal,
  contractual, financial, or privacy consequences.
- The default models and semantic vocabulary are optimized for English payment
  interfaces.
- Model weights may be downloaded on first use unless local paths are supplied.
- The bundled API is a local service boundary, not a hardened multi-tenant
  deployment.
- This project is not PCI DSS certification or legal/compliance advice.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Security design](docs/SECURITY.md)
- [Detection](docs/DETECTION.md)
- [Tracking](docs/TRACKING.md)
- [Rendering](docs/RENDERING.md)
- [Verification and repair](docs/VERIFICATION.md)
- [Configuration and reports](docs/CONFIGURATION.md)
- [HTTP API](docs/API.md)
- [Synthetic corpus](docs/SYNTHETIC_CORPUS.md)
- [Testing](docs/TESTING.md)
- [Release and acceptance](docs/RELEASE.md)
- [Roadmap](docs/ROADMAP.md)
- [V1 engineering specification](Video%20Card%20Redaction%20Engine%20%E2%80%94%20V1%20Build%20Specification.md)

## Contributing and support

Read [CONTRIBUTING.md](CONTRIBUTING.md), [SUPPORT.md](SUPPORT.md), the
[Code of Conduct](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before
participating. Security vulnerabilities must be reported privately.

## License

Copyright 2026 Video Redactor contributors.

Licensed under the [Apache License 2.0](LICENSE).
