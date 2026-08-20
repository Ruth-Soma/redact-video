# Contributing to Video Redactor

Thank you for helping improve Video Redactor. Contributions are welcome when
they preserve the project's privacy-first and local-processing guarantees.

## Before opening an issue

- Search existing issues first.
- Do not upload real payment-card recordings, OCR output, card numbers, CVVs,
  account details, private paths, or unredacted screenshots.
- Reproduce media bugs with the generated synthetic corpus whenever possible.
- For vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a
  public issue.

## Development setup

Video Redactor requires Python 3.12 or 3.13 for the complete OCR runtime, plus
FFmpeg and ffprobe on `PATH`.

```bash
uv sync --python 3.13 --extra dev --extra api --extra ocr
uv run pytest
uv run ruff check .
```

The deterministic suite does not need downloaded OCR models. To exercise the
real local model after installing the OCR extra:

```bash
RUN_OCR_RUNTIME_TEST=1 \
  uv run --isolated --python 3.13 --extra dev --extra ocr \
  pytest tests/smoke/test_ocr_runtime.py
```

## Privacy invariants

Changes must continue to satisfy these rules:

- Frames and OCR text remain local and are not persisted by OCR adapters.
- Exceptions, logs, reports, model representations, and API responses never
  contain recognized payment values or private media paths.
- Automatic jobs return `COMPLETED` only after rendered-output verification
  finds no readable value-bearing card details.
- A failed or unavailable accelerator falls back to the authoritative local OCR
  path; it must never silently skip a required scan.
- Sensitive tracks retain sticky masks through bounded detection loss.
- Privacy regressions cannot be accepted in exchange for speed.

Read [docs/SECURITY.md](docs/SECURITY.md) and
[docs/VERIFICATION.md](docs/VERIFICATION.md) before changing OCR, detection,
tracking, rendering, verification, reports, or error handling.

## Testing changes

Add the smallest test that would have caught the problem. Depending on the
change, run:

```bash
uv run pytest tests/unit
uv run pytest tests/integration/test_synthetic_corpus.py
uv run python -m benchmarks.synthetic --output synthetic-benchmark.json
```

Performance changes require a disabled control, stage-level timings, identical
privacy counts, and decoded-frame comparison on controlled non-production
media. Document rejected candidates as well as accepted ones.

Before submitting a pull request, run the complete local gate:

```bash
uv lock --check
uv run ruff check .
uv run pytest --cov=redact_video --cov-report=term-missing --cov-fail-under=85
uv build
```

## Pull requests

- Keep each pull request focused.
- Explain the behavior, privacy impact, tests, and performance evidence.
- Update user-facing documentation and `CHANGELOG.md` when appropriate.
- Do not commit generated media, private reports, model weights, virtual
  environments, or credentials.
- Preserve unrelated work already present in the branch.

By contributing, you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE).
