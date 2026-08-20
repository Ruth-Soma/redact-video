## Summary

Describe the problem and the focused change.

## Privacy and safety impact

Explain whether this touches OCR, detection, tracking, masks, verification,
reports, logging, errors, storage, or API boundaries.

## Validation

List the exact tests, benchmarks, and decoded-output comparisons performed.

## Checklist

- [ ] I did not commit real payment-card media, OCR values, credentials, private paths, or generated artifacts.
- [ ] Automatic success remains gated by rendered-output verification.
- [ ] New behavior has regression coverage.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run pytest` passes.
- [ ] User-facing documentation and `CHANGELOG.md` are updated when needed.
- [ ] Performance claims include a control and unchanged privacy evidence.
