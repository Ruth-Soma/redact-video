# Testing

Run all current tests with:

```bash
uv run pytest
```

Unit tests cover:

- bounding-box expansion and clipping
- manual-region parsing and rejection
- privacy-blur behavior
- deterministic detection and mask timelines without sensitive values
- timeline-driven rendering for masked and unmasked frames
- FFprobe metadata parsing and invalid inputs
- Phase 1 pipeline request validation
- structural compatibility of mock OCR engines
- OCR token representation and serialization privacy
- context normalization without numeric mutation
- field-gated PAN, expiry, CVV, and saved-card normalization
- PaddleOCR 3.x result parsing for rectangles and polygons
- lazy Paddle initialization and local model configuration
- bounded lossless Apple Vision BMP transport with oversized-input and
  unavailable-encoder PNG fallbacks
- fail-closed behavior for malformed results
- sanitization of initialization and inference exceptions
- local Luhn validation, separators, invalid check digits, and length bounds
- fuzzy and split-token payment-context recognition
- exact shared OCR-span filtering, single-build detector orchestration, and
  fuzzy-match upper-bound equivalence through the complete detector corpus
- semantic anchors before a field contains any value
- partial PAN and partial security-code typing
- semantic PAN handling when OCR causes a Luhn failure
- stronger Luhn requirements for unlabeled PAN candidates
- combined and split-field expiry formats
- masked/last-four saved cards, associated expiry, and saved-card policy
- OTP, order, price, room, date, invoice, and postal-code false positives
- detector-output serialization without recognized card values
- Lucas–Kanade translation on synthetic screen regions, including a 55-pixel
  structured-field jump under the capped default feature budget
- concurrent multi-region LK execution with deterministic result ordering,
  low-resolution sequential fallback, and idempotent pool shutdown
- bounded template recovery after optical-flow failure
- fast-scroll failure retaining a sticky mask
- velocity prediction during temporary tracking loss
- the inclusive 350 ms sticky-retention boundary and eventual `LOST` state
- positive off-screen `ENDED` transitions
- local-motion versus full-screen scene-change classification
- same-frame OCR reacquisition after scene changes or confidence loss
- periodic OCR scheduling without scanning every frame
- scheduled source OCR/optical-flow overlap with sequential-result equivalence
- deterministic multi-type track IDs and value-free track serialization
- rejection of duplicate or reordered tracking results in a mask timeline
- lower verification thresholds across every sensitive category
- bounded decoder/ONNX audit overlap with synchronous-disable coverage
- consecutive-pass clean-evidence reuse, changed-frame rescans, and the rule
  that leaking evidence is never reusable
- one exact clean-evidence guard build per static first-pass frame, with no
  impossible prior-pass comparison or duplicate OCR-reference resize
- compact grayscale frame-count/geometry preservation, bounded select-expression
  grouping, and absolute selected-frame decoding with pixel equality
- stale authoritative-audit cancellation with synchronous fail-safe scanning
- label-only anchor exclusion from readable-leak evidence
- repair-window expansion, duration clamping, merging, and type aggregation
- encoded synthetic-frame marker survival and OCR-oracle filtering
- golden coverage, missed-pixel, over-mask, and visible-region metrics
- API mode-to-scan-rate configuration without changing verification policy
- bounded queue admission and fixed worker-error sanitization
- terminal raw-upload deletion and retention-based artifact cleanup
- value-free child-process reports and progress-state mapping
- strict YAML defaults, nested overrides, codec normalization, and executable
  environment preservation
- unknown, malformed, non-mapping, non-YAML, and oversized configuration files
- configuration errors that expose field locations but not submitted values
- report schema exclusion of paths, timelines, pixels, and OCR text
- sanitized per-stage timing serialization and verification/repair pass counts
- private atomic JSON report writes, collisions, overwrite, and cleanup failures
- anonymous external-corpus ordering, outputs, statuses, timing, and errors
- acceptance reports and command output excluding private filenames and paths

Integration tests generate disposable synthetic MP4/MOV inputs with FFmpeg and
cover:

- H.264/yuv420p output
- MP4 input with copied AAC audio
- MOV input with PCM audio transcoded to AAC
- silent input without artificial audio
- output resolution and duration preservation
- display rotation normalization without changing displayed geometry
- audio/video duration difference below 100 ms
- visible detail reduction inside a manual region
- two-pass automatic detection/tracking/timeline rendering with injected OCR
- automatic timeline exclusion from serialized job results
- automatic audio preservation and visible detail reduction in detected regions
- automatic and manual CLI operation with documented exit codes
- mutually exclusive automatic `--safe` and `--fast` modes
- successful first-pass verification and `COMPLETED` promotion
- intentional leak detection, temporal overlay repair, and second-pass success
- persistent leak exhaustion after three passes with `NEEDS_REVIEW`
- CLI exit code 3 when verification cannot establish a clean result
- static landscape and portrait payment forms
- character-by-character PAN and CVV entry
- slow and fast vertical scrolling
- card forms disappearing and returning
- sliding modal motion
- saved-card last-four and expiry masking with the brand preserved
- 500 ms payment-screen appearances
- encoded OTP, order, price, phone, tracking, invoice, postal, date, and room
  false positives remaining completely unmasked
- generated intentional leaks repaired on the second verification pass
- zero uncovered sensitive pixels and bounded scenario-specific over-mask ratios
- multipart API creation, polling, output download, and sanitized reports
- HTTP rejection of wrong, empty, and oversized uploads
- pending-output conflicts and queue-full retry responses
- execution through the production local process-pool boundary
- configured CLI processing with sanitized JSON report output
- process-stable OpenCV inner-thread limiting without nested oversubscription
- invalid YAML and unverified manual-report CLI rejection
- API worker YAML loading before safe/fast mode overrides
- end-to-end synthetic benchmark privacy, timing, and report serialization

No committed secret fixture or production payment data is required.
The Phase 7 corpus is generated into pytest temporary directories. Run it alone
with:

```bash
uv run pytest tests/integration/test_synthetic_corpus.py
```

The normal suite does not download OCR weights. Paddle behavior is tested using
the documented 3.x result shape and an injected in-memory pipeline. A separate
runtime smoke check can be run on Python 3.12 or 3.13 after installing the `ocr`
extra; it scans a generated `Card Number` frame through the real local OCR
runtime, Phase 3 detector, Phase 4 track boundary, Phase 5 mask/render boundary,
and a Phase 6 rendered-frame verification check. It is intentionally excluded
from deterministic CI because first use may download model weights.

```bash
RUN_OCR_RUNTIME_TEST=1 \
  uv run --isolated --python 3.13 --extra dev --extra ocr \
  pytest tests/smoke/test_ocr_runtime.py
```
