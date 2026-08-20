# Security and Privacy

The V1 privacy boundary is payment-card information. No recognized sensitive
value may appear in a model, log, report, exception, API response, or persisted
OCR transcript.

`OcrToken.text` is necessarily available transiently to detectors, but Pydantic
representations and serialization exclude it, and validation errors hide input
values. Paddle adapter exceptions replace third-party error details with
sanitized messages and suppress exception chaining because a vendor error could
otherwise echo recognized text.

Numeric OCR confusion correction requires an explicit expected card-field type.
It is never applied to general OCR text. Canonical numeric candidates are
temporary return values for validation and must be discarded immediately.

Phase 3 detector candidates also suppress recognized text from their
representations. The public `SensitiveDetection` schema forbids extra fields and
contains no value property; serialization is limited to category, frame/time,
geometry, confidence, and a constrained source identifier. Detector exceptions
and diagnostics must not interpolate token or normalized candidate text.

Phase 4 tracking consumes only sanitized detections. `SensitiveTrack`,
`TrackedRegion`, and `TrackingFrameResult` contain geometry and state but no OCR
text or recognized-value property. Optical flow keeps only the immediately
previous frame, while template patches exist transiently in memory and are never
serialized or written to disk. Tracker IDs are deterministic counters and do not
encode detected content.
Parallel flow workers share read-only references to the same current and
previous in-memory frames; they do not copy frames to another process or persist
pixels. Results contain only motion vectors and confidence, are joined before
track mutation, and the worker pool is shut down when the job exits.

Phase 5 `DetectionTimeline`, `MaskRegion`, and `MaskTimeline` models contain
only category, frame association, track ID, confidence where applicable, and
geometry. The automatic pipeline keeps its completed mask timeline in memory;
it is excluded from result serialization and is not persisted by default. The
second render pass consumes only source frames and this value-free timeline.

Phase 6 `VerificationLeak`, `RepairWindow`, and `VerificationSummary` models are
also value-free. Findings carry category, frame/time, geometry, confidence, and
a constrained detector source—not recognized OCR content. Public potential
issues contain only time ranges and sensitive categories. Repair masks use
deterministic IDs that do not encode content, and neither verification OCR nor
repair scans create transcript or sidecar files.

Consecutive repair passes may retain compact grayscale guards for frames that
had no finding. The cache is keyed only by frame index and timestamp, never
contains OCR tokens or recognized values, is replaced rather than accumulated
after each completed pass, and is never serialized or written to disk. A prior
leak, missing evidence, timestamp or shape mismatch, material pixel change, or
nonconsecutive pass number forces OCR instead of evidence reuse.

Selective post-repair decoding does not weaken those conditions. Every frame is
still decoded into the bounded grayscale guard stream. Only a strict match can
avoid the full-resolution raw pipe; prior leaks and all mismatches are selected
by absolute frame index and processed through normal OCR. Selected indices and
guards stay in job memory and are not added to reports. Compact-decoder failure
clears partial evidence and restores the full-frame path, while selected-decoder
failure aborts verification rather than treating a frame as clean.

Phase 7 fixtures are generated only from explicitly synthetic payment values and
ordinary fake names/addresses. Encoded videos are written under pytest temporary
directories and are not committed. The frame-index marker contains only an
integer and encodes no OCR text or sensitive value. Ground truth remains in test
memory and never changes the production model or result privacy boundaries.

Phase 8 job identifiers are random and never derived from upload filenames or
recognized content. Only the validated `.mp4` or `.mov` suffix is retained; the
original filename is not stored in the job record or returned by any endpoint.
Job roots and files use private permissions. Request bodies and file bytes are
bounded before the core pipeline is submitted, and active plus pending worker
capacity is finite.

HTTP workers return fixed failure codes instead of raw exception strings. API
reports exclude server paths, the in-memory mask timeline, upload names, frame
content, and OCR text. Raw uploads and transient progress are deleted at every
terminal outcome. Redacted outputs remain only for the configured retention
window, and service shutdown removes all API job directories. The server binds
to loopback and disables access logs by default; public deployments must add
authentication, TLS, and appropriate network controls outside this V1 adapter.

Phase 9 configuration uses a bounded `yaml.SafeLoader`, a 1 MiB input limit, a
mapping-only root, alias/node composition limits, and `extra="forbid"` at every
external and runtime model boundary.
Malformed YAML and invalid values are replaced with fixed errors that contain at
most schema field locations. Parser context and submitted values are never
interpolated into CLI or API failures.

Sanitized JSON reports are constructed by explicitly copying approved counts,
categories, verification summaries, and warnings. Their schema has no fields
for media paths, filenames, pixels, bounding boxes, timelines, OCR tokens, or
recognized values. Reports use private permissions and a sibling atomic-write
strategy. Write failures delete the partial file and suppress raw filesystem
exception details.

Phase 10 does not commit, upload, or automatically discover real acceptance
media. The opt-in runner accepts one explicit private directory, ignores
symlinks, assigns sequential anonymous identifiers, and never writes source
filenames or paths into its report. Outputs and reports remain in the explicitly
requested private destination. The source corpus remains under operator control.

Synthetic benchmark videos are generated under an operating-system temporary
directory and deleted at completion. Benchmark JSON contains only environment
labels, timing, counts, and golden-mask metrics. CI uses read-only repository
permissions and immutable action revisions. The Docker runtime is a dedicated
non-root user with a privately owned temporary job directory.

PaddleOCR receives the in-memory NumPy frame locally. The adapter never invokes
a service API, saves a result image, calls Paddle's JSON persistence helpers, or
writes an OCR transcript. PaddleX model-host connectivity probing is disabled.
Official model weights may be downloaded on first use; air-gapped deployments
must configure local detection and recognition model directories.

On supported macOS hosts, source analysis and verification may also use Apple's
local Vision framework as an accelerator while retaining periodic Paddle
audits. Source analysis additionally forces Paddle on scene changes and fresh
tracking risk, then backfills newly discovered masks across the bounded audit
interval. The adapter encodes each frame in memory and sends it to one private,
job-scoped native helper through bounded, length-prefixed standard-input frames.
It uses lossless BMP only when its exact size estimate fits the existing 64 MiB
input limit and otherwise falls back to lossless PNG; neither representation is
written to disk. Bounded results return through standard output. The helper is compiled into a
mode-`0700` temporary directory, receives no media path, and never writes a frame
or transcript. The pipeline closes the helper and removes its compiled temporary
files on success or failure. Unsupported or failed initialization uses the
Paddle-only path. A timeout, protocol error, or helper crash permanently disables
that accelerator instance and immediately continues with Paddle on the same
frame.

The Phase 1 media path still handles only coordinates. Its progress and error
messages never inspect or print frame contents. Temporary encoded media is held
in an operating-system temporary directory and removed after success or
failure. The original source and explicitly requested final output are the only
media files intended to remain.

Every mask receives safety padding and uses aggressive downsampling,
nearest-neighbor enlargement, and a three-box approximation of the configured
Gaussian strength. This is a privacy transform, not an aesthetic blur.

The output is staged before atomic replacement. Existing outputs are rejected
unless the caller explicitly supplies `--overwrite`; the input path itself can
never be the destination.

Automatic output reports `COMPLETED` and returns exit code 0 only when a complete
rendered-output pass finds no value-bearing sensitive detections. Label-only
semantic anchors are excluded because they indicate an input field, not readable
output content. Findings trigger bounded opaque-overlay repair. If findings
remain after the configured maximum passes, output is still delivered but is
explicitly `NEEDS_REVIEW` with exit code 3. Manual-mode output is never promoted
to `COMPLETED`.
