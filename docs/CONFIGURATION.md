# Configuration and Reports

Phase 9 adds the YAML configuration and sanitized report contracts described in
sections 40–41 of the V1 specification.

## YAML configuration

Start from [`config.example.yaml`](../config.example.yaml). The CLI accepts it
with `--config`/`-c`:

```bash
redact-video recording.mp4 \
  --config config.example.yaml \
  --output recording.redacted.mp4
```

The document is parsed with a bounded PyYAML safe loader, is limited to 1 MiB,
caps aliases and composed nodes, and must use a `.yaml` or `.yml` extension.
Unknown top-level or nested fields are rejected. Validation errors report only
field locations—not submitted values or raw parser context.

All sections are optional and omitted values use the validated application
defaults. The accepted top-level sections are:

- `redaction`: `card_details` policy and privacy-blur settings
- `detection`: scan rates, thresholds, weights, and detector settings
- `tracking`: motion, sticky-mask, scene-change, and reacquisition settings
- `verification`: pass limit, thresholds, and repair windows
- `output`: H.264, quality, pixel format, and audio settings
- `ocr`: local model, inference-engine, and Apple Vision settings
- `media_limits`: accepted frame dimensions and pixel count

H.264 output uses the x264 `veryfast` preset by default. This keeps CRF 18
quality while substantially reducing render time for high-resolution screen
recordings. All named x264 presets from `ultrafast` through `veryslow` remain
available under `output.preset`; unknown preset names are rejected during
configuration validation.

On supported macOS hosts, `tracking.use_apple_vision_acceleration` enables
Apple Vision `fast` OCR for routine source scans. PP-OCRv6 tiny ONNX audits those scans every
`tracking.authoritative_audit_interval_ms` (350 ms by default). Masks for newly
discovered tracks are backfilled over that interval plus one scheduled OCR
interval so frame-cadence quantization cannot leave an uncovered gap.
Visually unchanged scheduled audits can reuse sanitized ONNX detections under
the strict `tracking.authoritative_reuse_*` pixel bounds. Forced scene-change
and tracking-risk audits never use this optimization.
The tracker evaluates motion on every active frame with a default maximum of 40
feature points per sensitive region. `tracking.maximum_flow_points` can be
raised for unusual high-texture footage, while the default
`tracking.flow_pyramid_levels: 3` retains four search levels for large motion.
For frames at or above `tracking.parallel_flow_min_frame_pixels` (one megapixel
by default), independent region estimates use
`tracking.optical_flow_workers` (four). Enabling outer flow parallelism
normalizes OpenCV's process-wide inner pool to one thread. Set
`optical_flow_workers: 1` to disable outer parallelism.
Scheduled source OCR overlaps that frame's independent ordered motion estimate
by default and is joined before detections mutate track state. Set
`tracking.overlap_source_ocr_and_tracking: false` to disable this job-scoped,
single-worker overlap without changing OCR or tracking policy.
Verification has independent acceleration and audit settings under
`verification`. Its 96-pixel luminance guard ignores bounded codec noise,
routine changed-frame OCR is capped at one scan per 100 ms, major scene changes
scan immediately, and clean frames are forcibly rescanned within 500 ms.
After an opaque repair, `verification.reuse_clean_frames_across_passes` reuses
only immediately preceding clean evidence whose frame index, timestamp, shape,
and compact guard still match. Previously leaking or changed frames rescan.
Set it to `false` for independent OCR scheduling on every pass; setting
`verification.reuse_clean_frames: false` disables reuse within and across
passes.
`verification.selective_cross_pass_decode` keeps that comparison in a compact
FFmpeg grayscale stream and emits full-resolution frames only when they require
a new decision. Set it to `false` to retain full-frame decoding while still
allowing guarded decision reuse.
Scheduled authoritative verification audits are prefetched by default through
the existing decoder and ONNX engine. The producer may lead ordered consumption
by at most `verification.authoritative_prefetch_frames` (12) and
`verification.authoritative_prefetch_max_bytes` (128 MiB); set
`verification.prefetch_authoritative_audits: false` to use synchronous audits.

Oversized local OCR inputs are reduced proportionally to an
`ocr.max_input_dimension` of 1920 pixels by default. Authoritative ONNX uses
that outer ceiling and applies an 832-pixel
`ocr.text_detection_limit_side_length` inside its text detector. The
higher-frequency Apple Vision accelerator has an additional
`apple_vision_max_input_dimension` ceiling of 1440 pixels; the lower of the two
limits applies. Recognized boxes are mapped back to exact source-frame
coordinates before detection and masking. If a reduced authoritative ONNX scan
produces no usable text, `ocr.full_resolution_fallback` retries the original
frame. Apple Vision skips that duplicate scan by default because ONNX provides
the independent audit/fallback; set
`ocr.apple_vision_full_resolution_fallback: true` to restore it. Set
`max_input_dimension: null` to disable scaling for both engines, or raise the
Apple-specific ceiling to match Paddle when desired.

ONNX/Paddle uses the available local CPU count with a default ceiling of eight
threads per OCR job. This preserves lower thread counts on smaller machines
while using the performance cores of larger local hosts. Set `ocr.cpu_threads`
explicitly to reserve capacity when running multiple API workers or sharing a
machine with other workloads.

Executable overrides still come from `REDACT_VIDEO_FFMPEG` and
`REDACT_VIDEO_FFPROBE`. Precedence for automatic CLI jobs is:

```text
validated defaults and executable environment
  -> YAML configuration
  -> --safe or --fast scan-rate override
```

The API uses the same loader when `REDACT_VIDEO_CONFIG` points to a YAML file.
The server validates it at startup, and every child worker applies its requested
`standard`, `safe`, or `fast` mode after loading that base configuration.

## Sanitized JSON reports

Automatic CLI jobs can atomically write a report:

```bash
redact-video recording.mp4 \
  --report recording.report.json
```

The destination must end in `.json`, must differ from both video paths, and is
not replaced unless `--overwrite` is present. New reports use private `0600`
permissions. A sibling partial file is flushed and then installed atomically;
failures remove the partial artifact and replace filesystem details with a fixed
error.

Reports contain only:

- terminal status
- input/output durations and processed-frame counts
- mask counts and detection counts by sensitive category
- verification passes, repairs, remaining-leak count, and timestamp-only issue
  windows
- millisecond timings for inspection, setup, source analysis, initial rendering,
  validation, each verification and repair pass, publication, and total runtime
- audio, variable-frame-rate, and sanitized-warning metadata

They never contain input/output paths, filenames, mask timelines, bounding
boxes, frames, OCR text, or recognized card values. Manual-mask jobs cannot
write this report because they do not execute automatic verification.
