# Automated Rendering

Phase 5 connects automatic detection and tracking to the existing privacy
renderer. Normal operation requires only an input path:

```bash
redact-video input.mp4
```

## Timeline models

`DetectionTimeline` groups sanitized detections by frame with deterministic
ordering. `MaskTimelineBuilder` accepts strictly increasing
`TrackingFrameResult` values and produces an immutable `MaskTimeline` containing
only frame indices and `MaskRegion` values. A mask region contains a constrained
track ID, sensitive category, and bounding box. No timeline type has a field for
OCR text or a recognized card value.

The mask timeline is retained in memory for the duration of the job and is
excluded from `AutomatedProcessResult` serialization. It is not written as JSON
or another sidecar file.

## Two-pass pipeline

The automatic pipeline performs two deterministic media passes:

1. Decode the display-oriented source frames, run adaptive OCR, detection, and
   tracking, and build the complete mask timeline.
2. Decode the source again, apply each frame's timeline regions through the
   pixelation-plus-Gaussian-equivalent privacy renderer, encode H.264/yuv420p,
   and mux the original audio. Three constant-time box filters approximate the
   configured Gaussian strength without changing the pixelation stage.

When a source audio stream ends materially before its video stream, muxing pads
the remainder with silence instead of allowing FFmpeg's shortest-stream behavior
to truncate redacted video frames.

The pipeline rejects a frame-count mismatch between passes. The muxed file is
probed in a uniquely named staging path and atomically replaces the requested
destination only after its displayed geometry, video duration, required audio,
and audio/video synchronization are validated. Duration drift of 100 ms or more
is rejected. Temporary output is removed if processing fails.

If rendered-output verification finds a leak, the pipeline does not repeat
these two source passes. It unions and pads each temporal repair window, burns
an opaque fail-safe overlay directly into the staged candidate with FFmpeg,
copies the existing audio stream, validates the repaired MP4, and verifies the
complete output again.

## CLI profiles and status

The default OCR scheduling uses the configured normal and risk scan rates.
`--safe` raises the normal rate to the configured privacy-oriented scan rate.
`--fast` halves both rates for development, with a floor of one scan per second.
The two options are mutually exclusive and do not apply to manual mode.

Automatic output reports the number of processed and masked frames, mask-region
applications, sanitized tracks by category, verification passes, repair windows,
and remaining possible leaks. Phase 6 now promotes a clean result to
`COMPLETED` with exit code 0. Persistent findings and all manual renders remain
`NEEDS_REVIEW` with exit code 3.
