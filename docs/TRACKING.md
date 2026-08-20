# Tracking

Phase 4 turns frame-local `SensitiveDetection` objects into temporal tracks
without requiring OCR on every video frame. Tracking remains frame-oriented and
does not depend on a video path, CLI, renderer, or remote service.

## State machine

```text
new detection -> DETECTED -> TRACKING
                              |     ^
                   confidence |     | OCR reacquisition
                         lost v     |
                            STICKY ->+
                              |
                   350 ms expires
                              v
                             LOST

predicted region fully leaves frame -> ENDED
```

`STICKY` is maskable just like `DETECTED` and `TRACKING`. A failed motion
estimate never removes a region immediately. The tracker continues applying its
recent velocity through the configured `sticky_duration_ms`, including the
exact duration boundary. Only then can it become `LOST`. A region predicted
fully outside the frame is positively treated as `ENDED`.

## Motion and fallback order

The primary estimator finds feature points in and around the prior region and
runs pyramidal Lucas–Kanade optical flow with a forward/backward consistency
check. Median point displacement supplies robust horizontal and vertical
translation. Motion is still estimated on every active frame. The default
feature budget is capped at 40 points per region; additional corners did not
improve accepted motion coverage but increased both the forward and backward
LK work. The original four-level pyramid depth remains in place so structured
fields can still survive large frame-to-frame translations.

On frames of at least one megapixel, independent active-region estimates run on
a bounded four-worker pool and are joined in deterministic track order before
any state is mutated. OpenCV's process-wide inner pool is normalized to one
thread when outer flow parallelism is enabled, preventing nested 4×10
oversubscription. Smaller frames stay sequential because thread coordination
costs more than their LK work. The pool holds only shared
references to the current and previous in-memory frames and is shut down at the
end of the job.

When flow is weak, the tracker predicts the next rectangle from smoothed recent
velocity and searches a bounded area with normalized template matching. The
template is extracted from the previous in-memory frame and is discarded after
the comparison. If the fallback also fails, the predicted rectangle becomes a
sticky region and OCR is requested immediately.

## Scene changes and adaptive OCR

Every frame is compared with its predecessor at a small grayscale resolution.
A large changed-pixel fraction indicates a transition, modal, large scroll, or
screen replacement. Existing regions become sticky for safety and the current
frame is sent through OCR and detection immediately.

`AdaptiveTrackingEngine` otherwise scans at `DetectionConfig.scan_fps` rather
than at the source frame rate. Active tracks are also bounded by
`reacquire_interval_ms`; sticky tracks use `safe_scan_fps`. The first frame is
always scanned. `None` detections mean OCR was not run, while an empty list means
OCR ran and found no sensitive region.

When a scheduled scan occurs while tracks are already active, the read-only OCR
scan and that frame's ordered optical-flow advancement run concurrently on one
bounded source-OCR worker. The engine waits for both before applying detections
to the current frame, so scan cadence, frame order, detector output, motion
fallbacks, and same-frame reacquisition are unchanged. Set
`overlap_source_ocr_and_tracking: false` to retain the fully sequential source
path. The worker is job-scoped and is shut down before tracker cleanup.

On supported macOS hosts, routine source scans use Apple Vision's `fast` mode
when `use_apple_vision_acceleration` is enabled. PP-OCRv6 tiny through
ONNX Runtime remains the independent audit and fallback engine: it scans the
first frame, audits at least every `authoritative_audit_interval_ms` (350 ms by
default), and is forced on scene changes or fresh tracking-confidence loss.
Audit findings are merged with the native findings, with ONNX geometry winning
for overlapping regions. On audit frames, the two independent OCR recognizers
run concurrently while detection and merging remain single-threaded. A native
runtime failure uses the ONNX result already produced for that frame and
disables Apple Vision for the rest of the job.

Scheduled ONNX audits may reuse the preceding authoritative detections only
when a full-color comparison finds no more than 64 pixels—or 0.002% of the
frame, whichever is smaller—that changed by over 6 intensity levels. Reused
detections are retimed to the current frame and contain no OCR text. Scene
changes and tracking-risk reacquisitions always force a fresh Paddle scan.

When a new track is first discovered, its current geometry is conservatively
backfilled across the preceding authoritative audit interval plus one complete
scheduled OCR interval. The extra scheduler allowance covers the delay between
an audit becoming due and the next sampled frame. This protects frames between a
sensitive value appearing and the next ONNX audit, including cases where the
native recognizer did not detect that value.

## Public output

`TrackingFrameResult.regions` is the set of regions that Phase 5 records and
masks on the current frame. `MaskTimelineBuilder` consumes ordered frame results
and copies only type, track ID, and geometry into the renderer timeline. Direct
tracker results include immutable snapshots of all tracks by default. Streaming
analysis and repair omit those repeated history snapshots and materialize the
final track summary once after analysis. Frame results also include flags showing
whether a scene changed, OCR was performed, or unresolved risk still requires
reacquisition. None of these models can contain OCR text or recognized
payment-card values.
