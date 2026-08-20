# Architecture

The command-line interface is a thin adapter over the manual and automatic
pipeline operations. Media, configuration, redaction, and pipeline code do not
import the CLI or HTTP adapter, so both interfaces call the same core operations.

## Phase 1 data flow

```text
MP4/MOV
  -> ffprobe validation
  -> display-oriented FFmpeg raw-frame decoder (OpenCV VFR fallback)
  -> static manual Mask regions
  -> privacy-blur renderer
  -> FFmpeg H.264 encoder
  -> original-audio muxer
  -> staged MP4 validation
  -> atomic output replacement
```

`FrameDecoder` yields frames rather than accepting a file path in detector or
renderer code. This preserves the frame-oriented boundary required by future
stored-video and live-capture products.

The renderer accepts domain `BoundingBox` values and is independent of how a
box was discovered. Manual regions and automatic timeline regions therefore
share the same privacy transform.

## Phase 2 OCR boundary

```text
display-oriented BGR frame
  -> OcrEngine.scan(frame)
  -> job-scoped macOS Vision helper through private in-memory pipes
     with PP-OCRv6 tiny ONNX audits/fallback
     or PP-OCRv6 tiny ONNX directly on other platforms
  -> strict result-shape validation
  -> transient OcrToken(text, confidence, bbox)
  -> detector boundary
```

Only `ocr/paddle.py` knows about PaddleOCR, and only `ocr/apple_vision.py` knows
about the optional native accelerator. Detectors consume the stable `OcrToken`
boundary and can use deterministic mock engines in tests. PaddleOCR is imported
and initialized on the first scan, so media-only commands do not load models or
require the inference runtime.

The default ONNX/Paddle worker count adapts to the local CPU count and is capped at
eight threads per OCR job. Explicit configuration still wins, allowing
multi-worker services and constrained containers to budget cores deliberately.

The Vision adapter compiles one private helper binary per automatic job and
reuses its process through a bounded length-prefixed protocol. Parallel
verification starts a second independent process from a job-private hard link
or copy of that binary instead of compiling the Swift source twice. This avoids
per-frame process startup while keeping frames and OCR results in memory.
Pipeline cleanup closes both helpers and their pipes on success and failure. It
passes ordinary frames as lossless BMP to avoid PNG compression work, but only
when the exact encoded-size estimate fits the existing 64 MiB helper boundary;
oversized inputs or unavailable BMP encoding fall back to lossless PNG. It uses
a dedicated 1440 pixel ceiling beneath the shared OCR input limit, while
authoritative ONNX retains the 1920 pixel default. Both engines remap
coordinates to the source frame. Authoritative ONNX retains a full-resolution
retry after empty reduced scans; Vision does not repeat that expensive scan by
default because ONNX supplies the independent audit and fallback result.
PP-OCRv6's text detector uses a separate 832-pixel internal side limit so its
tiny detection model avoids unnecessary work while recognition boxes are
still mapped through the 1920-pixel outer frame to source coordinates.
When an authoritative audit is due, Vision and ONNX recognition overlap; their
tokens still cross the detector boundary serially so detector semantics remain
deterministic.

The adapter consumes the PaddleOCR 3.x result fields `rec_texts`, `rec_scores`,
and `rec_boxes`, with `rec_polys` as a geometry fallback. Missing, mismatched,
non-finite, or empty geometry fails the scan rather than returning a misleading
safe result.

## Phase 3 detector boundary

```text
display-oriented BGR frame + transient OCR tokens
  -> adjacent-token span lattice (once per scanned frame)
  -> payment-context analysis over the shared spans
  -> semantic field anchors
  -> plugin-style SensitiveDetector implementations
       -> card number (semantic + pattern/Luhn)
       -> expiry (combined + split fields)
       -> CVV/CVC/CSC/CID (semantic only)
       -> saved card and saved expiry
  -> same-type region deduplication
  -> SensitiveDetection(type, frame/time, bbox, confidence, source)
```

`DetectionEngine` is frame-oriented and has no file-path, CLI, PaddleOCR, or
renderer dependency. It builds one immutable `DetectionContext` per frame, then
runs a sequence of detectors implementing `SensitiveDetector`. That context
contains one transient maximum-width OCR span set; detector-specific narrower
views filter it by token count, preserving the exact standalone span result
without repeating sorting, line grouping, text joining, or box unions. Custom
future detectors can be injected without changing orchestration.

OCR text exists only in transient tokens and internal candidate objects.
Detector outputs carry classification, geometry, confidence, and a sanitized
source identifier; they have no recognized-value field.

## Phase 4 tracking boundary

```text
ordered display-oriented frames
  -> cheap downscaled scene-change comparison
  -> periodic local OCR || current-frame optical flow when tracks are active
  -> join both before applying detections
  -> risk-triggered same-frame OCR + detection
  -> detection-to-track association
  -> Lucas-Kanade local translation
       -> velocity prediction
       -> bounded template matching
       -> STICKY prediction + immediate OCR reacquisition
  -> TrackingFrameResult(regions, track histories, scan flags)
```

`RegionTracker` consumes only frames and sanitized detections. It retains one
previous frame in memory for optical flow and template matching, never writes a
template to disk, and emits deterministic `SensitiveTrack` histories.
Optical flow remains per-frame and forward/backward checked, with at most 40
feature points per region by default and the original four-level pyramid for
large translations.
High-resolution active-region estimates may execute on four bounded workers,
but results are consumed in track order before state updates. OpenCV uses one
process-wide inner thread when outer parallelism is enabled to prevent nested
oversubscription, and frames below one megapixel remain sequential.
Scheduled source OCR uses one separate job-scoped worker while the main thread
advances existing tracks on the same immutable frame. The OCR future is joined
before detections are applied, so the state mutation sequence remains optical
flow followed by detection association, exactly as in the sequential path.
`AdaptiveTrackingEngine` composes the existing `OcrEngine`, `DetectionEngine`,
and tracker while avoiding OCR on every frame. Scene changes and confidence
loss override the periodic schedule and reacquire on the current frame.

## Phase 5 automated rendering boundary

```text
pass 1: decoded frames
  -> AdaptiveTrackingEngine
  -> ordered TrackingFrameResult values
  -> MaskTimelineBuilder
  -> immutable, value-free MaskTimeline

pass 2: freshly decoded frames + MaskTimeline
  -> privacy-blur renderer
  -> H.264/yuv420p encoder
  -> original-audio muxer
  -> staged media validation
  -> staged candidate
```

`DetectionTimeline` provides deterministic frame grouping at the detector
boundary. The automatic pipeline consumes tracking results directly because
adaptive tracking already owns scan scheduling and detection. For every frame,
`MaskTimelineBuilder` copies only track ID, sensitive type, and geometry into an
in-memory timeline. OCR text and recognized values cannot enter that model.

The production encoder defaults to x264 `veryfast` at CRF 18. The preset reduces
the CPU-bound render/encode stage without changing frame geometry, pixel format,
audio handling, or verification requirements.

Analysis and rendering are separate passes. The renderer never depends on
mutable tracker state, and the completed timeline is available to the Phase 6
repair engine. The public result excludes the timeline from serialization.

## Phase 6 verification and repair boundary

```text
staged rendered candidate
  -> first pass: full decode with compact codec-noise/scene guards
  -> repair pass: compact grayscale decode of every frame
  -> optionally match sanitized clean evidence from the immediately prior pass
  -> selective full-resolution decode for unmatched absolute frame indices
  -> cadence-bounded local OCR + lower-threshold DetectionEngine
  -> discard label-only semantic anchors
  -> readable-value leak evidence?
       no  -> COMPLETED -> atomic output replacement
       yes -> merge pre/post-roll repair windows
            -> union and pad sanitized leak boxes per window
            -> add fail-safe leak seeds to MaskTimeline
            -> burn opaque timed overlays into the staged candidate
            -> verify again (up to max_passes)
            -> NEEDS_REVIEW if findings remain
```

`SafetyVerifier` uses the same detector family as initial analysis with all
per-type thresholds scaled down. It does not classify a visible field label as
a leak: Phase 3 semantic anchors intentionally protect blank or not-yet-typed
fields, but do not prove that a value remains readable in rendered output.
The guard examines every decoded frame, immediately scans major scene changes,
and limits routine OCR to 10 scans per second by default. Clean near-identical
frames may be reused for at most 500 ms, and independent ONNX audits remain
forced every 500 ms when Vision acceleration is active.

On a consecutive post-repair pass, a frame with the same index, timestamp,
shape, and bounded compact guard can carry forward its prior clean decision.
Leak frames are excluded from that cache, so the pixels targeted by repair must
be inspected again. Evidence is committed only when a pass finishes and is
replaced on the next pass; it contains compact luminance pixels but no OCR text.
The repair pass obtains its guards directly from a tiny FFmpeg grayscale stream,
then uses an FFmpeg `select` expression to output only unmatched full-resolution
frames. Noncontiguous selections are split into bounded groups of at most 48
expression terms so FFmpeg's evaluator cannot be exhausted by a long repair
list. Every group still uses absolute frame indices and is emitted in order;
an omitted frame or decoder error fails verification closed. Compact decoding
failure clears partial evidence and restores the ordinary full decoder.

The verification decoder uses a bounded producer queue so it can reach an
upcoming audit frame while ordered Vision decisions consume earlier frames. The
existing authoritative ONNX engine scans that future frame on one serialized
worker; no second model or duplicate decode pass is created. The queue is
limited independently by frame count and bytes, and an immediate risk scan
shares the same engine lock. If the decoder has already passed a newly scheduled
audit frame, its still-pending future is cancelled and the ordered consumer runs
the authoritative engine synchronously instead of waiting indefinitely.

`RepairEngine` expands each finding into a configured temporal window and
merges overlaps. Leak geometry is value-free, conservatively unioned and
padded, then recorded across every affected timeline frame. FFmpeg applies the
same timed regions as opaque fail-safe overlays to the already-redacted staged
candidate and copies its audio. This avoids repeating source OCR, source
tracking, and Python frame rendering; the complete repaired MP4 is still probed
and compactly inspected while changed or previously leaking frames are fully
decoded and rescanned. Staged output is atomically installed only after the
verification loop ends, whether as `COMPLETED` or explicitly `NEEDS_REVIEW`.

## Phase 7 synthetic corpus boundary

```text
scenario definition
  -> generated BGR phone/browser frames
  -> embedded compression-resistant frame index
  -> H.264 MP4 fixture in a pytest temporary directory
  -> decoded-frame OCR oracle
  -> complete automatic pipeline
  -> final MaskTimeline
  -> rasterized golden-frame comparison
       -> sensitive coverage
       -> missed sensitive pixels
       -> over-mask area
       -> masked expected-visible pixels
```

The corpus is test-only and does not add scenario knowledge to production
detectors. The OCR oracle reads a small frame marker from the actual decoded
image, then returns that frame's predefined synthetic tokens. Tracking,
rendering, encoding, verification, and repair therefore operate over real
compressed video frames without depending on Paddle model downloads in normal
CI. The real Paddle boundary remains covered by the opt-in runtime smoke test.

The corpus exposed a viewport-exit edge case: a motion prediction could end a
track while a clipped value remained visible for one frame. An off-screen track
transition now requests immediate same-frame OCR reacquisition. A truly absent
region remains `ENDED`, while any residual readable value is rediscovered and
masked.

## Phase 8 HTTP boundary

```text
multipart MP4/MOV upload
  -> bounded request and private job directory
  -> QUEUED response
  -> bounded local ProcessPoolExecutor
  -> process_video_automated
  -> sanitized in-memory result
  -> status / output / report routes
  -> retention cleanup
```

FastAPI handlers persist the upload and submit a `WorkerRequest`; they never run
OCR, tracking, rendering, or verification directly. The child-process entry
point calls the same `process_video_automated` operation as the CLI. Its result
crosses the worker boundary without the output path, status duplication, mask
timeline, OCR tokens, or recognized values.

The local manager bounds active plus pending work with a semaphore before it
creates a job directory. Its executor is injectable, defining the adapter seam
for a future Redis/Celery-style backend while keeping local development free of
external infrastructure. Child progress is communicated through a small,
value-free temporary state file because process memory is isolated. Terminal
results move into the parent process's in-memory registry and the state file is
deleted.

## Phase 9 configuration and report boundary

```text
safe-loaded YAML
  -> strict ConfigurationDocument
  -> immutable AppConfig
  -> optional safe/fast scan override
  -> shared automatic pipeline

AutomatedProcessResult
  -> value-free SanitizedProcessReport
  -> private sibling partial JSON
  -> atomic report destination
```

The external YAML shape is separate from `AppConfig`, allowing the documented
`redaction.policy` and `redaction.blur` grouping without leaking interface
concerns into media, detector, or tracker code. All nested runtime models forbid
unknown fields. The loader preserves executable environment overrides, maps the
specification's `h264` spelling to the concrete `libx264` encoder, and returns
only sanitized field locations when validation fails.

The report mapper explicitly copies approved scalar/count models out of
`AutomatedProcessResult`. It cannot serialize the output path or excluded mask
timeline because neither exists in its schema. CLI files add terminal status;
the API embeds the same value-free details beneath the job-level status.

## Phase 10 release and acceptance boundary

```text
generated corpus
  -> complete injected-OCR pipeline
  -> golden privacy metrics + elapsed time
  -> value-free benchmark report

external non-production corpus
  -> anonymous sequential sample IDs
  -> normal local-Paddle verified pipeline
  -> anonymous redacted outputs
  -> value-free acceptance report + manual review
```

Benchmark code remains repository-only because it deliberately uses the test OCR
oracle and golden truth. The installed acceptance command uses the production
pipeline and local Paddle adapter. It never passes a source filename into a
report or output name; only ordered `sample_NNN` identifiers cross its reporting
boundary.

CI separates deterministic mock/oracle tests from the network-dependent model
runtime smoke. Packaging and container jobs consume only public source files.
The runtime image switches to a dedicated non-root user after installation.

## Timing and orientation

FFprobe records average and nominal rates. Near-CFR inputs retain their average
rate. Clearly variable-rate inputs are decoded in presentation order and
normalized to deterministic CFR at the average rate; the sanitized result
contains a warning. OpenCV applies display rotation during decoding, and the
encoder writes physically oriented frames without copied rotation metadata.

## Output safety

Frames are first encoded in a per-job temporary directory. The audio-muxed file
is written to a uniquely named sibling of the destination, probed again, and
only then atomically moved into place. Exceptions remove staged outputs, and
temporary job artifacts are automatically deleted.
