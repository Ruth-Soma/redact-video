# Verification and Automatic Repair

Phase 6 makes rendered-output verification mandatory for automatic success. A
successful render alone is not enough to produce `COMPLETED`.

## Safety scan

`SafetyVerifier` decodes and visually inspects every frame of the staged output.
It applies a compact 96-pixel luminance guard, then sends selected frames through
the configured local `OcrEngine` and the same payment-card detector family used
during initial analysis. The guard ignores codec deltas of up to 12 intensity
levels and at most 64 pixels—or 0.5% of its compact image, whichever is smaller.
Routine changed-frame OCR runs at most once per 100 ms, while a major scene
change bypasses that limit. An unchanged frame is forcibly rescanned within 500
ms. Reuse is disabled after a leak until a clean scan occurs. These bounds are
configurable with the `verification.reuse_*`, `minimum_scan_interval_ms`, and
`immediate_scene_change_fraction` settings; reuse can be disabled entirely.

After an opaque repair, the immediately following pass may also reuse a prior
clean decision when the frame index and timestamp are unchanged and the new
compact luminance guard remains inside the same strict codec-noise bounds.
Frames that leaked are never retained as clean evidence. Repaired frames whose
guards change, missing frames, timestamp changes, shape changes, and
nonconsecutive pass numbers all force a normal scan. The cache contains only
compact luminance guards, is replaced after each successful pass, and exists
only in job memory; it contains no OCR tokens or recognized text. Set
`reuse_clean_frames_across_passes: false` to disable this optimization while
retaining ordinary within-pass reuse. Setting `reuse_clean_frames: false`
disables both forms of reuse.

The first pass has no prior evidence, so it skips cross-pass guard matching
entirely. For a frame that requires OCR, the same computed guard becomes both
the new reuse reference and the clean evidence entry; it is never resized a
second time for those roles. This changes neither guard pixels nor scan policy.

With `selective_cross_pass_decode` enabled, the consecutive pass reads every
frame through an FFmpeg area-scaled grayscale stream whose longest dimension is
the configured guard size. Matching frames never cross the full-resolution raw
pipe. FFmpeg's `select` filter emits absolute full-resolution frames only for
missing evidence, prior leaks, timestamp/shape mismatches, or material guard
changes. The full decoder still processes those selected frames through the
normal Vision/ONNX detector policy. Selections are divided into ordered groups
of no more than 48 noncontiguous expression terms, avoiding FFmpeg evaluator
exhaustion without weakening or reindexing the requested frames. If compact
decoding is unavailable, the verifier discards partial guard evidence and falls
back to the ordinary full decode. A missing selected frame or selected-decode
failure remains fail-closed.

Verification scales every detector threshold down by `threshold_scale` (0.90 by
default), making the scan more privacy-sensitive.

On macOS, the default local Paddle pipeline is supplemented by Apple's on-device
Vision recognizer when `use_apple_vision_acceleration` is enabled. Vision checks
every frame selected by the visual guard, while Paddle remains authoritative: it
audits at least every 500 ms and runs immediately whenever Vision reports a
possible leak. Scheduled audit frames run the two independent recognizers
concurrently, then inspect and merge their tokens on one thread. Between audits,
two independent Vision processes can scan consecutive changed frames with a
bounded one-frame lookahead while leak decisions remain ordered. A single
decoder may stay up to 12 frames or 128 MiB ahead, whichever limit is reached
first, so the next scheduled ONNX audit overlaps ordered Vision work without a
second decoder or model instance. Immediate Vision findings still acquire the
same authoritative engine synchronously. Both Vision processes share one
compiled, job-private helper binary but retain separate pipes and failure state.
If Swift/Vision is not available at initialization or its helper fails during a scan,
verification uses the Paddle result already produced for that frame, disables
the accelerator, and processes subsequent frames with Paddle only. Frames and
OCR text travel only through memory and private process pipes; the native helper
never receives a media path or writes an image or transcript.

Set `prefetch_authoritative_audits: false` under `verification` to disable this
overlap. `authoritative_prefetch_frames` and
`authoritative_prefetch_max_bytes` bound its frame count and memory budget.

The detector intentionally creates `semantic_anchor` regions from visible field
labels so values are protected before typing begins. A label such as “Card
Number” remaining visible after rendering is not evidence that its value is
readable, so label-only anchors are excluded from leak results. Pattern,
semantic-value, expiry, CVV, and saved-card detections remain leak evidence.

All evidence is sanitized. `VerificationLeak` contains type, frame/time,
geometry, confidence, and detector source only. OCR text and recognized card
values are never copied into verification models, logs, or result summaries.

## Repair loop

Each leak timestamp becomes a window using these defaults:

```yaml
repair_pre_roll_ms: 350
repair_post_roll_ms: 500
repair_padding_px: 12
```

Overlapping windows are merged. For every candidate frame in a repair window, the
repair engine:

1. unions the sanitized leak boxes within each temporal window;
2. expands the union by the repair and renderer safety padding;
3. records the padded leak seeds on every corresponding timeline frame; and
4. burns an opaque fail-safe overlay directly into the already-redacted
   candidate with FFmpeg while copying its audio stream.

This surgical repair cannot reveal pixels protected by the first render and
does not repeat source OCR, tracking, or Python frame rendering. The resulting
full MP4 is probed for geometry, duration, and audio integrity, then every frame
is inspected through the compact stream. Strictly matching prior-clean frames
reuse their sanitized decisions; changed and previously leaking frames are
decoded at full resolution and rescanned. Repair therefore remains fail-closed
while avoiding a full source rebuild, redundant OCR, and unnecessary transfer
of full-resolution raw frames.

## Completion rules

`max_passes` defaults to 3. A pass with zero remaining value-bearing detections
sets the job to `COMPLETED` and the CLI exits 0. If findings remain after the
last pass, the final output is retained with `NEEDS_REVIEW`, CLI exit code 3,
and sanitized potential-issue time windows. The engine never reports success
merely because encoding or repair finished.

Setting `verification.enabled` to false is supported only for controlled
development; automatic output then remains `NEEDS_REVIEW` and cannot exit 0.
