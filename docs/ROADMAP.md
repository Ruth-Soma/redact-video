# Roadmap

## Phase 1 — Media pipeline

Implemented: FFprobe inspection, frame decoding, privacy rendering, H.264
encoding, audio muxing, manual-box CLI, input/output validation, tests, Docker
bootstrap, and documentation.

## Phase 2 — OCR

Implemented: `OcrEngine` protocol, lazy local PaddleOCR 3.x adapter, lightweight
English model configuration, constrained OCR normalization, serialization-safe
token models, malformed-result safeguards, and deterministic mock-backed tests.

## Phase 3 — Card detectors

Implemented: payment-context scoring, fuzzy semantic labels, field anchors,
plugin-style detector orchestration, local Luhn validation, full and partial
card-number detection, combined and split-field expiry detection, semantic
CVV/CVC/CSC/CID detection, configurable saved-card detection, sanitized outputs,
deduplication, and adversarial false-positive tests.

## Phase 4 — Tracking

Implemented: Lucas–Kanade optical flow, smoothed velocity prediction, bounded
template fallback, downscaled scene-change handling, deterministic track states,
mandatory sticky masks, adaptive OCR scheduling, and same-frame reacquisition.

## Phase 5 — Automated rendering

Implemented: deterministic value-free detection and mask timelines, ordered
tracking-result conversion, two-pass automatic analysis/rendering, privacy-blur
timeline application, audio-preserving MP4 output, automatic CLI mode, and safe
or fast OCR scan profiles.

## Phase 6 — Verification and repair

Implemented: every-frame rendered-output rescanning with privacy-sensitive
thresholds, value-bearing leak classification, merged pre/post-roll repair
windows, padded seed masks, deterministic timeline patching and direct
fail-safe candidate overlays, a three-pass default cap, `COMPLETED` only after
a clean scan, and fail-closed `NEEDS_REVIEW` results for persistent leaks.

## Phase 7 — Full synthetic corpus

Implemented: generated portrait/landscape payment screens, frame-indexed OCR
oracles, per-frame sensitive and visible ground truth, PAN/CVV typing, slow and
fast scrolling, modal and disappear/return transitions, saved cards, brief
appearances, encoded false positives, intentional repair, sensitive coverage,
missed-pixel, over-mask, and visible-region metrics. All required scenarios have
zero uncovered sensitive pixels.

## Phase 8 — HTTP API

Implemented: thin FastAPI routes for multipart job creation, sanitized status,
terminal output download, and value-free reports; bounded process-pool workers;
standard/safe/fast profiles; upload limits; queue saturation handling; private
job directories; terminal raw-upload deletion; output retention cleanup; fixed
public worker errors; an injectable executor boundary; API tests; and Docker and
local server entry points.

## Phase 9 — Configuration and reports

Implemented: strict safe-loaded YAML mapping for the configuration contract in
section 40; unknown-field rejection; sanitized validation failures; validated
default/YAML/CLI precedence; shared API-worker configuration; specification
codec alias normalization; core value-free report models; atomic private JSON
report output; collision and overwrite safeguards; and configuration/report
unit and integration tests.

The implementation phases and the remaining V1 configuration/report contract
are complete.

## Phase 10 — Release and acceptance hardening

Implemented: read-only, immutable-action CI; Python 3.12/3.13 coverage gates;
locked linting; wheel/sdist and installed-command checks; Docker build, CLI,
configuration, non-root, and API smokes; a separate real PaddleOCR workflow; a
repeatable synthetic privacy/performance benchmark; an anonymous external-corpus
acceptance runner; private aggregate reports; release documentation; and
representative-recording manual review criteria.

## Phase 11 — Performance scaling

In progress. Sanitized pipeline reports now include exact millisecond timings
for inspection, setup, source analysis, initial rendering and validation, every
verification and repair pass, publication, and total runtime. The first timed
private acceptance replay kept zero remaining leaks and measured source analysis
at 49.0% of runtime and verification at 34.4%; those OCR-heavy stages are the
next optimization boundary. Rendering is no longer the dominant stage. Bounded
authoritative-audit prefetch is implemented with the existing decoder and ONNX
engine; on the same private replay it reduced two-pass verification from 91.222
to 74.286 seconds while preserving four repair windows and zero remaining leaks.
Source analysis profiling then attributed 87.726 seconds to 15,250 per-region
Lucas–Kanade estimates. Capping the default flow feature budget at 40 while
retaining per-frame estimation and the original pyramid depth reduced an
accepted replay to 216.318 seconds (3 minutes 36 seconds): 117.165 seconds of
source analysis, two verification passes, three repair windows, preserved
audio, and zero remaining leaks. A cadence-reduction candidate was rejected
because its extra repair and verification pass increased total time.

The next source slice overlaps independent per-region LK estimates only on
frames of at least one megapixel. A representative eight-region frame improved
from 54.73 ms sequentially to 25.22 ms with four outer workers and OpenCV inner
parallelism disabled (2.17×), without changing motion results. Nested four-by-ten
threading was rejected after it collapsed under host contention. A contended
exact-code replay completed in 300.153 seconds (5 minutes 0.153 seconds) with
the same 1,812 masked frames, 13,460 regions, three repairs, two verification
passes, and zero leaks as the accepted baseline. A change-driven crop-OCR
experiment was also removed after it ended `NEEDS_REVIEW` with two residual
expiry findings.

Consecutive verification passes now retain only sanitized clean-frame guards
and reuse them when frame identity, timestamp, shape, and the strict codec-noise
comparison all match. Previously leaking and changed repair frames always
rescan. On the same controlled sample under heavy host contention, pass 2 fell
from 45.816 to 39.097 seconds (14.7%) with unchanged privacy counts: 1,812
masked frames, 13,460 regions, three repairs, two passes, and zero remaining
leaks. Full-frame decoding now dominates that pass and is the next bounded
verification optimization target.

That decoder slice is now implemented. Consecutive passes inspect all frames in
a 96-pixel grayscale FFmpeg stream and emit full-resolution frames only for
guard mismatches, missing evidence, and prior leaks. On the controlled sample,
pass 2 fell again from 39.097 to 9.969 seconds, a 74.5% reduction, while the
complete contended run finished in 241.109 seconds (4 minutes 1.109 seconds).
The accepted counts remained unchanged: 1,812 masked frames, 13,460 regions,
three repairs, two verification passes, preserved audio, and zero leaks. The
slice also closed a stale authoritative-prefetch race that could wait forever
when a short decoder finished before a later audit was scheduled.

A compact first-pass planning experiment was subsequently rejected. Although it
reduced full-resolution first-pass selections to 144 of 1,815 frames, it changed
OCR ordering, added a third verification pass and fourth repair, and manual
comparison found visible UI overmasking. The candidate path and configuration
were removed. The experiment did uncover a separate FFmpeg selector scalability
bug, so selected decoding now partitions long noncontiguous lists into ordered,
bounded 48-term expressions and fails closed on omissions. The next performance
slice must preserve first-pass OCR ordering or target source analysis instead of
preclassifying first-pass frames as clean.

The first accepted source-analysis follow-up removes repeated detector work
without changing scan cadence or ordering. `DetectionEngine` now builds one
maximum-width adjacent-token span set per OCR result, and payment context plus
all detector plugins consume immutable filtered views. Guaranteed fuzzy-match
upper bounds skip full SequenceMatcher ratios only when acceptance is
mathematically impossible. On the controlled replay, source analysis fell from
107.504 to 99.891 seconds (7.1%) and total runtime reached 235.432 seconds. The
approved counts remained exactly 1,812 masked frames, 13,460 regions, 13 tracks,
three repairs, two passes, preserved audio, and zero leaks; all 1,815 decoded
video frames were pixel-identical to the previously approved output. A source
decode-thread prototype was rejected because its 99.478-second isolated run did
not beat the 97.117-second sequential profile under comparable load.

Scheduled OCR and active-track advancement now overlap on the same immutable
source frame and join before detections are applied. The full controlled replay
reduced source analysis again from 99.891 to 98.440 seconds (1.5%) and completed
in 228.184 seconds. The accepted counts remained exact, with zero leaks and all
1,815 decoded output frames pixel-identical to the prior result. Whole-frame and
per-region exact-static shortcuts were rejected as immaterial after measuring
only one identical frame pair and 359 unchanged flow windows out of 12,914.

The next accepted OCR slice replaces per-scan PNG compression at the private
Apple Vision pipe boundary with bounded, lossless BMP. Seven representative
frames improved from 43.929 to 28.810 ms per warm Vision scan (1.53×), with
identical decoded pixels, OCR token counts, detection types, and geometries.
The controlled full replay completed in 205.970 seconds (3 minutes 25.970
seconds), 22.214 seconds (9.7%) faster than the preceding 228.184-second run;
source analysis fell from 98.440 to 80.901 seconds. It retained the exact
accepted privacy counts, AAC audio, two passes, three repairs, zero leaks, and
all 1,815 decoded output frames were pixel-identical. PNG remains the automatic
fallback whenever BMP would exceed the existing 64 MiB input boundary or BMP
encoding is unavailable. First-pass verification, now 75.253 seconds, is the
largest remaining measured stage.

The accepted verification follow-up removes duplicate guard construction
without changing the 96-pixel luminance algorithm. Pass 1 now skips prior-pass
matching when no prior evidence can exist, and an OCR frame shares one guard
between its reuse reference and clean-evidence record. On a controlled clean
pass, guard builds fell from 3,694 to exactly 1,815 and verification fell from
66.174 to 47.755 seconds (27.8%). The full replay reduced first-pass
verification from 75.253 to 50.771 seconds (32.5%) and total runtime from
205.970 to 187.272 seconds (3 minutes 7.272 seconds). The exact accepted counts,
AAC audio, two passes, three repairs, zero leaks, and all 1,815 decoded output
frames remained unchanged. The remaining 1,815 required OpenCV guard builds
consumed 33.273 seconds in the isolated profile and are the next concrete
verification optimization boundary.
