# Release and Acceptance

Phase 10 defines repeatable release gates for the completed V1 engine. Real
acceptance recordings remain private and external to the repository; generated
reports use anonymous sample identifiers and never retain source filenames.

## Automated release gates

The primary GitHub Actions workflow runs:

- lock-file integrity and Ruff
- the deterministic suite with at least 85% coverage on Python 3.12 and 3.13
- wheel and source-distribution builds plus installed-command smoke checks
- the static synthetic privacy/performance benchmark
- Docker build, configuration loading, non-root identity, CLI, and API health
  checks

All third-party actions are pinned to immutable release commits and workflow
permissions are read-only. Pull-request runs do not install PaddleOCR or download
model weights.

The separate `Local OCR smoke` workflow runs manually and monthly on Python
3.13. It installs the `ocr` extra and exercises generated image text through the
real PaddleOCR → detector → tracker → renderer → verifier boundaries. Network or
model-host availability therefore cannot make an ordinary pull request flaky.

## Local release check

```bash
uv lock --check
uv sync --python 3.13 --extra dev
uv run ruff check .
uv run pytest \
  --cov=redact_video \
  --cov-report=term-missing \
  --cov-fail-under=85
uv build
docker build -t video-redactor:release .
```

Inspect both artifacts under `dist/` and install the wheel into a clean virtual
environment before assigning a release tag. Version changes remain an explicit
release-owner decision; Phase 10 does not silently change the project version or
publish an artifact.

## Synthetic benchmark

Run the default release scenarios:

```bash
uv run python -m benchmarks.synthetic \
  --output synthetic-benchmark.json \
  --max-realtime-factor 20
```

The defaults cover static detection, partial PAN typing, fast scrolling, and an
intentional verification repair. Use `--all` for every generated scenario or
repeat `--scenario`. Source and redacted videos exist only in a temporary
directory and are deleted when the run ends.

The JSON contains OS/Python labels, frame counts, elapsed time, throughput,
realtime factor, verification passes, repairs, golden coverage, missed pixels,
over-mask ratio, and masked-visible pixels. It contains no media paths, frame
content, masks, OCR text, or recognized values. Privacy always gates success;
the realtime threshold is optional because hardware varies.

On the Phase 10 development machine (Apple arm64, Python 3.14), the four default
scenarios completed at 18.32–35.93 fps and 0.28–0.55× realtime, with 100%
sensitive-pixel coverage, zero missed sensitive pixels, and zero masked expected-
visible pixels. These numbers are an observation, not a portable performance
guarantee.

## Private real-recording acceptance

Use only controlled, non-production recordings with synthetic card values. Keep
the input and output directories outside the repository:

```bash
uv sync --python 3.13 --extra ocr
uv run redact-video-acceptance \
  /private/non-production-inputs \
  /private/reviewed-outputs \
  --config config.example.yaml \
  --mode safe \
  --report /private/reviewed-outputs/acceptance.json
```

The runner processes regular, non-symlink MP4/MOV files sequentially. Inputs are
sorted internally but represented only as `sample_001`, `sample_002`, and so on.
Outputs use those anonymous identifiers. The report records sanitized pipeline
statistics, timing, `COMPLETED`/`NEEDS_REVIEW`/`FAILED`, and fixed error codes.
It never records input filenames or paths.

An acceptance release candidate requires:

- every sample reports `COMPLETED`
- verification has zero remaining leaks
- every output opens and preserves expected resolution, orientation, duration,
  and audio
- manual visual review confirms card details are unreadable throughout motion
- names, addresses, brands, and other expected-visible regions remain visible
- the aggregate report has `acceptance_passed: true`

On the August 2026 development machine (Apple arm64, Python 3.13), the optimized
default cascade processed a controlled 30.259-second, 1260×2736, approximately
60 fps portrait recording in 251.19 seconds (4 minutes 11 seconds). It processed
all 1,815 frames, preserved audio, completed after two verification passes and
four bounded repair windows, and reported zero remaining leaks. The preceding
production-safe implementation took 372.10 seconds on the same recording. These
figures are a local acceptance observation, not a portable performance
guarantee; privacy verification remains the release gate.

A subsequent instrumented replay completed in 265.553 seconds under different
machine load. Its sanitized stage timings were 130.116 seconds for source
analysis, 32.909 seconds for initial rendering, 47.360 and 43.862 seconds for
the two verification passes, and 7.985 seconds for repair encoding. Source
analysis plus verification therefore represented 83.3% of total runtime. This
stage breakdown is the baseline for bounded OCR parallelism; aggregate wall time
alone is not used to attribute future improvements.

The bounded authoritative-audit slice reduced the same two verification passes
from 91.222 to 74.286 seconds (18.6%) while retaining the same 1,815 processed
frames, four repair windows, and zero remaining leaks. Source analysis varied
from 130.116 to 147.878 seconds between those runs, so aggregate runtime did not
improve; the stage-level result, plus an isolated clean-pass reduction from
50.435 to 35.192 seconds under the same cold setup, is the acceptance evidence
for this optimization.

Source-analysis profiling measured 15,250 per-region optical-flow estimates at
87.726 seconds, 49.2% of the instrumented source pass. The accepted tracker
slice reduced the maximum feature budget from 80 to 40 while keeping motion
estimation on every active frame and retaining the original four-level pyramid.
The same controlled recording then completed in 216.318 seconds (3 minutes 36
seconds): source analysis took 117.165 seconds, initial rendering 26.054
seconds, the two verification passes 28.359 and 34.160 seconds, and repair
encoding 7.634 seconds. All 1,815 frames were processed, audio was preserved,
three repair windows were applied, and verification reported zero remaining
leaks. This is 49.847 seconds (18.7%) faster than the preceding 266.165-second
accepted replay.

An alternative that estimated flow at a 30 ms cadence was explicitly rejected:
although it reduced source time, it required an additional repair and a third
verification pass, increasing total runtime to 317.588 seconds. That candidate
was removed rather than exposed as a production default.

The follow-up tracker slice keeps every-frame flow but overlaps independent
high-resolution region estimates. OpenCV uses one process-wide inner thread
when a bounded four-worker outer pool is enabled above one megapixel; this
avoids nested oversubscription and leaves smaller frames sequential. An isolated
eight-region frame improved from 54.73 to 25.22 ms (2.17×). A real replay under
heavy concurrent VM and browser load completed in 300.153 seconds (5 minutes
0.153 seconds), retained the accepted baseline's 1,812 masked frames, 13,460
regions, three repair windows, and two verification passes, and reported zero
remaining leaks. Its 153.202-second source stage is not compared directly with
the quiet 117.165-second baseline because the host load was materially
different. Initial rendering took 41.450 seconds, verification took 90.829
seconds, and repair encoding took 9.262 seconds under the same contention.

A full-width changed-band OCR prototype was rejected and removed. It took
327.079 seconds, required eight repairs and all three verification passes, and
still reported two residual expiry findings. Crop-level OCR therefore is not an
accepted optimization boundary without a context-preserving detector redesign.

The next verification slice carries sanitized clean-frame guards only into the
immediately following repair pass. Frame index, timestamp, shape, and the strict
codec-noise guard must all match; prior leak frames cannot be reused. A real
replay under the same class of heavy VM/browser contention retained the exact
accepted privacy counts—1,812 masked frames, 13,460 regions, three repairs, two
passes, preserved audio, and zero remaining leaks. Its second verification pass
completed in 39.097 seconds versus 45.816 seconds in the preceding exact-code
contended replay, a 6.719-second (14.7%) stage reduction. Total runtime was
316.658 seconds because this run's initial render rose from 41.450 to 62.091
seconds; aggregate wall time is therefore not attributed to the verification
change. The repaired output remained 1260×2736 H.264 with AAC audio and a
30.244-second probed duration. Full-resolution decoding remains the dominant
cost inside the optimized second pass.

Selective cross-pass decoding removes that raw-frame bottleneck. FFmpeg first
emits a 44×96 grayscale guard stream for all 1,815 frames, then a `select`
filter emits full-resolution frames only when clean evidence does not match.
An isolated production-media probe decoded the complete compact stream in
1.755 seconds and returned five requested absolute full-resolution frames in
1.318 seconds. The controlled real replay completed in 241.109 seconds (4
minutes 1.109 seconds) despite concurrent VM/browser load. Its stage timings
were 107.504 seconds for source analysis, 25.018 seconds for initial rendering,
87.916 seconds for verification pass 1, 9.969 seconds for verification pass 2,
and 7.401 seconds for repair encoding. Pass 2 was therefore 29.128 seconds
(74.5%) faster than the preceding guarded full-decode result of 39.097 seconds.
The output retained 1,812 masked frames, 13,460 regions, three repair windows,
two passes, AAC audio, and zero remaining leaks. It remained 1260×2736 H.264
with a 30.244-second probed duration.

Regression testing during this slice exposed a latent bounded-prefetch race:
the decoder could finish before a later authoritative audit was scheduled,
leaving its future pending forever on short media. A pending audit that has not
started is now cancelled and scanned synchronously. This preserves the
authoritative decision while making completion deterministic.

A follow-up first-pass compact-planning prototype was rejected and removed. It
selected 144 of 1,815 frames for full-resolution OCR and completed in 231.727
seconds, but it required three verification passes and four repairs, increased
the mask count from the accepted 13,460 to 13,761 regions, and manual comparison
found unrelated status-bar and date-picker pixels masked across frames 401–453.
Zero reported leaks was therefore not treated as sufficient acceptance evidence.
The experiment also exposed an independent decoder defect: its 139-term FFmpeg
`select` expression exhausted the evaluator before frame zero. The retained fix
splits noncontiguous selections into ordered groups of at most 48 expression
terms and fails closed if any requested frame is omitted. A 144-frame
noncontiguous production-media probe returned every requested absolute index in
order in 3.449 seconds. Only this decoder hardening—not the behavior-changing
first-pass planner—ships from the slice.

The next source-analysis profile completed in 97.117 seconds and identified
three independent remaining cost groups: ordered decode reads, per-region
optical flow, and OCR/detection. A bounded producer-thread decode prototype was
removed after an exact-count replay took 99.478 seconds; decoding and the
CPU-heavy analysis competed rather than producing useful overlap.

The accepted detector slice builds the adjacent-token OCR span lattice once per
scanned frame and shares its immutable, token-count-filtered views across
payment context, PAN, expiry, CVV, and saved-card detectors. It also applies
SequenceMatcher's guaranteed upper bounds before the full fuzzy ratio. A
2,000-iteration synthetic microbenchmark reduced payment-context matching from
10.354 to 6.485 seconds (1.60×) and reduced six equivalent span builds from
1.481 seconds to one 0.265-second build. No recognized text is cached across
frames.

On the controlled recording, source analysis fell from 107.504 to 99.891
seconds, a 7.613-second (7.1%) stage reduction. The complete replay finished in
235.432 seconds (3 minutes 55.432 seconds), retained the exact accepted counts—
1,815 processed frames, 1,812 masked frames, 13,460 regions, 13 tracks by type,
three repairs, two verification passes, AAC audio, and zero leaks—and remained
1260×2736 H.264 at 30.244 seconds. A decoded comparison against the previously
approved result was pixel-identical across all 1,815 frames: infinite PSNR and
SSIM 1.000000. The 5.677-second aggregate reduction is recorded as an
observation because render and repair timings varied between runs.

Exact-static shortcuts were measured and rejected before implementation: only
one of 1,814 consecutive full-frame pairs was byte-identical, and only 359 of
12,914 active flow windows (2.8%) were exactly unchanged. The comparison cost
would consume most of that limited opportunity.

The accepted follow-up overlaps scheduled read-only source OCR with the current
frame's ordered active-track advancement, then joins the OCR future before
applying detections. It does not prefetch frames, alter scan cadence, reduce OCR
engines, skip flow checks, or reorder state mutations. An isolated candidate
run completed source analysis in 98.048 seconds versus a disabled control at
106.542 seconds under the immediately following host conditions, with exact
1,815/1,812/13,301 source counts in both runs.

The full controlled replay reduced source analysis from the immediately prior
99.891 to 98.440 seconds, a 1.451-second (1.5%) stage reduction, and completed
in 228.184 seconds (3 minutes 48.184 seconds). It retained 1,812 masked frames,
13,460 regions, the same 13 tracks by type, three repairs, two verification
passes, AAC audio, and zero leaks. The 1260×2736, 30.244-second result was
pixel-identical to the prior accepted output across all 1,815 decoded frames
(infinite PSNR, SSIM 1.000000). Total runtime improved by 7.248 seconds, but the
larger aggregate change is treated as an observation because render,
verification, and repair load also varied.

The next accepted slice removes avoidable lossless PNG compression from the
private Apple Vision handoff. Ordinary frames use BMP only when its exact
24-bit row-aligned size fits the existing 64 MiB protocol limit; otherwise the
adapter falls back to PNG. A seven-frame production-media probe reduced warm
Vision scan time from 43.929 to 28.810 ms (1.53×) with identical decoded pixels,
token counts, detection types, and geometries.

The full controlled replay completed in 205.970 seconds (3 minutes 25.970
seconds), versus 228.184 seconds immediately before this change: a 22.214-second
(9.7%) reduction. Source analysis fell from 98.440 to 80.901 seconds (17.8%).
It retained all 1,815 frames, 1,812 masked frames, 13,460 regions, the same 13
tracks by type, three repairs, two verification passes, AAC audio, and zero
remaining leaks. Every decoded output video frame was pixel-identical to the
previously accepted result. The new stage timings were 80.901 seconds for
source analysis, 27.289 seconds for initial rendering, 75.253 and 10.559 seconds
for verification, and 8.701 seconds for repair. First-pass verification is now
the largest measured optimization boundary.

The next accepted verification slice eliminates work that cannot affect a
decision: pass 1 no longer constructs a guard for an empty prior-evidence map,
and each OCR frame shares its already computed guard between the reuse reference
and clean-evidence record. The guard algorithm, thresholds, OCR cadence, and
finding order are unchanged. In an isolated clean-pass profile, guard calls
fell from 3,694 to exactly 1,815 and wall time fell from 66.174 to 47.755
seconds (27.8%).

The full controlled replay completed in 187.272 seconds (3 minutes 7.272
seconds), 18.698 seconds (9.1%) faster than the preceding run. First-pass
verification fell from 75.253 to 50.771 seconds, a 24.482-second (32.5%) stage
reduction. It retained 1,815 processed frames, 1,812 masked frames, 13,460
regions, the same 13 tracks by type, three repairs, two verification passes,
AAC audio, and zero remaining leaks. Every decoded output frame was
pixel-identical to the prior accepted result. The complete suite passed 328
tests with one opt-in local-model smoke skipped. The 1,815 required evidence
guards still consumed 33.273 seconds in the isolated profile, making compact
first-pass guard delivery the next measured verification candidate.

`NEEDS_REVIEW` and redacted outputs are retained for inspection; they are never
promoted to success. The source corpus is explicitly user-managed and is not
deleted by this runner.

## Container runtime

The release container installs dependencies as root during build, then switches
to the dedicated system user `redactor`. `/tmp/video-redactor` is created with
private ownership and mode `0700`. Operators must provide writable, least-
privilege mounts for requested CLI outputs or configure a private API job root.
