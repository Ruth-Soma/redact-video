# Video Card Redaction Engine
## V1 Engineering Specification for Codex

**Project status:** New build
**Primary purpose:** Automatic card-detail redaction from uploaded screen recordings
**Primary interface:** Headless CLI
**Secondary interface:** HTTP API
**Input:** MP4 / MOV screen recording
**Output:** Redacted MP4
**V1 sensitive-data scope:** Payment-card information only

---

# 1. Product Definition

Build a headless video-redaction engine that reproduces this workflow automatically:

```text
screen-recording.mp4
        ↓
inspect video
        ↓
locate card-related UI
        ↓
detect sensitive card values
        ↓
track those regions while the UI scrolls/moves
        ↓
blur sensitive pixels
        ↓
verify the rendered video
        ↓
automatically patch anything missed
        ↓
screen-recording.redacted.mp4
```

The system must require **no manual timestamp or coordinate input** during normal operation.

A developer should eventually be able to run:

```bash
redact-video input.mp4
```

and receive:

```text
input.redacted.mp4
```

The system should also support:

```bash
redact-video input.mov --output safe.mp4
```

---

# 2. Important Terminology

Although this project may be described informally as a "headless browser", **V1 is not actually a browser**.

Do not install:

- Chromium
- Playwright
- Puppeteer
- Selenium
- WebKit

for V1.

The input already consists of rendered video pixels, so V1 should be a:

> **Headless Video Redaction Engine**

"Headless" simply means that it can operate from:

- CLI
- server
- CI worker
- Docker container
- API
- another application

without requiring a graphical interface.

A real browser extension and live screen-redaction application are future products built on concepts from this engine.

---

# 3. V1 Scope

V1 must detect and redact:

```text
CARD_NUMBER
CARD_EXPIRY
CARD_CVV
SAVED_CARD_NUMBER
SAVED_CARD_EXPIRY
```

Examples include:

```text
Card Number
4242 4242 4242 4242
```

```text
Expiry
08 / 29
```

```text
CVV
729
```

```text
Visa
•••• •••• •••• 4242
Expires 08/29
```

For V1, do **not** automatically redact:

```text
customer name
email
phone
address
postcode
country
city
ordinary dates
prices
order IDs
OTP codes
account balances
```

unless they are explicitly determined to be payment-card information.

---

# 4. Critical Privacy Principle

The engine must prefer:

> **a slightly larger blur over a sensitive-data leak.**

A false negative is substantially worse than a small amount of over-redaction.

However, the system must still attempt to blur only the sensitive value rather than an entire form.

Good:

```text
Card Number
████ ████ ████ ████

Name
John Example

Address
123 Example Street
```

Bad:

```text
Card Number
████████████████████████████
Name
████████████████████████████
Address
████████████████████████████
```

---

# 5. High-Level Architecture

Implement the system as independent layers:

```text
                 INPUT VIDEO
                      │
                      ▼
            ┌──────────────────┐
            │ Media Inspector  │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Frame Scanner    │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ OCR Engine       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Card Detector    │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Region Tracker   │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Mask Timeline    │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Blur Renderer    │
            └────────┬─────────┘
                     │
                     ▼
            TEMP REDACTED VIDEO
                     │
                     ▼
            ┌──────────────────┐
            │ Safety Verifier  │
            └────────┬─────────┘
                     │
              Leak detected?
                /          \
              YES           NO
               │             │
               ▼             ▼
        Generate patches    OUTPUT
               │
               ▼
          Re-render
               │
               └──────► Verify again
```

The **verification loop is mandatory**.

Do not consider the job successfully completed merely because the first render finished.

---

# 6. Technology Stack

Use:

```text
Python 3.12+
OpenCV 4.x
PaddleOCR 3.x
NumPy
FFmpeg
ffprobe
Pydantic
Typer or argparse
FastAPI — API wrapper only
pytest
```

The OCR layer must be abstracted so PaddleOCR can later be replaced without modifying the card detector.

Create:

```python
class OcrEngine(Protocol):
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        ...
```

Then:

```text
OcrEngine
   │
   ├── PaddleOcrEngine     ← V1 production backend
   ├── MockOcrEngine       ← tests only
   └── FutureOnnxEngine
```

Do not make the rest of the system import PaddleOCR directly.

---

# 7. Repository Structure

Codex should create:

```text
video-redactor/
│
├── pyproject.toml
├── README.md
├── Dockerfile
├── .gitignore
├── .env.example
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SECURITY.md
│   ├── DETECTION.md
│   ├── TESTING.md
│   └── ROADMAP.md
│
├── src/
│   └── redact_video/
│       │
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       │
│       ├── media/
│       │   ├── probe.py
│       │   ├── decoder.py
│       │   ├── encoder.py
│       │   └── audio.py
│       │
│       ├── ocr/
│       │   ├── base.py
│       │   ├── paddle.py
│       │   └── normalizer.py
│       │
│       ├── detection/
│       │   ├── engine.py
│       │   ├── payment_context.py
│       │   ├── card_number.py
│       │   ├── expiry.py
│       │   ├── cvv.py
│       │   ├── saved_card.py
│       │   └── luhn.py
│       │
│       ├── tracking/
│       │   ├── tracker.py
│       │   ├── optical_flow.py
│       │   ├── template_match.py
│       │   └── scene_change.py
│       │
│       ├── redaction/
│       │   ├── mask.py
│       │   ├── renderer.py
│       │   └── timeline.py
│       │
│       ├── verification/
│       │   ├── verifier.py
│       │   └── repair.py
│       │
│       ├── pipeline/
│       │   ├── job.py
│       │   └── processor.py
│       │
│       └── api/
│           ├── app.py
│           ├── routes.py
│           └── jobs.py
│
└── tests/
    ├── unit/
    ├── integration/
    ├── fixtures/
    └── synthetic/
```

Keep business logic out of:

```text
cli.py
api/
```

Both interfaces must invoke the exact same core pipeline.

---

# 8. Core Domain Models

Define:

```python
class SensitiveType(str, Enum):
    CARD_NUMBER = "card_number"
    CARD_EXPIRY = "card_expiry"
    CARD_CVV = "card_cvv"
    SAVED_CARD_NUMBER = "saved_card_number"
    SAVED_CARD_EXPIRY = "saved_card_expiry"
```

Bounding box:

```python
class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
```

OCR token:

```python
class OcrToken(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox
```

Detection:

```python
class SensitiveDetection(BaseModel):
    type: SensitiveType
    frame_index: int
    timestamp_ms: int
    bbox: BoundingBox
    confidence: float
    source: str
```

Important:

**Never include the recognized sensitive value in this model.**

Do not create:

```python
value: str
```

Raw values may exist temporarily inside a detector for validation but must immediately be discarded.

Track:

```python
class SensitiveTrack(BaseModel):
    id: str
    type: SensitiveType
    start_frame: int
    end_frame: int | None
    confidence: float
    regions: dict[int, BoundingBox]
    state: str
```

States:

```text
DETECTED
TRACKING
STICKY
LOST
ENDED
```

---

# 9. Media Inspection

Before processing, use `ffprobe` to inspect:

```text
width
height
duration
frame rate
rotation
video codec
audio presence
audio codec
pixel format
```

Reject:

```text
corrupt files
zero-length videos
unsupported dimensions
files containing no video stream
```

Accept at minimum:

```text
.mp4
.mov
```

The output should be:

```text
H.264
MP4
yuv420p
```

unless configuration says otherwise.

Preserve:

```text
audio
orientation
aspect ratio
resolution
```

as closely as practical.

The redactor must never intentionally resize the final output.

---

# 10. OCR Strategy

Do **not** run expensive OCR blindly at 60 frames per second.

Use an adaptive scanning system.

Default:

```yaml
scan_fps: 8
safe_scan_fps: 12
reacquire_interval_ms: 500
```

At 60 fps:

```text
OCR
track
track
track
track
track
track
OCR
track
...
```

Additional OCR scans must be triggered immediately when:

```text
large scene change detected
modal appears
page abruptly scrolls
screen transitions
tracker loses confidence
new payment context appears
```

This solves the performance problem while still examining transition frames.

---

# 11. OCR Normalization

OCR frequently confuses:

```text
O → 0
o → 0
I → 1
l → 1
S → 5
```

Do **not** globally mutate OCR text.

Only apply constrained corrections when analyzing a candidate that is already expected to contain numeric card data.

For example:

```text
4242 4242 4O42 4242
```

may temporarily normalize to:

```text
4242424240424242
```

for validation.

Never store the normalized PAN.

---

# 12. Payment Context Detector

Before interpreting numbers, identify whether the surrounding screen appears related to payment-card entry.

Context terms include:

```text
card
card number
credit card
debit card
payment method
payment details
expiry
expiration
cvv
cvc
csc
security code
card security code
name on card
save card
add card
billing
```

Normalize OCR strings:

```text
lowercase
trim whitespace
collapse repeated whitespace
strip non-semantic punctuation
```

Use fuzzy comparison for OCR errors.

For example:

```text
"Card Nurnber"
```

should still be capable of matching:

```text
"Card Number"
```

with a lower confidence.

---

# 13. Card Number Detection

The detector must support both:

### Semantic field detection

Example:

```text
Card Number
4242 4242 4242 4242
```

If a strong `Card Number` label exists, numeric content located in the corresponding field should be treated as sensitive even if the user has only typed:

```text
4
42
424
4242
```

Do not wait for 16 digits.

This is extremely important.

### Pattern detection

Remove:

```text
spaces
hyphens
```

from candidate numeric strings.

Candidate PAN length:

```text
13–19 digits
```

Run Luhn validation.

However:

**Luhn failure must not automatically mean safe.**

OCR could have misread one digit.

Confidence should combine:

```text
payment context
label proximity
digit count
Luhn result
field geometry
neighboring expiry/CVV fields
```

An unlabeled 16-digit number should require stronger validation than a value directly beneath `Card Number`.

---

# 14. Luhn Implementation

Implement locally.

Function:

```python
def luhn_valid(number: str) -> bool:
    ...
```

Requirements:

```text
numeric string only
13–19 digits
unit tested
no dependency required
```

Test:

```text
valid synthetic PAN
invalid check digit
spaces removed correctly
hyphens removed correctly
too short
too long
```

Do not log the number.

---

# 15. Expiry Detection

Support:

```text
08/29
08 / 29
08-29
08 29
08/2029
```

Also support split fields:

```text
Month        Year
08           2029
```

Valid month:

```text
01–12
```

Expiry detection requires either:

```text
expiry/expiration context
```

or strong proximity to an already detected card-number field.

Do not automatically classify every:

```text
08/29
```

as card information.

---

# 16. CVV Detection

Recognize labels:

```text
CVV
CVC
CSC
CID
Security Code
Card Security Code
```

Card security codes are generally:

```text
3 or 4 digits
```

But:

> Never classify arbitrary three-digit strings as CVV.

Require strong semantic proximity.

Examples that must NOT be blurred merely because they contain three digits:

```text
OTP 729
Order #729
£729
729 items
Room 729
```

If the system detects:

```text
CVV
```

then the associated field region can be marked sensitive **before the entire value is visible**.

---

# 17. Semantic Field Anchors

This is an important V1 feature.

The detector should not only search for sensitive values.

It should also search for **field anchors**.

Example:

```text
Card Number
[                         ]
```

Before the user types anything, identify that field region as:

```text
SEMANTIC_CARD_NUMBER_FIELD
```

Then track it.

When digits begin appearing, the blur already knows where the sensitive area is.

Conceptually:

```text
detect "Card Number"
        ↓
infer corresponding value region
        ↓
track region
        ↓
digits appear
        ↓
region already protected
```

Do the same for:

```text
CVV
Expiry
```

This greatly reduces leaks during typing.

---

# 18. Saved Card Detection

Support:

```text
**** 4242
•••• 4242
xxxx 4242
ending in 4242
Visa •••• 4242
Mastercard **** 4242
```

If V1 configuration says:

```yaml
blur_saved_card: true
```

blur:

```text
last four digits
masked PAN
saved expiry
```

Do not blur:

```text
Visa logo
Mastercard logo
Delete button
Set default button
card nickname
```

unless future policies require it.

---

# 19. Detection Confidence

Every detection must receive a confidence score.

Example scoring concept:

```text
semantic label                +0.35
payment-screen context        +0.20
valid numeric structure       +0.15
Luhn passes                   +0.15
correct field geometry        +0.10
near expiry/CVV field         +0.05
```

Do not hard-code this exact formula throughout the code.

Centralize configuration.

Example:

```yaml
thresholds:
  card_number: 0.72
  expiry: 0.70
  cvv: 0.68
  saved_card: 0.70
```

Privacy mode should favor recall over precision.

---

# 20. Region Tracking

Once a sensitive region has been detected, **do not require OCR to find it again on every frame**.

Track it.

Primary tracking:

```text
Lucas-Kanade optical flow
```

Use feature points around the field/value region.

Estimate:

```text
delta x
delta y
optional scale
```

between frames.

For normal screen scrolling, movement will often mostly consist of translation.

Update:

```python
bbox.x += dx
bbox.y += dy
```

for each subsequent frame.

---

# 21. Tracker Fallback

If optical flow loses confidence:

```text
1. predict the next location from recent velocity
2. search a limited area around predicted position
3. use template matching
4. trigger OCR reacquisition
```

Do not immediately remove the mask.

Tracker failure must transition to:

```text
STICKY
```

not:

```text
UNREDACTED
```

---

# 22. Sticky Blur

This is a mandatory privacy feature.

Consider:

```text
Frame 100  detection 99%
Frame 101  detection 98%
Frame 102  detection 97%
Frame 103  scroll blur / detection 32%
Frame 104  detection 95%
```

Wrong:

```text
100 blur
101 blur
102 blur
103 visible
104 blur
```

Correct:

```text
100 blur
101 blur
102 blur
103 blur
104 blur
```

Default:

```yaml
sticky_duration_ms: 350
```

A sensitive region must remain redacted for at least that period after confidence is lost unless the system positively determines the element has left the screen or been replaced.

The sticky region should continue moving using predicted velocity when possible.

---

# 23. Safety Padding

Never mask exactly the OCR bounding rectangle.

Add a privacy margin.

Example:

```text
Detected:
x=50
y=454
w=470
h=42
```

Rendered mask:

```text
x=43
y=447
w=484
h=56
```

Calculate padding relative to video resolution.

Default:

```python
padding = max(
    6,
    round(min(frame_width, frame_height) * 0.004)
)
```

Allow separate:

```text
x padding
y padding
```

in configuration.

---

# 24. Scene Change Detection

Run a cheap visual-change detector on all frames or heavily downscaled frames.

Detect:

```text
major page transition
modal opening
modal closing
large scroll jump
navigation
screen replacement
```

When a major change occurs:

```text
trigger OCR immediately
invalidate weak tracks
retain sticky masks briefly
search for new payment fields
```

Do not rely solely on fixed OCR intervals.

---

# 25. Mask Timeline

All detection/tracking output must eventually become a deterministic timeline:

```python
MaskTimeline
```

Example conceptual representation:

```json
{
  "1780": [
    {
      "type": "card_cvv",
      "x": 49,
      "y": 1184,
      "width": 108,
      "height": 49
    }
  ],
  "1781": [
    {
      "type": "card_cvv",
      "x": 49,
      "y": 1161,
      "width": 108,
      "height": 49
    }
  ]
}
```

Do not save this JSON permanently by default.

It may be kept in memory or a sanitized temporary job file containing regions only.

It must never contain card values.

---

# 26. Blur Rendering

V1 should use a **privacy blur**, not a cosmetic blur.

Recommended algorithm:

```text
crop sensitive ROI
      ↓
downscale aggressively
      ↓
upscale using nearest-neighbor
      ↓
strong Gaussian blur
      ↓
replace original ROI
```

Conceptually:

```python
roi = frame[y1:y2, x1:x2]

small = cv2.resize(
    roi,
    small_size,
    interpolation=cv2.INTER_AREA,
)

pixelated = cv2.resize(
    small,
    original_size,
    interpolation=cv2.INTER_NEAREST,
)

safe = cv2.GaussianBlur(
    pixelated,
    (0, 0),
    sigmaX=8,
)

frame[y1:y2, x1:x2] = safe
```

The result must make individual digits impossible to read.

Allow future modes:

```text
privacy_blur
pixelate
solid
```

V1 default:

```text
privacy_blur
```

---

# 27. Video Encoding

Rendering pipeline:

```text
original video
      ↓
decode frames
      ↓
apply MaskTimeline
      ↓
encode processed video
      ↓
mux original audio
      ↓
MP4
```

Use FFmpeg for final encoding/muxing.

Default:

```text
H.264
CRF 18
yuv420p
faststart
```

Preserve the original audio where possible.

If the source has no audio:

```text
do not generate artificial audio.
```

---

# 28. Variable Frame Rate

Screen recordings may not use a perfectly constant frame rate.

For V1:

1. Inspect frame-rate metadata.
2. If the source behaves as CFR or near-CFR, process normally.
3. If clearly VFR, normalize internally to a deterministic CFR representation.
4. Preserve duration.
5. Preserve audio synchronization.
6. Emit a warning in the report.

Do not silently create major A/V drift.

Add an integration test verifying:

```text
audio/video duration difference < 100 ms
```

for supported test files.

---

# 29. The Most Important Feature: Verification Pass

After the first redacted video is rendered:

**scan the output again.**

The verifier must ask:

```text
Can I still detect a card number?
Can I still detect CVV digits beside a CVV label?
Can I still detect expiry beside an expiry label?
Can I still detect saved-card details?
```

The verifier uses the same detector but with slightly more privacy-sensitive thresholds.

Pipeline:

```text
Render #1
   ↓
Verification scan
   ↓
No leak?
   ├── YES → complete
   │
   └── NO
         ↓
     create patch masks
         ↓
     Render #2
         ↓
     verify again
```

This is the automated equivalent of a human saying:

```text
"Check 00:18."
"Check 00:43."
"Check 00:49."
```

---

# 30. Automatic Repair

If verification detects:

```text
CVV leak at 00:43.150
```

do not patch exactly one frame.

Create a repair window.

Default:

```yaml
repair_pre_roll_ms: 350
repair_post_roll_ms: 500
```

Example:

```text
detected leak:
43.150

repair:
42.800 → 43.650
```

Re-run full detection/tracking inside that repair window with higher OCR frequency.

Apply masks.

Re-render only the affected segment where practical.

Then verify again.

---

# 31. Maximum Verification Cycles

Default:

```yaml
max_verification_passes: 3
```

If leaks are still detected after the final pass:

```text
DO NOT report success.
```

Return:

```text
needs_review
```

Example CLI:

```text
⚠ Redaction completed but verification could not guarantee all detected
card details were obscured.

Status: NEEDS_REVIEW
Potential issue: 00:47.220–00:47.510
```

Never print the sensitive value.

---

# 32. Success States

Jobs have:

```text
QUEUED
INSPECTING
DETECTING
TRACKING
RENDERING
VERIFYING
PATCHING
COMPLETED
NEEDS_REVIEW
FAILED
```

`COMPLETED` means:

```text
render succeeded
AND
verification passed
```

Not merely:

```text
FFmpeg exited successfully
```

---

# 33. CLI

Implement:

```bash
redact-video input.mp4
```

Optional output:

```bash
redact-video input.mp4 --output safe.mp4
```

Safe mode:

```bash
redact-video input.mp4 --safe
```

Fast development mode:

```bash
redact-video input.mp4 --fast
```

Verbose progress:

```bash
redact-video input.mp4 --verbose
```

Example UX:

```text
Video Redactor

Input:
ScreenRecording.mp4

Duration: 01:42
Resolution: 1260×2736

Inspecting video...
Scanning for payment-card information...

Card-number regions detected: 3
Expiry regions detected: 4
CVV regions detected: 3
Saved-card regions detected: 1

Tracking sensitive regions...
Rendering privacy masks...

Verification pass 1/3...
2 possible leaks detected.

Repairing 00:42.80–00:43.65
Repairing 01:24.70–01:25.60

Verification pass 2/3...
No readable card details detected.

✓ Complete

Output:
ScreenRecording.redacted.mp4
```

Do not display recognized card values.

---

# 34. CLI Exit Codes

Use predictable codes:

```text
0   completed and verified
1   processing failure
2   unsupported input
3   needs manual review
4   invalid configuration
```

This makes the tool usable in CI and other software.

---

# 35. Optional HTTP API

The API must be a thin layer over the core engine.

Endpoint:

```http
POST /v1/redactions
```

Multipart:

```text
file = video
```

Optional:

```json
{
  "policy": "card_details",
  "mode": "safe"
}
```

Response:

```json
{
  "job_id": "red_01J...",
  "status": "queued"
}
```

Status:

```http
GET /v1/redactions/{job_id}
```

Response:

```json
{
  "job_id": "red_01J...",
  "status": "verifying",
  "progress": 87
}
```

Download:

```http
GET /v1/redactions/{job_id}/output
```

Report:

```http
GET /v1/redactions/{job_id}/report
```

---

# 36. API Job Processing

Do not place core video processing directly inside the request handler.

Implement:

```text
HTTP
 ↓
create job
 ↓
bounded worker
 ↓
core pipeline
 ↓
result
```

For V1, a local process pool is enough.

Design worker abstraction so Redis/Celery/etc. could be added later.

Do not make Redis mandatory for local development.

---

# 37. Privacy and Logging

This is a critical security requirement.

Never log:

```text
card number
CVV
expiry value
OCR text containing complete PANs
raw payment screenshots
unredacted frame crops
```

Logs may contain:

```text
CARD_NUMBER detected
frame=1210
confidence=0.94
bbox=(48,454,492,51)
```

They must NOT contain:

```text
CARD_NUMBER detected: 4242424242424242
```

---

# 38. Temporary Data

Default job directory:

```text
/tmp/video-redactor/<job-id>/
```

Temporary contents may include:

```text
encoded segment
mask metadata
render intermediates
```

Do not save raw OCR transcripts.

Delete temporary artifacts:

```text
after successful completion
after failure
after configured timeout
```

The original input and final output can remain if explicitly requested.

---

# 39. Local-Only OCR

V1 must perform OCR locally.

Do not send frames or recognized card information to:

```text
OpenAI
Google Vision
AWS Textract
Azure OCR
any remote AI service
```

by default.

A future remote detector can exist behind an explicitly enabled provider interface.

It must never become an implicit fallback.

---

# 40. Configuration

Create:

```yaml
redaction:
  policy: card_details

  blur:
    mode: privacy_blur
    gaussian_sigma: 8
    min_padding_px: 6
    padding_ratio: 0.004

detection:
  scan_fps: 8
  safe_scan_fps: 12

  thresholds:
    card_number: 0.72
    expiry: 0.70
    cvv: 0.68
    saved_card: 0.70

tracking:
  sticky_duration_ms: 350
  reacquire_interval_ms: 500

verification:
  enabled: true
  max_passes: 3
  repair_pre_roll_ms: 350
  repair_post_roll_ms: 500

output:
  codec: h264
  crf: 18
  preserve_audio: true
```

CLI parameters may override configuration.

---

# 41. Report

Generate a sanitized report.

Example:

```json
{
  "status": "completed",
  "input_duration_ms": 102441,
  "output_duration_ms": 102440,
  "frames_processed": 6145,

  "detections": {
    "card_number": 3,
    "card_expiry": 4,
    "card_cvv": 3,
    "saved_card_number": 1,
    "saved_card_expiry": 1
  },

  "verification": {
    "passes": 2,
    "repairs": 2,
    "remaining_leaks": 0
  }
}
```

Never include sensitive values.

---

# 42. Synthetic Test Video Generator

Build a test-video generator as part of the repository.

It should create fake phone/browser screens containing synthetic card fields.

Generate scenarios such as:

```text
static form
typing card number
typing CVV
vertical scroll
fast scroll
slow scroll
modal slide
screen transition
saved card
payment card appearing twice
card details appearing for 300 ms
portrait recording
landscape recording
```

Use synthetic, non-production test values.

Tests must not depend on manually created secret files.

---

# 43. False-Positive Tests

The following should remain unblurred:

```text
OTP: 729
Order 424242
£729.00
Phone: +44...
Tracking number
Invoice ID
Postal code
Date: 08/29
Room 729
2029
```

unless sufficient payment-card context exists.

---

# 44. Required Integration Tests

Create at least these cases:

### Case 1
Static card form.

Expected:

```text
PAN blurred
expiry blurred
CVV blurred
name/address visible
```

### Case 2
Card number typed character-by-character.

Expected:

```text
no readable partial PAN once field is identified
```

### Case 3
CVV typed character-by-character.

Expected:

```text
CVV region protected during entry
```

### Case 4
Slow vertical scroll.

Expected:

```text
blur follows sensitive values
```

### Case 5
Fast vertical scroll.

Expected:

```text
sticky mask prevents frame leakage
```

### Case 6
Card form disappears then returns.

Expected:

```text
fields rediscovered and redacted again
```

### Case 7
Saved card.

Expected:

```text
last-four digits blurred
expiry blurred
card brand visible
```

### Case 8
Payment screen appears for 0.5 seconds.

Expected:

```text
transition-triggered scan finds it
```

### Case 9
False positives.

Expected:

```text
OTP/order numbers remain visible
```

### Case 10
Verification repair.

Intentionally disable one mask during test render.

Expected:

```text
verifier detects leak
repair pipeline adds mask
second render passes
```

This test is essential.

---

# 45. Golden-Frame Testing

For synthetic videos, maintain ground-truth rectangles.

Compare generated masks against ground truth.

Measure:

```text
coverage
over-mask area
missed sensitive pixels
```

Primary goal:

```text
zero uncovered sensitive-value regions
```

False-positive minimization is secondary.

---

# 46. Acceptance Criteria for V1

V1 is not complete until all of the following work:

```text
✓ MP4 input
✓ MOV input

✓ card number detection
✓ partial card-number protection
✓ expiry detection
✓ split month/year detection
✓ CVV detection
✓ saved-card detection

✓ vertical tracking
✓ fast-scroll tracking
✓ sticky masks
✓ scene-change reacquisition

✓ strong privacy blur

✓ audio preservation

✓ sanitized logs

✓ CLI

✓ verification pass
✓ automatic repair pass
✓ NEEDS_REVIEW state

✓ unit tests
✓ integration tests
✓ Docker build
✓ README
```

---

# 47. Explicit Non-Goals for V1

Do not spend time building:

```text
desktop GUI
mobile app
browser extension
Chrome automation
Playwright
Puppeteer
real-time screen recording
face detection
email redaction
phone redaction
address redaction
cloud storage
user accounts
billing
authentication
multi-tenant database
dashboard
```

Build the redaction engine first.

---

# 48. Future Architecture

The current enum:

```python
SensitiveType.CARD_NUMBER
SensitiveType.CARD_EXPIRY
SensitiveType.CARD_CVV
```

should later accept:

```python
SensitiveType.EMAIL
SensitiveType.PHONE
SensitiveType.ADDRESS
SensitiveType.PASSPORT
SensitiveType.BANK_ACCOUNT
SensitiveType.IBAN
SensitiveType.API_KEY
SensitiveType.PASSWORD
SensitiveType.FACE
SensitiveType.LICENSE_PLATE
```

Therefore detectors must use a plugin-style architecture.

Example:

```python
class SensitiveDetector(Protocol):

    @property
    def types(self) -> set[SensitiveType]:
        ...

    def detect(
        self,
        frame: np.ndarray,
        ocr: list[OcrToken],
        context: DetectionContext,
    ) -> list[SensitiveDetection]:
        ...
```

Then:

```text
DetectionEngine
   │
   ├── CardDetector
   ├── FutureEmailDetector
   ├── FuturePhoneDetector
   └── FutureAddressDetector
```

---

# 49. Future Live Recorder

Do not implement now, but preserve the architecture.

Future:

```text
screen capture
      ↓
frame
      ↓
same SensitiveDetector
      ↓
same RegionTracker
      ↓
same BlurRenderer
      ↓
video encoder
```

The difference is only:

```text
V1 input = stored video frames

V2 input = live screen frames
```

The detector/tracker should therefore not depend on a file path.

It should operate on frames.

---

# 50. Future Browser Extension

The browser extension will use DOM information instead of OCR.

Conceptually:

```text
DOM card detector
        ↓
SensitiveDetection
        ↓
same SensitiveType model
        ↓
blur
```

The core domain model should therefore remain independent of OCR.

---

# 51. Implementation Order for Codex

Codex must implement in this order.

## Phase 1 — Media pipeline

Build:

```text
ffprobe inspection
frame decoder
frame encoder
audio muxing
CLI
```

Create a temporary manual-region feature:

```bash
redact-video input.mp4 \
  --manual-box 50,450,500,50
```

Confirm video rendering/audio works before introducing OCR.

## Phase 2 — OCR

Build:

```text
OcrEngine interface
PaddleOCR adapter
OCR normalization
OCR unit tests
```

## Phase 3 — Card detectors

Implement:

```text
payment context
card number
Luhn
expiry
CVV
saved cards
```

No tracking yet.

## Phase 4 — Tracking

Implement:

```text
optical flow
template fallback
velocity prediction
sticky masks
scene changes
OCR reacquisition
```

## Phase 5 — Automated rendering

Create:

```text
DetectionTimeline
→
MaskTimeline
→
Renderer
```

## Phase 6 — Verification

Implement:

```text
output rescan
leak detection
repair windows
automatic rerender
max verification cycles
NEEDS_REVIEW
```

## Phase 7 — Test suite

Build synthetic videos and all required integration cases.

## Phase 8 — HTTP service

Only after CLI passes all tests.

---

# 52. Codex Working Rules

Codex must follow these constraints:

1. Do not implement all modules as one giant Python file.
2. Do not put detector logic inside CLI code.
3. Do not log recognized sensitive values.
4. Do not persist OCR transcripts.
5. Do not use cloud OCR.
6. Do not install Chromium.
7. Do not skip automatic verification.
8. Do not call a render successful before verifier passes.
9. Do not remove a mask immediately because tracking confidence drops.
10. Do not blur an entire form when only a value region needs protection.
11. Do not hard-code coordinates for individual websites.
12. Do not hard-code timestamps from test videos.
13. Do not require manual corrections for the happy path.
14. Do not silently swallow FFmpeg errors.
15. Do not leave placeholder TODO implementations in production paths.
16. Run tests after every major phase.

---

# 53. Master Prompt for Codex

Use the following instruction when starting the repository:

---

**Build a production-quality Python project called `video-redactor` according to the engineering specification in this repository.**

The purpose of V1 is to accept an MP4 or MOV screen recording and automatically redact payment-card information from the resulting video.

The protected data types for V1 are:

- full or partial card numbers when they are located in a card-number field
- card expiry month/year
- CVV/CVC/CSC/CID
- saved-card last-four/masked PAN displays
- saved-card expiry

Do not redact unrelated names, addresses, phone numbers, prices, order IDs, OTP codes, ordinary dates, or other information unless it is confidently associated with payment-card context.

The implementation must be headless. Do not use Chromium, Playwright, Puppeteer, Selenium, or a GUI.

Use a modular architecture consisting of:

```text
media ingestion
OCR abstraction
payment-context detector
card detectors
temporal tracker
sticky-mask system
scene-change detector
mask timeline
privacy-blur renderer
verification scanner
automatic repair loop
CLI
optional HTTP API
```

Use Python 3.12+, OpenCV, local PaddleOCR, NumPy, FFmpeg/ffprobe, Pydantic, pytest, and a clean CLI framework.

The core engine must remain independent of CLI and HTTP layers.

Implement a local `OcrEngine` protocol and a PaddleOCR-backed implementation. Never send video frames to external/cloud OCR services.

Never log, persist, include in exceptions, or include in API responses the actual card number, CVV, or recognized sensitive values.

For card-number detection, use semantic payment context, OCR geometry, numeric pattern recognition, and Luhn validation. Do not require Luhn success when strong semantic card-field context exists because OCR may misrecognize digits.

For CVV detection, require CVV/CVC/CSC/CID/security-code context. Do not classify arbitrary 3–4 digit numbers as CVVs.

For expiry detection, support MM/YY, MM/YYYY, and split month/year fields, requiring payment context.

Create semantic field anchors so that once a `Card Number`, `Expiry`, or `CVV` field has been recognized, its value area can be protected even while the user is still typing partial content.

Run OCR adaptively rather than on every video frame. Use periodic OCR plus immediate scans after scene changes or tracker confidence loss.

After a sensitive region is detected, track it frame-by-frame using optical flow. Add template matching and OCR reacquisition as fallbacks.

Implement mandatory sticky redaction. If a sensitive region is temporarily lost during scrolling, continue masking it for a configured period while predicting its movement. Never expose a region for a single frame merely because detection confidence temporarily drops.

Apply safety padding around every sensitive bounding box.

Use a strong privacy blur consisting of pixelation/downsampling followed by Gaussian blur. The result must make card values unreadable, not merely aesthetically blurred.

Preserve video resolution, orientation, approximate frame timing, duration, and audio.

Most importantly, implement an automatic verification-and-repair loop:

```text
detect
track
render
verify redacted output
if leak detected:
    generate repair window
    rescan at higher frequency
    patch mask timeline
    rerender
    verify again
```

Allow up to three verification passes by default.

Only return `COMPLETED` after the verifier finds no remaining readable card details.

If verification cannot pass after the maximum attempts, return `NEEDS_REVIEW`; never falsely return success.

Build a synthetic-video test generator and test:

- static forms
- partial card-number typing
- partial CVV typing
- expiry selection
- vertical scrolling
- rapid scrolling
- modal transitions
- card details disappearing and returning
- saved cards
- brief sensitive-data appearances
- OTP/order-number/price false positives
- verification detecting and automatically repairing an intentionally leaked region

The repository must contain:

```text
README.md
docs/ARCHITECTURE.md
docs/SECURITY.md
docs/DETECTION.md
docs/TESTING.md
docs/ROADMAP.md
Dockerfile
pyproject.toml
unit tests
integration tests
```

Begin by implementing and testing the media pipeline and CLI before introducing OCR.

Then implement OCR.

Then detectors.

Then tracking.

Then rendering.

Then verification/repair.

Then the API.

Do not jump directly to a large monolithic implementation.

At the end of every phase:

1. run the test suite,
2. fix failures,
3. update documentation,
4. show the files changed,
5. explain what was implemented,
6. identify the next phase.

The finished developer experience should be:

```bash
redact-video ScreenRecording.mp4
```

with output resembling:

```text
Inspecting video...
Scanning card fields...
Tracking 6 sensitive regions...
Rendering...
Verification pass 1...
2 possible leaks detected.
Repairing...
Verification pass 2...
No readable card information detected.

✓ Complete
Output: ScreenRecording.redacted.mp4
```

The engine must reproduce automatically the workflow of manually inspecting screen-recording videos, locating card details, following them through scrolling and transitions, blurring them precisely, checking the resulting video for missed frames, and correcting any missed regions automatically.

---

# 54. Definition of Done

Do not call V1 complete because one demo video works.

V1 is complete when:

```text
input.mp4
   ↓
one command
   ↓
automatic detection
   ↓
automatic tracking
   ↓
automatic blur
   ↓
automatic verification
   ↓
automatic repair
   ↓
verified output.mp4
```

works repeatedly across the synthetic test corpus and a representative collection of real mobile/desktop screen recordings without hard-coded timestamps or coordinates.

The central product promise for V1 is:

> **Feed it a screen recording. It finds the card details, keeps them protected while they move, checks its own work, and returns a safely redacted video.**
