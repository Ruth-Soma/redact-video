# Detection

Phase 3 implements automatic sensitive-region detection for a single
OCR-scanned frame. Phase 4 tracks those detections, and Phase 5 converts the
resulting maskable regions into the deterministic timeline used by the renderer.

`DetectionEngine.detect` receives a display-oriented BGR frame, transient OCR
tokens, a frame index, and a timestamp. It evaluates payment context once,
builds the adjacent-token span lattice once, derives semantic anchors, runs the
configured `SensitiveDetector` plugins against that immutable per-frame span
set, and returns deterministic, deduplicated `SensitiveDetection` objects.
Narrow consumers such as split month/year detection filter the shared set by
token count; they do not rebuild it. The span set remains transient and is not
cached across frames or jobs.

## Payment context and anchors

Context scoring combines exact or fuzzy matches for payment vocabulary such as
`Card Number`, `Payment Method`, `Expiry`, `CVV`, and `Billing`. Matching is
case-insensitive, ignores non-semantic punctuation, supports adjacent OCR tokens,
and tolerates OCR errors such as `Card Nurnber` with reduced confidence. Cheap
SequenceMatcher upper bounds reject impossible fuzzy matches before the full
ratio calculation; because both bounds are guaranteed not to understate the
final ratio, accepted matches and confidence scores are unchanged.

Strong labels create an inferred value-area anchor below or beside the label.
Card-number, expiry, and security-code fields can therefore be protected before
their value is complete—or before any value is visible. `Month` and `Year`
become expiry anchors only after payment context has been established.

## Card-number evidence

Within a `Card Number` anchor, one to nineteen normalized digits are sensitive;
this covers incremental input such as `4`, `42`, `424`, and `4242`. A Luhn
failure does not cancel strong semantic evidence because OCR may have confused a
digit. Outside a semantic field, a candidate must contain 13–19 digits and pass
the local Luhn implementation.

PAN confidence combines semantic label, payment-screen score, numeric structure,
Luhn result, field geometry, neighboring payment fields, and OCR confidence.
Spaces and hyphens are accepted only during transient candidate validation.

## Expiry and security code

Expiry accepts `MM/YY`, `MM/YYYY`, slash/hyphen/space variants, and split
month/year values. It requires an expiry anchor or payment context plus proximity
to a card-number field; an ordinary date does not qualify on format alone.

CVV detection accepts one to four digits in an anchored `CVV`, `CVC`, `CSC`,
`CID`, `Security Code`, or `Card Security Code` field. The short partial forms
protect typing in progress. Arbitrary three- or four-digit values are never
classified as security codes without that semantic association.

## Saved cards

When `blur_saved_card` is enabled, the saved-card detector recognizes masked PAN
and last-four presentations such as `**** 4242`, `•••• 4242`, `xxxx 4242`, and
`ending in 4242`. It also classifies an expiry near the saved-card region. Its
boxes exclude unrelated brand labels and controls. Saved-card classifications
take precedence over overlapping ordinary card classifications.

## Configuration and output

All evidence weights, per-type thresholds, fuzzy matching, payment-context
minimum, anchor confidence, Luhn confidence floors, saved-card policy, and OCR
span length are centralized in `DetectionConfig`.

No detector stores its recognized sensitive value. The stable output contains
only type, frame/time, bounding box, confidence, and a sanitized source
identifier. Candidate text and normalized values exist only in transient helper
objects whose representations suppress the text.

Numeric correction is available only through:

```python
normalize_numeric_candidate(text, expected_field=NumericField.CARD_NUMBER)
```

This explicit gate allows constrained `O` to `0`, `I`/`l` to `1`, and `S` to
`5` corrections without altering labels or ordinary OCR output.
