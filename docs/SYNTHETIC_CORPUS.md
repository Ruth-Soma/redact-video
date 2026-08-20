# Synthetic Video Corpus

Phase 7 provides a deterministic, generated-on-demand regression corpus for the
complete automatic pipeline. No binary video fixture or production payment data
is committed to the repository.

## Scenarios

The corpus generates H.264 MP4 videos for:

- static landscape and portrait card forms;
- card-number typing from the first digit through a complete synthetic PAN;
- CVV typing from one through three digits;
- slow and fast vertical scrolling;
- a form disappearing and returning;
- horizontal modal motion;
- saved-card last-four and expiry presentation;
- a payment screen visible for 500 ms;
- OTP, order, price, phone, tracking, invoice, postal, date, and room-number
  false positives; and
- an intentionally omitted CVV mask repaired by the Phase 6 loop.

Every frame records synthetic OCR tokens, sensitive rectangles, and regions that
must remain visible, such as the fake cardholder name, billing address, card
brand, and false-positive text.

## Decoded-frame OCR oracle

Each generated frame contains an eight-bit black/white marker in a reserved
corner. The marker survives H.264 encoding, decoding, privacy rendering, and
rerendering. The injected test OCR engine decodes the marker from the actual BGR
frame and returns only that frame's predefined tokens. This avoids fragile call
counts or hard-coded timestamps while keeping normal CI independent of Paddle
model downloads.

Verification tests can independently suppress already-masked synthetic value
tokens. The intentional-leak case exposes CVV evidence for exactly one output
scan, verifies that repair adds a temporal mask window, and requires the second
scan to pass.

## Golden-frame metrics

Timeline boxes are expanded using the production blur padding and rasterized
against every frame's ground truth. Tests measure:

```text
sensitive pixels
covered sensitive pixels
missed sensitive pixels
total mask pixels
over-mask pixels
masked expected-visible pixels
```

The hard acceptance rule is zero missed sensitive pixels. Expected-visible
regions must have zero intersection with the mask. Over-mask ratios are capped
per scenario, with larger allowances for typing cases because semantic anchors
intentionally protect blank fields before values appear.

Run the corpus with:

```bash
uv run pytest tests/integration/test_synthetic_corpus.py
```
