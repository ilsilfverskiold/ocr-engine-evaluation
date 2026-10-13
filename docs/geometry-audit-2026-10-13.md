# Geometry integration audit — 2026-10-13

Follow-up to the [expansion engine audit](engine-integrations.md). This records
implemented geometry corrections, not merely findings. No paid inference,
deployment, judge calls or model downloads. Published results remain unchanged
until an offline rescore and report publication.

## Confirmed defects fixed

- **Nine new block engines bypassed block handling.** Dispatch recognized only
  three legacy engine names, ignoring `geometry_granularity=block`. New block
  outputs were mislabeled `word_or_line` and lacked the coarse coverage diagnostic.
  Dispatch now uses the declared contract. Line/word fallbacks retain their labels.
  Actual boxes still enter the headline as requested; no word boxes are invented.
- **Already-normalized boxes could be divided by 1,000 again.** All 13
  geometry-capable additions now declare `geometry_coordinates=normalized_xyxy`
  after conversion. This disables the legacy scale guess. A synthetic x1=2
  previously became .002; it now stays 2 and is an invalid out-of-frame box.
- **Six new adapters clipped model coordinates at page edges.** DeepSeek-OCR-2,
  dots.mocr, Surya OCR 2, Chandra OCR 2, PaddleOCR-VL 1.6 and OpenDoc now preserve
  out-of-frame normalized extents instead of hiding those errors.
- **Finite invalid rectangles could earn credit and were not counted invalid.**
  The shared v8 scorer gives out-of-frame, reversed and zero-area rectangles no
  location credit, counts them once in the prediction denominator, retains their
  text in CER/WER and keeps the converted boxes visible in evidence.
- **MinerU classic pipeline mixed tables with lines.** Nine of its 17 saved
  geometry calls contain whole-table HTML among the line regions. The shared
  evaluation route now renders this native HTML to text without changing boxes,
  preserves each region's granularity, and labels such pages `mixed_line_block`.
  It works on existing saved calls and future captures. This does not turn tables
  into word boxes or upgrade the old MinerU checkpoint.
- **Mathpix withheld rotated boxes could become a false zero.** An unresolved
  rotated coordinate frame now produces an explicit adapter error, instead of
  passing prose to the box scorer. Mapping the provider's rotation remains an
  upstream/frame-validation task; the error is not an engine-quality verdict.
- Corrected the stale block-diagnostic docstring that claimed those engines were
  inapplicable. Their actual location scores remain displayed, with the coarse
  diagnostic separate.

## Evidence and impact

The published archive is `results/current.zip`, run `20261009T080833Z`.
Replayed all **196 applicable successful geometry results** from their saved
converted boxes, with the current reference loader and no inferred scaling.
Eight existing error results remain errors.

The invalid-coordinate correction alone changes one headline:
`mineru_pipeline / fintab_002`, **0.0488 → 0.0122**. Invalid boxes occur on four
results: Gemini `fintab_000` (27), Qwen `fintab_001` (102), MinerU pipeline
`fintab_001` (17) and `fintab_002` (10). Supporting metrics can change even when
the headline does not. This replay tests the scorer's handling of saved converted
boxes; it does not certify every original provider coordinate transform.

A second replay used all 17 original MinerU pipeline call payloads through the
corrected dispatcher. Nine are now labeled mixed line/block. Their table HTML
had severely inflated box-text error rates: `fintab_000` CER **1.7164 → 0.0311**,
`fintab_003` **1.7326 → 0.0642**, and `funsd_003` **3.1919 → 0.2832**.
These text corrections do not create word geometry: all 17 headline scores stay
at their prior values except the invalid-box change on `fintab_002` above.

The archive contains 128 scored v6 and 68 scored v7 results. Version labels alone
do not demonstrate a scoring difference. The next publication should replay
all geometry results through the same version instead of patching isolated rows.
The existing ZIP and HTML have not been overwritten with these audit results.

## Sonnet was already rerun with the pixel prompt

All **17 saved Sonnet geometry calls** are from `20261007T113310Z` and record
`normalized_from_recorded_pixel_frame`. The exact requests ask for absolute pixel
coordinates, give the sent image's width and height, request one tight box per
word and forbid reconstructing a cleaner layout. The runner divides coordinates
by those recorded dimensions. These are not the earlier generic 0–1 requests.
This confirms the request provenance; it does not claim that a prompt guarantees
accurate geometry. No further Sonnet call was made during this audit.

## Validation and remaining limits

Ten offline regression tests in `tests/test_geometry_contract.py` cover all nine
new block routes, line fallback labels, declared frames, invalid/duplicate box
penalties, preserving box text, no synthetic words, unclipped pixel/polygon
conversion, MinerU table text/granularity and Mathpix withheld-frame errors.
Run with `python -m unittest discover -s tests -v`.

Scratch replay output is outside Git in `/private/tmp/ocr-geometry-audit-20261013.json`
and `/private/tmp/ocr-mineru-geometry-audit-20261013.json`.

Before trusting new measurements, validate actual returned frame conventions
(particularly Google deskew, Mathpix rotation and Mistral page dimensions).
Some upstream parsers still discard malformed/degenerate entries; this work does
not claim complete preservation of errors already lost upstream. The native
block/line/word contract must be checked on the first approved captured outputs.
GPU fit and the other source-audit findings remain separate from these fixes.
