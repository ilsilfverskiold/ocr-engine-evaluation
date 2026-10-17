# Text and Markdown output repair

The text leg must receive plain text from the conversion, and the Markdown leg
must receive the native Markdown view of the **same** conversion. Recognized
printed blocks must not disappear merely because a provider labels them as
page furniture or discarded. This repair changes output extraction; it does
not change OCR models, prompts, ground truth or scoring rules.

## Corrected adapters

| Engine | Previous behavior | Current behavior and evidence |
| --- | --- | --- |
| LlamaParse | Used `pages[].md` as `raw_text`, although each saved page also has `pages[].text`. | Scores `pages[].text` as text and stores `pages[].md` in `structured_output.markdown`. The page items and header/footer fields remain in the retained provider payload; `text_item_differences` records where their rendering differs from native text. No second API call. All 93 archived calls have both views. A difference is not automatically a processor omission: on receipt `sroie_001`, plain text says `Recelpt` while an item and Markdown say `Receipt`. The [provider's current response reference](https://developers.llamaindex.ai/llamaparse/parse/guides/response-format/) confirms separate text, Markdown and item views; the saved calls use an earlier client shape. |
| Docling | Sent `export_to_markdown()` to the text leg and used the export defaults, which omit recognized furniture. | From one `DocumentConverter.convert`, calls `export_to_text` and `export_to_markdown` with every `ContentLayer` and `traverse_pictures=True`. Retains the native document dictionary, then checks its text items and table cells against both exports. A nested table footnote on saved bank statements survived in the native document but was missing from both exports; the adapter records and adds it to both views. The [Docling document API](https://docling-project.github.io/docling/reference/docling_document/) documents both exports and the layer option; its [advanced options](https://docling-project.github.io/docling/usage/advanced_options/) explain that furniture is omitted by default. A real Docling-core fixture confirms that default text drops a `FURNITURE` item and explicit layers restore it. |
| PaddleOCR-VL v1 | Used the pipeline's filtered Markdown for both legs. The saved parsing blocks retained text that Markdown omitted. | Requests `markdown_ignore_labels=[]` and scores plain text assembled from **every** content-bearing parsing block, regardless of label or box. Table HTML becomes cell text only in the plain-text view. Native Markdown is stored separately, with any still-unrendered blocks recorded as supplements. The [PaddleX API](https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.html) documents the ignore option. The 93 archived calls contain 124 text-bearing blocks from default-ignored classes; the historical backfill adds each to Markdown, including repeated short page numbers whose text might occur elsewhere. One archived page, `report_005`, retained neither blocks nor text and cannot be recovered from that call. |
| PaddleOCR-VL 1.6 | Preserved only default-filtered Markdown as `raw_text` in the later-engine branch. | The later-engine implementation makes the same no-filter request and block-to-text split. A controlled live v1.6 call on `invoice_003` previously showed that setting only `markdown_ignore_labels=[]` restored `Invoice no: 26343874` to Markdown while leaving recognized blocks unchanged. This engine has no rows in the currently published old benchmark, so there is nothing historical to rescore for v1.6. |
| MinerU 2.5 pipeline | Used only `para_blocks` for text, omitted `discarded_blocks` from both views, and left literal HTML table tags in text. | Text uses every nonempty line from paragraph and discarded blocks, independent of geometry. Tables become cell text. The provider's `union_make` formatting remains intact; discarded blocks are inserted before the next paragraph where a source anchor survives, or at the end. Two paragraph lines the renderer joined incorrectly are recorded as supplements too. The provider Markdown and each supplement are retained separately. On saved receipt `sroie_001`, the visible `GOODS SOLD ARE NOT RETURNABLE` line now appears in both views. All 17 saved calls passed a line-by-line text and Markdown coverage replay. |

The Docling bank-statement footnote was checked against the hash-pinned
`bank_001.png` image: it is visibly printed just below the account-overview
table. This is an export omission, not a made-up item in the native document.
On that page, adding the retained footnote changes text coverage from 0.9685
to 0.9895 and CER from 0.1611 to 0.1208 under the unchanged scorer.

The historical backfill is reproducible with `scripts/backfill_complete_views.py`.
It reads the old `.call.json` evidence and creates new, separate source calls for
the affected text and Markdown legs. It does not edit the archived calls or
make provider requests. The backfill checks every retained Paddle block and
MinerU line for representation in both scored views, including short text.
LlamaParse's native text is left unchanged when its native items disagree;
their differences are recorded rather than silently mixed into the score.
The pre-repair `results/current.zip` remains in Git history at
`2cdb1725ba078e037c34c2e9a2be80de3b37cece` (SHA-256
`c0d1c3b3687dca19b4c6fba6567202ae4cee71f428629c03f82bee0cd58b814c`).

## Other current benchmark engines to review

| Engine | Finding | Current status |
| --- | --- | --- |
| Marker | `text_from_rendered` returns Markdown, which is currently sent to `raw_text` for the shared document converter. Some calls also save JSON blocks; the older audit found no confirmed missing four-character tokens in 51 pages with blocks, but short fields, tables and 31 calls without blocks remain unchecked. | **Flagged:** text is still scored from Markdown. Confirm an appropriate plain-text rendering or necessary lossless flattening before changing historical scores. |
| Mistral OCR 4 | The OCR API's saved page surface is `pages[].markdown`; the current runner uses it directly for `raw_text`. No independent full plain-text view is retained. | **Flagged:** Markdown is the only native conversion text in these calls. A text leg would need a lossless plain-text rendering, with original Markdown kept for Markdown scoring. |
| DeepSeek-OCR | The default grounding route requests Markdown and currently preserves it as `raw_text`; dedicated prompted Markdown/table routes also exist. Grounding markup is stripped, but the saved calls do not expose a separate plain-text view. | **Flagged:** text is sourced from a Markdown-style model reply. Any plain-text rendering must be checked against the full original reply, especially HTML tables and figure text. |
| GLM-OCR | Its plain-text leg uses a `Text Recognition:` response. A separate layout/Markdown route returns Markdown as `raw_text` for its own prompted pass. | No Markdown-as-text issue in the scored text leg; keep the two passes distinct. |
| MinerU VLM, Tesseract, Textract, PaddleOCR PP-OCRv5, Qwen3-VL, Claude and Gemini | The old audit found native blocks/text or separate text requests for these engines, without this Markdown-selection error. | No equivalent loss established by the saved-output checks. This is not a proof that every recognized token is correct. |

The flagged-engine findings come from code and saved-call checks; they are not
a claim that the providers recognized every printed word correctly.

## Rescore provenance

The corrected benchmark run uses the old saved run, the saved-call backfill,
and a fresh local Docling conversion of the exact 93 corpus images restored
from the published SHA-256 manifest. The Docling rerun is necessary because
the old calls did not retain the Docling document or plain-text export.
The tested `docling==2.118.1`, `docling-core==2.91.0`,
`llama-cloud-services==0.6.94` and `llama-cloud==0.1.46` versions are pinned
in `requirements.txt`. Provider/model version and runtime are recorded in
the new Docling calls; its score
is a **new local conversion**, not a replay of the old Docling model output.
The other three engines' repaired text/Markdown scores are replays of saved
provider results. MinerU pipeline has only 17 saved pages, so its new text and
Markdown numbers are 17-page measurements, not a 93-page ranking.

The published scored run is `20261017T065717Z`, assembled from the old
`20261013T154858Z` run, saved-call patch `20261016T205200Z`, and Docling
native-document patch `20261017T065648Z`. The Docling patch comes from local
conversion run `20261016T202358Z`. The score uses the unchanged text coverage
and Markdown-structure scorers. It made no provider or judge requests during
rescoring. The publisher SHA-256-verified the 37.2 MiB archive with 4,753
score rows, 2,950 inference records and 2,857 source calls. Seven historical
failed calls remain errors, and 110 historical judge verdicts were copied;
the changed text views were **not** re-judged.

| Engine | Old text coverage | Repaired text coverage | Old Markdown structure | Repaired Markdown structure |
| --- | ---: | ---: | ---: | ---: |
| LlamaParse | 0.8584 | 0.8814 | 0.6911 | 0.6911 |
| Docling | 0.8248 | 0.8643 | 0.6520 | 0.6869 |
| PaddleOCR-VL v1 | 0.8525 | 0.8749 | 0.6362 | 0.6831 |
| MinerU pipeline | no historical text score | 0.8599 (17 pages) | no historical Markdown score | 0.6348 (17 pages) |

The full-ranking text figures use 82 reference pages and the Markdown figures
use 92 pages. Docling's new figures include a fresh conversion, so they do not
isolate the effect of the export change. The other full-ranking figures use
the same saved provider calls as before. A higher coverage score does not
mean lower edit distance: PaddleOCR-VL v1 text CER rose from 0.2490 to 0.3137
and WER from 0.2499 to 0.3952 as the block-built text included omitted content
in a different order. Its Markdown rendered-text CER also rose from 0.2490 to
0.3102. These losses are visible in the README alongside the coverage gains.
Docling's rerun averaged 24.84 seconds per text-reference page on local CPU;
the old 15.60-second figure was replaced in the README.
