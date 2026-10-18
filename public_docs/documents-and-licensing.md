# Documents, references and licensing

The benchmark uses **93 test documents**, each as a single-page image with a
ground-truth sidecar (`<stem>_gt.txt`, plus `<stem>_gt.json` with table HTML
for the financial tables). Every image is an excerpt from a public dataset;
`documents/manifest.json` records, for every file: the source dataset ID,
split, row index, tier, doc type, and the image's SHA-256 — so the exact
corpus is verifiable and rebuildable.

Ground-truth sidecars, accepted references and the saved result bundle ship in
this repository. Source images are hosted at
[`ilsilfverskiold/ocr-benchmark`](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark).
Download images for new inference or HTML source-page previews:

```bash
python scripts/download_data.py
```

Image downloads are checked against `documents/manifest.json` and never replace
repository reference files. The committed references are authoritative for this
benchmark; several contain reviewed repairs that differ from upstream datasets.

---

## Corpus layout

| Tier | Doc types (docs each) | Total |
|---|---|---|
| easy | receipt (6), invoice (6) | 12 |
| medium | bank_statement, shipping, tax_form, medical, payslip, lease, photo_receipt (5 each) | 35 |
| hard | form (6), handwritten (6), scanned_legacy (6), financial_table (5), chart (5), newspaper (6), legal (6), report (6) | 46 |

---

## Per-dataset attribution and licenses

| Source dataset | Docs | Used for | License | GT provenance |
|---|---|---|---|---|
| [jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (ICDAR 2019 SROIE competition) | 6 | easy/receipt | **CC-BY-4.0** — attribution: ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction (SROIE), Huang et al. | Dataset line annotations (human) |
| [mychen76/invoices-and-receipts_ocr_v1](https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1) | 6 | easy/invoice | **None declared** on the dataset card | `parsed_data` field values (human-labeled) |
| [getomni-ai/ocr-benchmark](https://huggingface.co/datasets/getomni-ai/ocr-benchmark) (OmniAI OCR Benchmark) | 40 | all 7 medium types + hard/chart | **MIT** | `true_markdown_output` (human-annotated markdown) |
| [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) (FUNSD, Jaume et al. 2019) | 6 | hard/form | **Upstream FUNSD terms: non-commercial research/educational use** | Word annotations reordered by bounding box (human) |
| [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) (IAM Handwriting Database, FKI/University of Bern) | 6 | hard/handwritten (15 lines collaged per page) | Card says MIT, but the **underlying IAM database terms are registration-required, non-commercial research** — we treat IAM's original terms as governing | Line transcriptions (human) |
| [pixparse/idl-wds](https://huggingface.co/datasets/pixparse/idl-wds) (UCSF Industry Documents Library) | 6 | hard/scanned_legacy | **Custom "idl-train" license** (see the dataset card) | The library's bundled legacy OCR annotations, page-1 text only (machine-generated but page-accurate) |
| [ethanbradley/synfintabs](https://huggingface.co/datasets/ethanbradley/synfintabs) (SynFinTabs) | 5 | hard/financial_table | **MIT** | Exact synthetic cell text; `table_html` from the dataset's own `representations` |
| [nielsr/rvl_cdip_10_examples_per_class](https://huggingface.co/datasets/nielsr/rvl_cdip_10_examples_per_class) (RVL-CDIP, Harley et al.; from IIT-CDIP / UCSF tobacco-litigation documents) | 18 | hard/legal, hard/newspaper, hard/report | **RVL-CDIP terms: research use** | Source OCR references; 7 pages enter deterministic text, 11 use judged text |

### Honest statement on the restricted datasets

FUNSD, IAM, and RVL-CDIP (30 of the 93 documents) carry research-use or
non-commercial restrictions, and the invoice dataset card declares no license.
This repository redistributes small excerpts (5–18 images per dataset) solely
for reproducible research benchmarking, with full attribution above, which we
believe is consistent with the spirit of those terms — but it has not been
individually cleared with each rights holder:

- **If you are a rights holder** and want any document removed, open an issue
  or contact the maintainer; it will be removed promptly.
- **If you use this benchmark commercially**, note that the research-use
  restrictions attach to those documents, not to this repo's MIT-licensed
  code. The easy/medium tiers (getomni-ai, SROIE) plus SynFinTabs — 51 docs —
  are under permissive licenses (MIT / CC-BY-4.0).

## Structured references

The active targets for 65 pages are accepted records in
`documents/structured_references/gold/`. Each contains `true_json`, `json_schema`,
exclusions, aliases and source-image identity. They override adjacent
`<stem>_structured_gt.json` sidecars. KV applies the printed-label review;
schema uses the frozen shared field sample. See [field selection](scoring-design.md#shared-structured-references).

Table gold is separate, under `documents/table_references/gold/`. Shipping also
compares the frozen original table reference; other scored pages use accepted gold.

## Ground-truth provenance summary

The current loader classifies 63 pages as human references, 19 as model-created
and verified, and 11 as unreliable silver text. The first 82 enter deterministic
text metrics; the other 11 have separate saved model-judged estimates. These
classifications and repairs are recorded in `documents/gt_provenance.json` and
the reference loader.

Markdown structural catalogues are model-authored and cached. Tables use 42
accepted human-reviewed gold references; KV/schema share 65 accepted structured
references, with their documented field-selection rules. References can contain
errors; see [the scoring rules](scoring-design.md) for their limits.

## Selection method

Documents were taken deterministically from each source (fixed row indices or
first-N matching rows), not cherry-picked by content. The getomni medium set prefers `documentQuality == "PHOTO"` rows
for the photo_receipt type. IAM pages are collages of 15 consecutive dataset
line-images on a white canvas, so each "page" is genuine handwriting with a
known-exact transcript.

## Adding documents

Place images under `documents/<tier>/<doc_type>/`. The loader requires a matching
`<stem>_gt.txt` or `<stem>_gt.json`; when both exist their text must agree. JSON
references may include `plain_text` and `table_html`. Other legs need their own
references: Markdown catalogues, structured keys/schema, or text/box annotations.
A missing leg reference is N/A, not zero. Table structure diagnostics are custom
metrics, not the official PubTabNet TEDS implementation.

Register provenance and image hashes in `documents/manifest.json` and classify
text-reference reliability in `documents/gt_provenance.json`. Accepted gold,
label reviews and schema samples are bound to reference/image identities;
update them when those identities change. Discovery finds images automatically,
but does not generate references or approve them.

Only contribute documents whose terms permit redistribution, and include source
attribution. The repository's code license does not replace dataset terms.
