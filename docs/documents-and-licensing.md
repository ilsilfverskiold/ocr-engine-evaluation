# The document corpus: provenance, attribution, and licensing

The benchmark uses **93 test documents**, each as a single-page image with a
ground-truth sidecar (`<stem>_gt.txt`, plus `<stem>_gt.json` with table HTML
for the financial tables). Every image is an excerpt from a public dataset;
`documents/manifest.json` records, for every file: the source dataset ID,
split, row index, tier, doc type, and the image's SHA-256 — so the exact
corpus is verifiable and rebuildable.

The document images and ground truth are stored on HuggingFace at
[`ilsilfverskiold/ocr-benchmark`](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark)
— one `train` split, one row per document, with the image and every GT sidecar
as columns. The reference run results ship with the git repository under
`results/`. Rebuild the local corpus with:

```bash
python scripts/download_data.py
```

This recreates the exact on-disk layout the harness expects
(`documents/<tier>/<doc_type>/<stem>.png` plus GT sidecars), byte-identical to
the corpus the reference run was measured on.

**The HuggingFace copies are authoritative.** Several GT files were repaired or
hand-labelled after download and cannot be reproduced by re-downloading from
the original source datasets.

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
| [jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (ICDAR 2019 SROIE competition) | 6 | easy/receipt | **CC-BY-4.0** — attribution: ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction (SROIE), Huang et al. | Dataset word annotations (human) |
| [mychen76/invoices-and-receipts_ocr_v1](https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1) | 6 | easy/invoice | **None declared** on the dataset card | `parsed_data` field values (human-labeled) |
| [getomni-ai/ocr-benchmark](https://huggingface.co/datasets/getomni-ai/ocr-benchmark) (OmniAI OCR Benchmark) | 40 | all 7 medium types + hard/chart | **MIT** | `true_markdown_output` (human-annotated markdown) |
| [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) (FUNSD, Jaume et al. 2019) | 6 | hard/form | **Upstream FUNSD terms: non-commercial research/educational use** | Word annotations reordered by bounding box (human) |
| [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) (IAM Handwriting Database, FKI/University of Bern) | 6 | hard/handwritten (15 lines collaged per page) | Card says MIT, but the **underlying IAM database terms are registration-required, non-commercial research** — we treat IAM's original terms as governing | Line transcriptions (human) |
| [pixparse/idl-wds](https://huggingface.co/datasets/pixparse/idl-wds) (UCSF Industry Documents Library) | 6 | hard/scanned_legacy | **Custom "idl-train" license** (see the dataset card) | The library's bundled legacy OCR annotations, page-1 text only (machine-generated but page-accurate) |
| [ethanbradley/synfintabs](https://huggingface.co/datasets/ethanbradley/synfintabs) (SynFinTabs) | 5 | hard/financial_table | **MIT** | Exact synthetic cell text; `table_html` from the dataset's own `representations` |
| [nielsr/rvl_cdip_10_examples_per_class](https://huggingface.co/datasets/nielsr/rvl_cdip_10_examples_per_class) (RVL-CDIP, Harley et al.; from IIT-CDIP / UCSF tobacco-litigation documents) | 18 | hard/legal, hard/newspaper, hard/report | **RVL-CDIP terms: research use** | **Silver** — Tesseract run at corpus-build time; judge-only evaluation |

### Honest statement on the restricted datasets

FUNSD, IAM, and RVL-CDIP (30 of the 93 documents) carry research-use or
non-commercial restrictions, and two more sources declare no license at all.
This repository redistributes small excerpts (5–18 images per dataset) solely
for reproducible research benchmarking, with full attribution above, which we
believe is consistent with the spirit of those terms — but it has not been
individually cleared with each rights holder:

- **If you are a rights holder** and want any document removed, open an issue
  or contact the maintainer; it will be removed promptly.
- **If you use this benchmark commercially**, note that the research-use
  restrictions attach to those documents, not to this repo's MIT-licensed
  code. The easy/medium tiers (getomni-ai, SROIE) plus SynFinTabs — 57 docs —
  are under permissive licenses (MIT / CC-BY-4.0).

## Structured-extraction ground truth (JSON sidecars)

65 documents additionally carry a `<stem>_structured_gt.json` sidecar for the
schema and key_value legs:

| Source | Docs | What the sidecar contains |
|---|---|---|
| getomni-ai/ocr-benchmark | 40 | The dataset's own human-annotated `json_schema` + `true_json_output` per document |
| mychen76 invoices | 6 | `parsed_data` JSON: header fields, line items, summary totals |
| SROIE receipts | 6 | The SROIE task-2 entities: company, date, address, total |
| FUNSD forms | 6 | **Hand-labeled** key-value pairs (this repo's own annotation — the HF version has no key→value linking) |
| SynFinTabs financial tables | 5 | Column/row records from the dataset's own `representations` |
| Scanned-legacy memos | 2 | **Hand-labeled** memo header fields (TO/FROM/DATE/SUBJECT/CC) |

Format: `{dataset, split, manifest_index, dataset_offset, doc_type,
json_schema, true_json}` (hand-labeled files use `"dataset": "hand-labeled"`
and carry a `note` explaining document-specific judgment calls). The FUNSD and scanned-legacy sidecars are hand-labelled and are not
regenerable; the handwriting and RVL-CDIP silver docs have nothing
schema-shaped to extract.

## Ground-truth provenance summary

- **Human GT (75 docs):** SROIE words, invoice page text (`raw_data.ocr_words`),
  getomni human-annotated markdown, FUNSD annotations, IAM transcripts,
  SynFinTabs exact cell text, and all 6 scanned-legacy docs (page-1 text).
- **Silver GT (18 docs):** the RVL-CDIP legal/newspaper/report docs, whose GT
  was generated by Tesseract at corpus-build time and varies down to genuinely
  garbled. Excluded from text metrics; the document judge estimates them instead.
  Defined once in [`evaluation/ground_truth.py`](../evaluation/ground_truth.py).
- **No GT anywhere is LLM-generated.**

Two GT defects that used to be listed here — invoice GT holding values without
field labels, and scanned-legacy GT spanning pages the engine never saw — have
been repaired at the source; `scanned_001–004` are consequently no longer
silver. Known remaining imperfections include IAM tokenisation and the SROIE
address annotation.

## Selection method

Documents were taken deterministically from each source (fixed row indices or
first-N matching rows), not cherry-picked by content. The getomni medium set prefers `documentQuality == "PHOTO"` rows
for the photo_receipt type. IAM pages are collages of 15 consecutive dataset
line-images on a white canvas, so each "page" is genuine handwriting with a
known-exact transcript.

## Adding your own documents

1. Drop the image under `documents/<tier>/<doc_type>/` (create the type folder
   freely — tiers must be `easy`/`medium`/`hard`).
2. Write `<stem>_gt.txt` next to it (plain text; markdown if structure
   matters). For tables, a `<stem>_gt.json` with `plain_text` and `table_html`
   keys enables TEDS.
3. If the GT is machine-generated or partial, add the doc type or stem to the
   silver sets in `evaluation/ground_truth.py` so text metrics don't lie.
4. Re-run — discovery is automatic (any `.png/.jpg/.jpeg` not ending in
   `_gt`).

Please only contribute documents whose license permits redistribution, and add
them to the attribution table above.
