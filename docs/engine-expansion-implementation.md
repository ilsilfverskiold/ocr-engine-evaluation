# Engine expansion: what was built, where each engine runs, how to test it

Date: 2026-09-30. Implementation of the 16 integrations researched in
[engine-expansion-research.md](engine-expansion-research.md) and
[engine-leg-and-cost-matrix.md](engine-leg-and-cost-matrix.md). Nothing in
this document has been deployed, run or measured: every runtime claim below is
a contract read from source, and section 6 lists what the first deploy must
confirm. No judge, engine or benchmark command was executed as part of this
work (one exception, disclosed in section 8).

The 14 engines of the reference run `20260810T130609Z` are untouched: same
registry entries, same apps, same saved results. All 16 new engines are
`default=False` and are opted into with `--engines`.

## 1. Files

| Kind | Files |
| --- | --- |
| Modal apps (self-hosted) | `modal_apps/ovisocr2_app.py`, `lighton_ocr_2_app.py`, `deepseek_ocr_2_app.py`, `dots_mocr_app.py`, `surya_ocr_2_app.py`, `chandra_ocr_2_app.py`, `paddleocr_vl_1_6_app.py`, `opendoc_app.py`, `mineru_4_standard_app.py`, `teleocr_app.py`, `rapidocr_app.py` |
| Cloud runners | `runners/mistral_ocr_4_1_runner.py`, `google_document_ocr_runner.py`, `mathpix_runner.py`, `datalab_runner.py` |
| Local runner | `runners/apple_vision_runner.py` + `runners/apple_vision/recognize.swift` |
| Registry | 16 `EngineSpec` entries appended to `engines.py` (30 engines total, 14 default) |
| Harness | `run_benchmark.py`: `--collect-only`, `--score`, `--extends`, `--docs`, combined `--report A,B`; `modal_apps/_shared.py`: a `CPU` container rate, unknown GPU types now raise |
| Config | `.env.example` (new keys), `requirements.txt` (`google-cloud-documentai`, `requests`) |
| Tests | `tests/conftest.py` (strips paid credentials from the environment for every test), membership test split into baseline / expansion, two stale tests corrected |

Every app returns the same batch envelope the existing Modal engines do
(`results`, `batch_gpu_seconds`, `gpu_type`, `startup_gpu_seconds`,
`is_cold_start`), plus per item `prompt_used`, `model_version` with the pinned
revision, and `structured_output["generation"]` (finish reason and token
counts) wherever the backend exposes them. A fixed pipeline that receives a
prompt returns an error item rather than silently ignoring the prompt.

## 2. Where each engine runs, and what it enters

GPU choices are the first fit probe per the Modal skill, not recommendations:
the smallest card that holds weights + KV at the chosen concurrency. "Next"
is the one-variable hardware experiment after the fit probe. Costs are $0
in the registry because nothing is measured; the report derives cost from the
persisted GPU-seconds and the actual GPU each batch reports.

| Engine | Runs on | Passes for `--legs all` | Legs entered | Geometry granularity |
| --- | --- | --- | --- | --- |
| `mistral_ocr_4_1` | Mistral API | default + schema | T M B G, schema (Annotations) | block |
| `google_document_ocr` | Document AI (ADC) | default | T G | word (tokens) |
| `mathpix` | Mathpix API | default + geometry (word-data config) | T M B, G | word |
| `datalab_ocr` | Datalab API | default + schema (Extract) | T M B, schema | none yet (see §6) |
| `paddleocr_vl_1_6` | Modal **A10** → L40S | default | T M B G | block |
| `opendoc` | Modal **CPU** (4 cores, 8 GiB) → L4 | default | T M B G | block |
| `deepseek_ocr_2` | Modal **L4** → A10 → L40S | default(`Free OCR.`) + grounding(M+B+G) + KV + schema | T, M B G, KV, schema | block (grounding boxes) |
| `dots_mocr` | Modal **L4** → A10 → L40S | default + KV + schema | T M B G, KV, schema | block |
| `surya_ocr_2` | Modal **L4** → A10 → L40S | default + KV + schema | T M B G, KV, schema | block |
| `chandra_ocr_2` | Modal **A10** (concurrency 1) → L40S | default + KV + schema | T M B G, KV, schema | block |
| `mineru_4_standard` | Modal **A10** (8 CPU, 32 GiB) → L40S | default | T M B G | block |
| `teleocr` | Modal **A10** (8 CPU, 16 GiB) → L4 → L40S | default | T M B G | block |
| `rapidocr` | Modal **CPU** (4 cores, 8 GiB) | default | T G | word, else line |
| `ovisocr2` | Modal **A10** → L4 → L40S | default + KV + schema | T M B, KV, schema | none |
| `lighton_ocr_2` | Modal **A10** → L4 | default | T M B | none |
| `apple_vision` | this Mac (opt-in) | default | T G | word |

T/M/B/G = text / markdown / tables / geometry. Leg decisions that were
judgement calls, with the reasoning recorded beside each registry entry:

- **`deepseek_ocr_2` key_value is enabled.** v1 dropped it after 0/65 JSON
  replies; the leg matrix said not to inherit that. The app forwards the
  prompt, so under the interface rule in [scoring-design.md](scoring-design.md)
  the engine competes and its score is its score. Expectation is low (the
  paper describes no instruction stage); that is a result, not a reason to
  skip. Cost: 65 extra L4 passes.
- **`ovisocr2` has no geometry leg** although the matrix listed a prompted
  candidate: neither the card nor the report documents text grounding, so a
  word-box prompt would be an invented ask. Its `<img>` boxes locate figures.
- **`chandra_ocr_2` extraction runs without constrained decoding.** Its
  vLLM route forwards the prompt but sends no `response_format`; Surya's does.
  Recorded per item as `response_format_applied=False`. The two are not
  equivalent extraction routes.
- **`mathpix` geometry is a second billed call** because `include_word_data`
  cannot be combined with `enable_document_layout`. The registry prompt
  `__mathpix_word_data__` is a configuration selector (the same device marker
  uses), never sent to the API.
- **`datalab_ocr` geometry is withheld** until two paid conversions confirm the
  JSON block shape; the runner already parses it into `text_regions`.
- **key_value is not offered** for `mistral_ocr_4_1` and `datalab_ocr`: both
  vendors' extraction needs a caller-supplied schema, and inventing a generic
  one would be our construction. Schema (their documented feature) is offered.
- **`apple_vision` is local inference on this machine.** It exists because it
  was requested; it stays opt-in and off by default. Whether it runs is your
  decision.
- **`rapidocr` and `opendoc` run on Modal CPU**, not locally, per the
  no-local-inference rule. `GPU_TYPE="CPU"` bills 4 cores + 8 GiB per occupied
  second (`modal_apps/_shared.py`); an app declaring `CPU` must request
  exactly that allocation.

## 3. Harness changes (why each exists)

- **`--collect-only`**: runs inference and persists every output, scores
  nothing and calls no judge. Keeps key_value/schema passes (plain
  `--skip-judge` also keeps them) and keeps all 93 pages (leg-based GT filtering
  was removing the 11 silver-text pages). Needs no `GOOGLE_API_KEY`.
- **`--score TS`**: scores a stored run from its `.call.json` files, then
  prints the report. Appends only rows not already in `TS_results.jsonl`, so a
  rescore requires moving the old results aside first — earlier verdicts are
  never overwritten. `--score TS --skip-judge` uses current code scorers and existing caches;
  unresolved KV judgments remain pending. Use `--rescore` for a new score ledger.
- **`--extends TS`**: writes the baseline run ID into the new run's manifest.
- **`--report A,B`**: one report over an extension run and its baseline.
  Rows keep their own `timestamp`; the same (engine, leg) in two runs is
  refused rather than averaged. Nothing is rescored across runs.
- **`--docs stem,stem`**: run a chosen cross-section (section 4).
- **`--resume` no longer overwrites the manifest** with the resume's
  `--engines/--legs` subset; it merges. Resuming a collect-only run with the
  scored mode (or vice versa) is refused.
- **`calculate_modal_cost` raises on an unknown GPU type** instead of billing
  at the T4 rate.
- **The document judge follows the text-serving pass** instead of the pass
  named `default`, and a collect-only run does not skip that pass on the 11
  silver-text pages. In the baseline those two gates left deepseek, glm, qwen
  and marker with no judged estimate (audit §5.3). Schema passes are still
  skipped where the document has no schema.
- **`tests/conftest.py`** removes every paid credential from the environment
  before each test. See section 8 for why.

## 4. Test order

Run these yourself; none has been run here.

1. **Deploy one app and smoke it.** `scripts/deploy_modal.py` derives the app
   list from the registry, so
   `python scripts/deploy_modal.py --apps rapidocr` deploys and smoke-tests
   one app. Build failures are expected on first contact for the heavier
   stacks (Paddle 1.6, MinerU 4, TeleOCR, Surya, Chandra): section 6 lists the
   pins most likely to need a nudge.
2. **Validation cross-section, collect only** — the same set for every engine,
   chosen to touch every leg, every tier and every GT provenance class:

   | Stem | Type | Legs with GT |
   | --- | --- | --- |
   | `sroie_000` | receipt (easy) | T B KV S G |
   | `invoice_000` | invoice (easy) | T M B KV S |
   | `photo_000` | photographed receipt | T M B KV S |
   | `tax_000` | tax form | T M B KV S |
   | `fintab_000` | financial table | T M B KV S G |
   | `funsd_000` | form | T M KV S G |
   | `iam_000` | handwriting | T |
   | `legal_000` | legal fine print (model-verified GT) | T |
   | `chart_000` | chart | T M KV S |
   | `scanned_000` | legacy scan | T KV S |
   | `newspaper_000` | newspaper (silver GT) | none — output only |

   ```
   python run_benchmark.py --engines <engine> --collect-only \
       --docs sroie_000,invoice_000,photo_000,tax_000,fintab_000,funsd_000,iam_000,legal_000,chart_000,scanned_000,newspaper_000
   python run_benchmark.py --score <TS> --skip-judge
   ```
   Then read the `.call.json` files under `results/ocr_outputs/<TS>/`: this is
   where the section-6 probes are answered (token counts, finish reasons,
   box alignment, table HTML present, `prompt_used`).
3. **Full collection into one extension run**, all new engines that passed
   step 2:
   ```
   python run_benchmark.py --engines <list> --collect-only --extends 20260810T130609Z
   ```
   Cloud engines run in parallel with Modal ones as today. Expect 93 default
   passes per engine plus GT-limited prompted passes (65 schema / 65 KV / 17
   geometry) — `--estimate` and `--dry-run` print the exact call counts.
4. **Score later**, when judging is wanted:
   `python run_benchmark.py --score <TS>` (Gemini) or `--skip-judge`.
   Then `--report <TS>,20260810T130609Z` for the combined view.
5. **Hardware sweeps** only after step 3 has one clean run per engine: change
   one variable (active concurrency, then GPU), keep the prompt, revision and
   output caps fixed, and re-run the cross-section, not the corpus.

Suggested build order by risk: `mistral_ocr_4_1`, `rapidocr`,
`google_document_ocr`, `mathpix`, `datalab_ocr` (plain APIs / tiny image) →
`lighton_ocr_2`, `ovisocr2`, `deepseek_ocr_2`, `dots_mocr` (single vLLM
server) → `surya_ocr_2`, `chandra_ocr_2` (vendor manager + separate vLLM
venv) → `paddleocr_vl_1_6`, `opendoc`, `mineru_4_standard`, `teleocr`
(multi-stage pipelines, heaviest dependency stacks).

## 5. Comparability with the baseline

- Cost per page is comparable across GPUs because each batch is billed at the
  rate of the GPU it actually reports. Seconds per page are not: A10/L4 rows
  cannot be read as slower models than the A100 baseline rows.
- Use the same fixed scorers and accepted references as the current baseline
  replay `20261007T073234Z`; see [scoring design](scoring-design.md). KV uses
  contextual value-assisted review only when needed; schema uses the frozen
  common field sample. `--skip-judge` reuses cached decisions without spending.
- Record current provider model IDs. Historical aliases are not proof that a
  new call uses the same build as the original baseline.
- `mistral_ocr_4`'s model id (`mistral-ocr-4-0`) retires on 2026-09-30 per
  Mistral's changelog, and the saved baseline recorded `mistral-ocr-latest`.
  The baseline row stays as a historical record; it cannot be re-run as what
  it was.

## 6. Unverified before the first deploy

Condensed from the per-engine handoffs; each app's docstring has the full
list with the probe for it. Anything not listed as verified was read from
source, not observed.

- **Every Modal app**: image build resolves (pins chosen from each project's
  lock or CI file, but never installed together here); server starts and
  warm-up passes on the declared GPU; peak memory at the declared concurrency
  on the largest corpus page (2480×3508).
- **ovisocr2**: the nested `images_kwargs` pixel bounds are honoured by vLLM
  0.22.1 (check `generation.prompt_tokens` ≈ 8,152 on a 2480×3508 page);
  `gdn_prefill_backend=triton` on an A10; whether `schema_render` should be
  `instance_template` as GLM needed.
- **lighton_ocr_2**: HTML tables actually appear (sets `table_format`);
  temperature 0.2 makes runs non-deterministic — measure run-to-run variance.
- **deepseek_ocr_2**: rendered prompt is `<image>\n<prompt>`; grounding labels
  in v2 (is `image` still the picture label?); ngram 20/90 vs the recipe's
  30/90.
- **dots_mocr**: server-side resize matches the author's `smart_resize`
  (decides whether boxes land); the CLI's default 200-dpi re-render for images
  is NOT applied here.
- **surya_ocr_2 / chandra_ocr_2**: pip vLLM 0.20.1 / 0.17.0 instead of the
  vendors' Docker images; both upstream recipes let max visual tokens + max
  output exceed the 18,000 context, so vLLM clamps output to ~11–12k tokens
  (recorded as `possible_truncation`); Chandra is **5.3B / 10.6 GB**, not the
  4B in the research doc — A10 headroom is thin at concurrency 1.
- **paddleocr_vl_1_6**: vLLM 0.21.0 serving the 1.6 checkpoint with the
  explicit chat template; A10 fit at util 0.5 beside PP-DocLayoutV3;
  `layout_detection_model_dir` accepting the pinned snapshot.
- **opendoc**: ONNX path equivalence to the source path behind the published
  OmniDocBench number is not established; ORT thread count inside the 4-core
  allocation.
- **mineru_4_standard**: which `max_model_len` vLLM picks for the VLM
  (32,768 vs 8,192 in `text_config`); the pinned-download wrapper around
  MinerU's own downloader; A10 fit with the ~6 GiB vLLM reservation.
- **teleocr**: `pip install -c requirements.txt -e .` resolution on Python
  3.10; Pillow's 72-dpi PDF default implying upscaling toward 3500 px; vLLM
  0.11 ending at `max_model_len` with finish_reason `length`.
- **mistral_ocr_4_1**: table placeholder id spelling vs `tables[].id`; block
  pixel frame vs `pages[].dimensions`; whether annotated pages bill $5 total
  or $5 + $4.
- **google_document_ocr**: `revisions[].processor` populated; `imageless_mode`
  accepted by OCR v2.1; token polygons stay in the original frame on deskewed
  pages.
- **mathpix**: **three corpus images (medical_000 18.4 MB, bank_000 14.4 MB,
  shipping_000 7.5 MB) exceed the 5 MB request limit** and will error — a
  policy decision (skip, or a disclosed re-encode configuration).
- **datalab_ocr**: JSON tree shape and page bbox base (then enable geometry);
  `cost_breakdown.final_cost_cents` presence; form-string booleans.
- **rapidocr**: whether word results appear on real pages (upstream drops
  them for the page if any line lacks them → line fallback).
- **apple_vision**: compile once and run one receipt; `word_range_failures`.

## 7. Errors found in the research documents and existing code

Research documents (not edited; corrections recorded here):

- OvisOCR2: geometry listed as a prompted candidate — no grounding output
  exists. The card's default post-processing (`parse()`: img-tag filtering +
  truncated-repeat cleanup) is part of the recipe and was omitted.
- LightOnOCR-2: tables are HTML targets; trained at 6,144 sequence length
  with completions over 3,100 tokens filtered (context for the 4,096 cap).
- DeepSeek-OCR-2: vLLM support starts at 0.16.0; max visual tokens are 1,121
  including the separator; `Parse the figure.` is a v1-only prompt.
- dots.mocr: repo moved to `studio-dots-ai/dots.mocr`; general QA needs the
  system prompt `You are a helpful assistant.`; the parser's vLLM path uses
  32,768 output tokens (24,000 is the HF example).
- Surya 2: the full-page budget is 12,288 tokens (8,192 is block mode).
- Chandra 2: 5.3B parameters / 10.6 GB, not 4B; 192 DPI applies to PDFs only
  — images have their short side upsampled to 1,536 px.
- The leg matrix implies Surya and Chandra are equivalent for extraction;
  only Surya's route supports guided JSON.
- OpenDoc: the Torch 2.6 / Transformers 4.49 / Paddle recipe belongs to the
  source path; `openocr-python==0.1.5`'s `task='doc'` is ONNX Runtime only.
- TeleOCR: layout detection is done by the same 1.2B VLM, not a separate
  layout model.
- MinerU 4: code is under the custom "MinerU Open Source License" since
  3.1.0 (commercial threshold, attribution duty), not Apache/AGPL; weights are
  Apache-2.0.
- Mathpix: the 5 MB limit is on the whole multipart body, so "use uploads to
  avoid base64 limits" does not help the three large pages.
- Google: image quality scores are not a billed add-on (math, checkbox and
  font-style are).
- Datalab: Extract parses in "accurate" mode unless `mode` is set, and
  parsing is billed separately from extraction.
- The matrix lists RapidOCR and Apple as $0 local engines; RapidOCR is on
  metered Modal CPU here.

Historical pre-fix code findings (not a current defect inventory; originally
quantified in `benchmark-audit-2026-09-30.md` (historical audit retained in Git; current rules: [scoring design](scoring-design.md))):

- `modal_apps/deepseek_ocr_app.py` (v1) stores each grounding box's **label**
  (`title`, `text`, `table`…) as its text — the `<|ref|>` group is the layout
  category; the content follows the tag. Baseline geometry for `deepseek_ocr`
  is 0.0 on all 17 applicable pages. It also stores region boxes under
  `words`. Its `Free OCR.` output is flattened (headings/table markup
  stripped) before the text leg reads it.
- `modal_apps/paddleocr_vl_app.py` (v1): PaddleX 3.7's `res.json` has no
  `overall_ocr_res`, so none of the 93 saved outputs carries `text_regions`;
  the geometry leg scored this engine from raw markdown text. All 106 stored
  results used the `parsing_res_list` table fallback; every `model_version`
  says `layout=unknown` (the probe attribute names never match). Its
  docstring says PP-DocLayout**V3**; v1 uses V2 (the skill's case-study
  reference repeats this). Two comments are stale ("torch 2.8", "doc-parser
  unpinned"); `paddlex` itself is unpinned. **Edited:** only the
  `engines.py` comment claiming `use_seal_recognition=False` (the app sets
  True) and a note recording the geometry finding.
- `modal_apps/mineru_app.py` stores layout-block boxes under `words` and does
  not EXIF-transpose inputs. `modal_apps/glm_ocr_app.py` also puts line boxes
  under `words`.
- `runners/mistral_runner.py` dumps the whole reply into `structured_output`
  instead of `raw_response`, reports the requested rather than returned model,
  and with mistralai 2.9.1 sends `include_blocks=True` by default, so today's
  `mistral_ocr_4` request already differs from the historical one.
- Upstream `rapidocr` 3.9.2: `word_results` can fall out of alignment with
  `boxes`/`txts`, so the app never maps words back to lines. Upstream
  `openocr` 0.1.5: suffixed labels defeat its own chart/seal handling.

## 8. Disclosure: one unrequested judge run

While verifying the harness edits, the unit-test suite was run. The
pre-existing `tests/test_reference_self_consistency.py` scored every markdown
reference with `use_judge=None`, and because `run_benchmark` loads `.env` at
import, the real `GOOGLE_API_KEY` was present: **57 `gemini-3.6-flash` verdict
calls were made** (checklists were already cached). The 57 cache files it
wrote (`results/markdown_judge_cache/verdicts/<stem>__.json`, engine name
empty) were deleted; nothing else under `results/` changed. `tests/conftest.py`
now strips every paid credential before each test, and the test passes
`use_judge=False`. The by-product (the markdown judge scores the human
markdown references below 1.0 on 42 of 57 pages) is recorded in the audit
document as expected behaviour — the judge grades against Gemini's reading of
the page image, not against the human sidecar — and only as a reminder that
judged and deterministic markdown numbers are different metrics.

## 9. Expansion decisions (historical scoring choices superseded)

1. Apple Vision: run it on this Mac, or leave it opt-in and unrun.
2. Mathpix: skip the three >5 MB pages, or add a disclosed re-encode
   configuration for that engine only.
3. `deepseek_ocr_2` key_value: enabled by the interface rule; say if you want
   v1's exclusion carried forward instead.
4. `datalab_ocr`: Balanced only, or also an Accurate row (separate engine id).
5. Scoring is settled: run the same fixed six-leg evaluation and accepted
   references as the current baseline replay. Further paid runs need approval.
6. The old adapter/scorer defects listed above are historical findings. Current
   eligibility, conversions and remaining unmeasured pipeline controls are
   documented in the scoring design and follow-up, not inferred from that list.
