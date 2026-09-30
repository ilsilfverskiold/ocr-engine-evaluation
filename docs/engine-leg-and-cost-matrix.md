# Engine expansion: benchmark legs, passes and preliminary costs

Research date: 2026-09-30. Read this alongside the
[pipeline/GPU implementation handoff](engine-expansion-research.md). This
document supersedes that handoff's abbreviated leg suggestions. Research only;
no new engine, inference run or judge execution has been performed.

## What this benchmark actually measures

Checked the executable registry, pass planner, GT loader, scoring dispatcher
and individual scorers, not just the README. Counts below were computed with
`GroundTruthLoader` over the current 93-page corpus, without inference.

| Leg | Task and headline | Eligible references | With judges disabled |
| --- | --- | ---: | --- |
| `text` | Read the page; deterministic GT coverage: fraction of unique reference words present. CER/WER/word-F1 support it. Order and extra text are not penalised by the headline. | 82 | Score normally. |
| `markdown` | Preserve headings, lists, tables and reading order. Normally image-derived structural checklist judged against output. | 57 | Deterministic structural F1 fallback; report as a different metric from judged element accuracy. |
| `tables` | Recover table grid/cells, including spans. Normally judged cell accuracy with TEDS supporting. | 42 | Deterministic TEDS; report as a different metric from judged cell accuracy. |
| `key_value` | Extract all printed labels and values, including blank fields, without being given the document's schema. Judge checks scalar field values. | 65 pass-eligible; **56 scoreable** | No supported deterministic scorer; collection must be separated from scoring. |
| `schema` | Fill the supplied document-specific schema, including nested fields/arrays. Judge checks leaf values; this is not a strict JSON-validity leaderboard. | 65 | No supported deterministic scorer; collection must be separated from scoring. |
| `geometry` | Locate text: exact normalised word match plus mutual box-centre containment. IoU is diagnostic. Line GT uses line merging/fuzzy text matching. | 17: 6 line, 11 word | Score normally. |

The 11 silver-text pages have a separate **document judge**, not a seventh
leg. Leave their text quality unscored until judging is requested, but still
collect their outputs. The KV scorer excludes pages with fewer than three
scalar reference fields; `has_gt_for('key_value')` does not apply that cutoff,
so current pass planning can spend 65 calls for only 56 scored pages. Charts
are excluded from table scoring even when a table sidecar exists.

Sources: [leg definitions/planner](../evaluation/legs.py),
[registry](../engines.py), [GT eligibility](../evaluation/ground_truth.py),
[dispatcher/CLI](../run_benchmark.py), [text scorer](../evaluation/text_leg.py),
[Markdown scorer](../evaluation/markdown_leg.py), [table scorer](../evaluation/table_leg.py),
[KV scorer](../evaluation/keyvalue_leg.py), [schema scorer](../evaluation/schema_leg.py),
[geometry scorer](../evaluation/geometry_leg.py).

## Eligibility rules and existing inconsistencies

An engine enters a leg through native output, deterministic conversion of that
output, a supported prompt, or a vendor extraction interface. A table rendered
as HTML is still table output. Plain text placed in a Markdown file does not
establish structural Markdown capability. Do not add a second model to infer
missing structure and silently attribute it to the OCR engine.

The [scoring design](scoring-design.md) says promptable engines should compete
even when their scores are poor. Follow the author's task recipe, record the
exact prompt, and check reasonable alternatives before publishing poor results.
An OCR fine-tune's expected weakness alone is not evidence that it has no
prompt interface.

There are existing discrepancies to record, not silently copy into new rows:

- The prose excludes layout engines/DeepSeek from geometry, but the current
  registry includes Paddle, MinerU, Marker and DeepSeek. The dispatcher accepts
  native `text_regions` as well as `words`. Coarse text regions can therefore
  compete, with granularity disclosed; they are not equivalent to word boxes.
- The registry explicitly dropped DeepSeek v1's KV leg after failed prompting,
  yet retained schema. This conflicts with the prose's interface-based rule.
  Keep the historical run intact; do not inherit this exception automatically
  for DeepSeek-OCR-2. Record the chosen policy in its new registry entry.
- The prose says 45 table references and roughly 60 KV references. Executable
  eligibility/scoring currently gives 42 and 56 respectively.

For geometry, preserve genuine text-to-box associations and original polygons.
Normalise to the shipped page coordinates, undoing resizing/cropping/rotation.
Use `words` only for actual words, `text_regions` for coarser text. Never split
a paragraph into fabricated word coordinates. Image-only bounding boxes (Ovis
image tags, LightOn bbox variants) do not qualify as native text geometry.

## Candidate matrix for the requested engines

**N** = native output/default conversion; **P** = separate prompted candidate;
**A** = vendor feature/configuration; **R** = native text-region geometry,
requiring granularity validation; **?** = unresolved interface/adapter contract;
**—** = no route established in the proposed integration. These indicate what
to implement/test, not successful accuracy results. Every native route still
needs response validation. P extraction results are to be collected without
judging when collection support exists.

| Engine | Text | Markdown | Tables | KV | Schema | Geometry | Initial pass design / caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TeleOCR 1.2B | N | N | N | — | — | R | One full layout/crop pipeline; retain region text and boxes. No arbitrary full-document extraction route established in this pipeline. |
| RapidOCR | N | — | — | — | — | N | One detection/recognition call; request word results if available, otherwise disclose lines. |
| OvisOCR2 0.8B | N | N | N | P | P | P | One official conversion prompt supplies T/M/tables. Separate extraction/word-grounding prompts are candidates; image tags alone supply no word geometry. |
| PaddleOCR-VL 1.6 | N | N | N | — | — | R/A | Full v1.6 pipeline; native regions or documented spotting configuration. Fixed recognition prefixes are not a schema interface. |
| DeepSeek-OCR-2 | N | P | P | P | P | R | Plain `Free OCR.` pass; one shared grounded conversion for M/tables/regions; separate extraction asks. Do not inherit v1 KV exclusion without an explicit policy decision. |
| Surya OCR 2 | N | N | N | P | P | R | Full-page/predictor conversion; manager also accepts custom prompts. Separate extraction passes must preserve raw output. |
| MinerU 4.0 Standard | N | N | N | — | — | R? | One complete Standard pipeline; verify text-region association in the new 4.x response schema. A bare VLM is a different integration. |
| Mistral OCR 4.1 | N | N | N | ? | A | R/A | Default conversion with blocks; document annotations for schema. KV requires an annotation contract that discovers printed labels without receiving gold schema keys. |
| Google Enterprise Document OCR | N | — | — | — | — | N | One OCR processor call, native tokens/polygons. Do not substitute Form Parser/Layout Parser under this name. |
| Mathpix OCR | N | N | N | — | — | A | Image conversion; word-data request separate when document-layout mode is used because those options conflict. |
| OpenDoc-0.1B | N | N | N | — | — | R? | Full OpenOCR document pipeline, not recognition-only. Validate native layout text associations. |
| LightOnOCR-2-1B | N | N | N | — | — | — | Official image-only OCR recipe, one conversion. No arbitrary-task route established; image bbox variant is not word grounding. |
| Apple Vision OCR | N | — | — | — | — | N | One local OCR call, derive actual word-range boxes. Structured document API would be a distinct macOS 26 integration. |
| Chandra OCR 2 | N | N | N | P | P | R | `ocr_layout` conversion; custom `BatchInputItem.prompt` for extraction. Preserve `raw` before HTML/Markdown parsing. |
| dots.mocr | N | N | N | P | P | R | One native layout conversion; general QA interface makes extraction a prompted candidate. Preserve text-region granularity. |
| Datalab hosted OCR | N | N | N | ? | A | A | Convert with an explicit mode; paid word boxes. Extract supplies schema; KV contract must discover labels without leaking the GT schema. |

RapidOCR and LightOn duplicates in the user's list are each one integration.
Native T/M/tables for a prompted conversion model means the adapter's official
default conversion answers all three, not that no prompt exists internally.

Pipeline/model sources and contracts are linked per engine in the
[main handoff](engine-expansion-research.md#model-contracts-and-pitfalls).
Additional interface checks materially expand that handoff's extraction list:

- Chandra's [input schema](https://github.com/datalab-to/chandra/blob/master/chandra/model/schema.py)
  and [vLLM implementation](https://github.com/datalab-to/chandra/blob/master/chandra/model/vllm.py)
  explicitly pass caller text to the model. Its
  [manager](https://github.com/datalab-to/chandra/blob/master/chandra/model/__init__.py)
  always parses layout output, so the adapter must retain raw extraction text.
- Surya's [input schema](https://github.com/datalab-to/surya/blob/master/surya/inference/schema.py)
  exposes a prompt override and guided JSON. Its
  [manager](https://github.com/datalab-to/surya/blob/master/surya/inference/__init__.py)
  forwards batches to the backend. Test extraction through that route without
  replacing the default predictor pipeline.
- [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) uses a text-and-image prompt;
  [dots.mocr](https://huggingface.co/dots-studio/dots.mocr) documents QA as well
  as layout tasks. This establishes candidates, not measured schema quality.

For fixed pipelines, a lower-level backbone experiment is possible future work,
but must be explicitly identified. Do not call a wrapper `accepts_task=True`
when it ignores the supplied task. Conversely, do not label Surya/Chandra
unpromptable simply because their common OCR examples use fixed task names.

## Passes, costs and deferred judging

`plan_passes()` merges native legs into one call and prompted legs only when
their resolved prompt, response format and schema requirement match. Count
unique inference passes, not six costs for six scored legs. Current examples:
Paddle/MinerU use one pass; DeepSeek uses default, shared M/tables/geometry,
and schema; Qwen uses five passes because M/tables share a prompt.

For a full 93-page collection, plan 93 default conversions. Additional
GT-limited passes are up to 65 schema, 65 KV under today's planner (56 if the
future collection planner explicitly uses scorer eligibility), and 17 geometry.
DeepSeek's shared M/tables/geometry union is **63 pages**, so its no-judge
collection can be 93 + 63 passes rather than 93 + 57 + 42 + 17. Retain the
same pass recipe/provenance when comparing with historical outputs.

Current `--skip-judge` drops schema/KV from `all`, rejects explicit requests
for them, and filters the document union down to **82 pages**. It therefore
cannot collect all requested future-leg outputs or all 93 pages as-is. The
implementation needs collection-only execution and later scoring of saved
responses. `--report` only aggregates stored scores; `--resume` is not a
deferred-judge command. Keep the prior run immutable and record an extension
run. Vendor extraction is engine inference, distinct from a benchmark judge.

## Preliminary spend: hosted engines

USD, public rates checked 2026-09-30, excluding tax, credits, retries and
benchmark judges. Arithmetic below assumes 93 **separate single-page** inputs,
including the 11 pages collected without text scores. No requests were sent.

| Engine/configuration | Published rate | 93 default conversions | Additional eligible passes |
| --- | --- | ---: | --- |
| Mistral OCR 4.1 | $4/1,000 OCR pages; $5/1,000 annotated pages | $0.372 | Separate 65-page annotation pass: $0.325; combined $0.697. |
| Google Enterprise OCR, base | $1.50/1,000 at standard paid tier | $0.1395 gross | Native geometry shares conversion; no extraction in this processor. Unused first-1,000-page free allowance may cover this. |
| Mathpix image OCR | $0.002/image; images with >12 text rows may incur $0.005/page | $0.186–$0.465 | Separate 17 geometry calls: $0.034–$0.085; total $0.220–$0.550. |
| Datalab Balanced Convert | $4/1,000 pages, rounded up per request to cents | **$0.93** | Word boxes +$3/1,000 before rounding; same-call Balanced + words remains $0.01/page, hence $0.93 total under this rule. |
| Datalab Accurate Convert | $10/1,000 pages, same rounding | $0.93 | Accurate + words is $0.013 before rounding: $1.86 for all 93; $1.10 if enabled only on the 17 geometry pages within their conversion calls. |

Sources: [Mistral model pricing](https://docs.mistral.ai/models/ocr-4-1),
[Google pricing](https://cloud.google.com/products/document-ai/pricing),
[Mathpix pricing](https://mathpix.com/pricing/api),
[Datalab rates](https://www.datalab.to/pricing) and
[per-request billing/rounding](https://documentation.datalab.to/platform/billing).

Mathpix's first key requires a $19.99 activation payment with $29 testing
credit; that cash outlay is separate from processing value. Google lists OCR
add-ons at $6/1,000; budget another $0.558 for 93 pages if selected and verify
the processor/options before enabling. Base OCR is sufficient for T/G.
Datalab extraction is separately billed: Turbo/Fast starts at $6/1,000,
Balanced at $15/1,000, Accurate at $20/1,000. For 65 separate requests the
rounded base amounts are $0.65/$1.30/$1.30, before any conversion component or
variable compute/large-output charge. Do not present those as total quotes.
Use returned `cost_breakdown` for actual spend. EU processing adds 25% before
final billing effects; the table assumes US list rates. Credits may reduce cash
spend but are not a cheaper engine. Do not combine unrelated benchmark pages
into a PDF merely to avoid rounding.

## Preliminary spend: Modal and local engines

There are no measured runtimes for these new integrations. The table below is
**sensitivity arithmetic**, not a throughput prediction. Seconds mean total
occupied GPU seconds divided by successful pages, including crop work, not
client latency or concurrent request durations added together.

| GPU | GPU $/hour | 93 pages at 1 s/page | At 5 s/page | At 15 s/page |
| --- | ---: | ---: | ---: | ---: |
| L4 | 0.7992 | $0.0206 | $0.1032 | $0.3097 |
| A10 | 1.1016 | $0.0285 | $0.1423 | $0.4269 |
| L40S | 1.9512 | $0.0504 | $0.2520 | $0.7561 |
| A100 80 GB | 2.4984 | $0.0645 | $0.3227 | $0.9681 |

Use `GPU_seconds × GPU_rate + CPU_core_seconds × 0.0000131 +
RAM_GiB_seconds × 0.00000222`, plus applicable storage/other charges.
One physical core is two vCPUs. Example allocation of two physical cores and
16 GiB adds $0.00006172 per occupied second. Startup, retries and retained
container time must be included in run spend, separately from warm throughput.
Base task rates: [Modal pricing](https://modal.com/pricing).

| Engine | Candidate deployment/inference route | Preliminary cost basis |
| --- | --- | --- |
| TeleOCR | Native layout/crop pipeline + vLLM; L4/A10 → L40S | L4/A10 scenario initially; include layout and all crops. |
| RapidOCR | Persistent local ONNX Runtime CPU | $0 metered API/GPU spend on owned host; record CPU time. |
| OvisOCR2 | Native vLLM batches; L4/A10 → L40S | L4/A10 scenario plus separate prompted-pass time. |
| Paddle 1.6 | Full Paddle pipeline + vLLM; A10 → L40S | A10/L40S scenario; include both model stages. |
| DeepSeek 2 | vLLM; L4/A10 batch 1 → L40S | Default plus 63 grounded passes for current deterministic-leg coverage; measure both tasks separately. |
| Surya 2 | Predictor/manager + external vLLM; L4/A10 → L40S | L4/A10 scenario; explicit extraction passes add time. |
| MinerU 4 | Standard full pipeline; A10 → L40S | A10/L40S scenario; whole pipeline, not recognizer alone. |
| OpenDoc | Full pipeline on CPU, then L4 probe | Local metered spend $0; optional L4 scenario. Layout remains part of cost. |
| LightOn 2 | Image-only vLLM; L4/A10 → L40S | One default conversion; L4/A10 scenario. |
| Apple OCR | Local Swift/Vision on this Mac | $0 metered API/GPU spend; record host/runtime. |
| Chandra 2 | Official manager + vLLM; A10 → L40S | A10/L40S scenario plus custom-prompt passes/retries. |
| dots.mocr | Native parser + vLLM; L4/A10 → L40S | L4/A10 scenario plus QA/extraction pass time. |

These hardware choices are fit/throughput probes from the Modal skill, not
guarantees. Keep one replica initially and benchmark native batching. A bigger
GPU can cost less per page if its throughput gain exceeds its rate increase.
Illustrative probe allocation: 30 billable minutes per engine on L40S with
the example CPU/RAM costs $1.0867 each, or $10.87 for ten potentially hosted
engines including OpenDoc. This is a planning allocation, **not a guarantee
that builds, all GPU sweeps or corpus runs finish within it**, and no budget
limit has been configured. Licensing fees, if required under model terms,
need a separate quote; compute estimates do not imply a commercial licence.

## Handoff decisions before a run

1. Encode native/prompted eligibility from this matrix after response-contract
   checks. Resolve `?` entries explicitly; log reasons and avoid GT leakage.
2. Preserve author conversion prompts, raw outputs and coarse geometry; make
   extraction routes use actual task inputs rather than fixed OCR defaults.
3. Implement independent collection/scoring; include all 93 source pages and
   persist each task/configuration. Judges remain disabled until requested.
4. Measure GPU fit and occupied time, then replace scenarios with measured
   per-pass estimates. Record provider charges, credits and startup separately.
5. Compare like metrics with like metrics against the preserved baseline;
   deferred judges can later supply the original judged headlines.

## Follow-up audit: flaws versus measurement limits

The user asked whether the existing benchmark itself has flaws. The following
were checked against code and saved run `20260810T130609Z`; no scores were
changed and no judges were called. This is a targeted audit, not proof that
every historical result is affected or a completed ranking-impact analysis.

1. **Confirmed false-positive field verification.**
   `evaluation/schema_judge.py::_verify_locally` removes punctuation and
   accepts an expected value as a substring anywhere in the output. Direct
   local calls reproduced all three incorrect `correct` verdicts:
   expected `total=100`, output `{"total":1000}`;
   expected `total=12.34`, output `{"total":1234}`;
   expected `total=100`, output `{"subtotal":100,"total":900}`.
   These local verdicts bypass the judge in both schema and KV scoring.
   This is a correctness defect, beyond harmless formatting tolerance.
   Historical incidence and ranking impact remain unmeasured. A fix should
   preserve numeric meaning and field association, with ambiguous cases sent
   to the judge; do not restore brittle exact schema-key matching for all
   engine formats.
2. **Schema task and headline do not establish the same capability.** The ask
   is to fill a schema, but the scorer checks whether its values occur in
   output, including raw text. A high score does not establish valid JSON,
   correct nesting/types, or correctly assigned fields. Preserve value
   recovery as useful evidence, but add separate schema-conformance and
   association measurements if claiming schema-following quality. KV also
   needs label/value association beyond value presence.
3. **Verdict-cache identity is insufficient for reruns.** The schema/KV
   verdict path uses document stem and engine name (KV has an engine suffix),
   without output, prompt, model revision, GT or judge/scorer hashes. The
   cache is read before the current output is evaluated. A changed response
   under the same engine ID can reuse an old verdict. This is a confirmed
   invalidation risk, not a demonstrated occurrence in the baseline. Later
   rescoring must use content/configuration-aware cache keys.
4. **Eligibility policy is inconsistent.** DeepSeek v1's KV exclusion after
   failed attempts conflicts with the written rule that promptable engines
   remain eligible despite poor scores; schema remains enabled. This calls
   for a consistent task-support policy, not automatically adding arbitrary
   prompting to every fixed OCR pipeline. Keep historical results intact.
5. **Historical Mistral identity is unresolved.** All 158 saved inference
   entries use `mistral-ocr-latest`. The row name/current runner cannot prove
   which service revision produced them. Label that limitation and pin new
   versions; do not retrospectively rename the baseline as a verified version.
6. **Accuracy averages are conditional on successful calls.**
   `aggregate_legs` drops rows carrying `error`. Comparing saved error keys
   against later successful `(engine, document, leg)` records leaves five
   unresolved Marker geometry cases. Other saved error rows were recovered;
   counting all error rows would overstate failures. The report does expose
   inference errors, but accuracy alone is not end-to-end reliability.
   Show completion coverage alongside quality, with any failure-adjusted
   metric separately labelled.

Text GT coverage's insensitivity to extra words, repetition and word order is
an explicitly documented design limit, rather than a newly discovered bug.
It should be read with CER/WER/precision and repetition indicators. The
no-judge collection/filtering limitations above concern the planned extension;
they do not establish that the prior judged run omitted those 11 pages.
