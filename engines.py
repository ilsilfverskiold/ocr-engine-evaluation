"""
Single engine registry — the ONE place an engine is declared.

Adding an engine:
  1. Write a runner in runners/<name>_runner.py subclassing OCRRunner
     (and, for self-hosted engines, a Modal app in modal_apps/<name>_app.py).
  2. Add one EngineSpec entry to ENGINES below.

Everything else — CLI defaults, key validation, --check, --estimate, Modal
deploy, batch sizes, cost math — reads from this registry. See
docs/adding-an-engine.md for worked examples.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineSpec:
    name: str
    # "local"  — runs on this machine (CPU); executed sequentially.
    # "cloud"  — an HTTP API; needs the env vars in required_env.
    # "modal"  — self-hosted model deployed via scripts/deploy_modal.py.
    kind: str
    # Import path of the runner class, "module:ClassName". Modal engines all use
    # the generic ModalRunner, constructed with the engine name.
    runner: str
    # Env vars that must be set for this engine to run (checked by --check and
    # validate_keys). Empty for local and Modal engines.
    required_env: tuple[str, ...] = ()
    # Static $/page used by --estimate. For Modal engines this is a measured
    # batch estimate; real runs compute the exact figure from GPU-seconds.
    cost_per_page: float = 0.0
    cost_notes: str = ""
    # Whether the engine is in the default --engines list.
    default: bool = True
    # --- Modal-only fields -------------------------------------------------
    modal_app: str | None = None        # deployed Modal app name
    modal_class: str | None = None      # modal.Cls name inside the app
    modal_app_file: str | None = None   # app file path relative to project root
    # WARNING: gpu_type here is only a FALLBACK for cost calculation. The
    # authoritative value is GPU_TYPE in the corresponding modal_apps/*_app.py,
    # which decides which GPU Modal actually allocates AND is returned in every
    # response. ModalRunner._to_result() prefers the response value, so a
    # mismatch here no longer mis-bills a run — but keep them in sync anyway.
    # Changing only the app file previously billed A100 runs at the A10G rate,
    # understating cost by 2.3x (4.2x for marker on T4).
    gpu_type: str | None = None
    # Documents per run_ocr_batch call. Every Modal engine is measured through
    # one explicit batch on one container; see runners/modal_runner.py for why.
    batch_size: int = 0
    supports_pdf: bool = False
    # --- Leg eligibility ---------------------------------------------------
    # Whether runner.run() can be handed an OCRTask (a prompt, and for the
    # schema leg a JSON schema). False for fixed pipelines with no prompt:
    # they compete in a leg only through their default conversion output.
    accepts_task: bool = False
    # Legs this engine answers from its DEFAULT conversion — no extra call, no
    # prompt. Legs 1-3 are scored from one pass for almost every engine.
    native_legs: tuple[str, ...] = ("text",)
    # Legs this engine answers only when given that leg's task. Requires
    # accepts_task=True; each distinct task is one extra inference pass.
    prompted_legs: tuple[str, ...] = ()
    # Human-readable reason an engine sits out the legs it cannot answer.
    # Printed in the report so a blank cell is never unexplained.
    leg_notes: str = ""
    # How this engine's text carries a table, for the table leg's converter:
    # "markdown" (pipe tables), "html" (real <table> markup), "tsv"
    # (tab-separated), or None when tables come from structured_output or not
    # at all. Declared here rather than in evaluation/table_extractor.py so the
    # registry cannot disagree with the converter about what an engine emits.
    # Per-leg prompt overrides, keyed by leg name. A specialised model degrades
    # under a generic instruction, so where its model card prescribes a prompt or
    # task prefix, that wins over the generic leg prompt in evaluation/legs.py.
    # Cite the source in a comment beside each entry.
    # See docs/scoring-design.md, "Prompting is part of the measurement".
    leg_prompts: dict = field(default_factory=dict)
    table_format: str | None = None
    # How the schema leg's {schema} placeholder is rendered for this engine:
    # "json_schema" (the schema document) or "instance_template" (an empty
    # example object). Model cards differ on which they demonstrate, and the
    # wrong one is not a style choice — GLM-OCR echoes a JSON Schema back
    # instead of filling it. See evaluation/legs.py:instance_template.
    schema_render: str = "json_schema"
    # Legs whose answer, for THIS engine, is not JSON — so the provider's JSON
    # mode must not be switched on. deepseek_ocr answers geometry with the
    # grounding markup its card documents (<|ref|>…<|det|>[[…]]), not a JSON
    # array: forcing JSON mode would both corrupt the ask and split the call
    # off from the markdown pass it shares a prompt with, paying twice for one
    # answer. Declared per engine because it is a fact about the engine, not
    # about the leg.
    non_json_legs: tuple[str, ...] = ()

    def response_format_for(self, leg: str, needs_schema: bool) -> str | None:
        """The provider response format for this engine on this leg."""
        if leg in self.non_json_legs:
            return None
        return "json" if needs_schema or leg in ("key_value", "geometry") else None

    def prompt_for(self, leg: str, generic: str | None) -> str | None:
        """This engine's prompt for a leg: its own if the model card gives one,
        otherwise the generic leg prompt."""
        return self.leg_prompts.get(leg, generic)

    def answers(self, leg: str) -> str | None:
        """How this engine answers `leg`: "native", "prompted", or None."""
        if leg in self.native_legs:
            return "native"
        if leg in self.prompted_legs:
            return "prompted"
        return None

    @property
    def legs(self) -> tuple[str, ...]:
        return tuple(self.native_legs) + tuple(self.prompted_legs)


# Shorthand for the three legs almost every layout-aware engine answers from a
# single default conversion call — see docs/scoring-design.md "Two passes".
_CONVERSION_LEGS = ("text", "markdown", "tables")

ENGINES: dict[str, EngineSpec] = {
    # --- Local (free, no keys) --------------------------------------------
    "tesseract": EngineSpec(
        name="tesseract", kind="local",
        runner="runners.tesseract_runner:TesseractRunner",
        cost_notes="local, free",
        native_legs=("text", "geometry"),
        leg_notes="flat text only: no markdown, no table structure, no prompt",
    ),
    "docling": EngineSpec(
        name="docling", kind="local",
        runner="runners.docling_runner:DoclingRunner",
        cost_notes="local, free",
        native_legs=_CONVERSION_LEGS,
        leg_notes="fixed pipeline: cannot be given a prompt or a schema",
        table_format="markdown",
    ),
    # --- Cloud APIs --------------------------------------------------------
    "textract_text": EngineSpec(
        name="textract_text", kind="cloud",
        runner="runners.textract_text_runner:TextractTextRunner",
        required_env=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        cost_per_page=0.0015,
        native_legs=("text", "geometry"),
        leg_notes="DetectDocumentText returns lines only: no structure, no prompt",
    ),
    "textract_structured": EngineSpec(
        name="textract_structured", kind="cloud",
        runner="runners.textract_structured_runner:TextractStructuredRunner",
        required_env=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        cost_per_page=0.065,
        cost_notes="AnalyzeDocument TABLES+FORMS",
        # schema removed 2026-09-30: AnalyzeDocument(FORMS, TABLES) takes no
        # schema and no field names, so nothing was ever asked of it on that
        # leg; its saved schema score came from value presence in its page
        # text. Amazon's interface for this is Textract Queries, which would
        # be a separate, separately billed configuration. key_value stays:
        # FORMS is a genuine label->value interface.
        native_legs=("text", "tables", "key_value", "geometry"),
        leg_notes=(
            "no prompt; key_value judged from its FORMS pairs; schema not "
            "applicable (no schema/queries interface in this integration)"
        ),
    ),
    "mistral_ocr_4": EngineSpec(
        name="mistral_ocr_4", kind="cloud",
        runner="runners.mistral_runner:MistralOCRRunner",
        required_env=("MISTRAL_API_KEY",),
        cost_per_page=0.004,
        native_legs=_CONVERSION_LEGS,
        # schema via the vendor's own extraction feature (Mistral Annotations /
        # LlamaExtract), not a prompt. UNVALIDATED until the 2-doc check runs.
        prompted_legs=("schema",),
        accepts_task=True,
        leg_notes="OCR endpoint takes no prompt: markdown/tables from default conversion",
        table_format="markdown",
    ),
    "llamaparse": EngineSpec(
        name="llamaparse", kind="cloud",
        runner="runners.llamaparse_runner:LlamaParseRunner",
        required_env=("LLAMA_CLOUD_API_KEY",),
        cost_per_page=0.00375,
        cost_notes="cost_effective tier: 3 credits × $1.25/1K; free tier: 10K credits/mo",
        native_legs=_CONVERSION_LEGS,
        # schema via the vendor's own extraction feature (Mistral Annotations /
        # LlamaExtract), not a prompt. UNVALIDATED until the 2-doc check runs.
        prompted_legs=("schema",),
        accepts_task=True,
        leg_notes="parse endpoint takes no prompt in the cost_effective tier",
        table_format="markdown",
    ),
    "gemini_flash": EngineSpec(
        name="gemini_flash", kind="cloud",
        runner="runners.gemini_runner:GeminiRunner",
        required_env=("GOOGLE_API_KEY",),
        cost_per_page=0.0007,
        cost_notes="variable — token-based",
        accepts_task=True,
        # The runner's transcription prompt serves text. Markdown and tables
        # are prompted with the leg's own ask: the 2026-08 run scored both
        # from transcription prompts requesting structure, Markdown tables
        # and reading order. Dedicated headings/lists and table-only asks
        # remain unmeasured on those saved pages.
        native_legs=("text",),
        prompted_legs=("markdown", "tables", "key_value", "schema", "geometry"),
        table_format="markdown",
    ),
    "claude_sonnet_4_6": EngineSpec(
        name="claude_sonnet_4_6", kind="cloud",
        runner="runners.claude_runner:ClaudeRunner",
        required_env=("ANTHROPIC_API_KEY",),
        cost_per_page=0.012,
        cost_notes="variable — token-based",
        accepts_task=True,
        # As for gemini_flash: text from the transcription prompt, markdown
        # and tables from the leg's own ask.
        native_legs=("text",),
        prompted_legs=("markdown", "tables", "key_value", "schema", "geometry"),
        table_format="markdown",
    ),
    # --- Self-hosted VLMs on Modal -----------------------------------------
    "mineru": EngineSpec(
        name="mineru", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.000922, cost_notes="Modal A100-80GB; measured batch estimate",
        modal_app="ocr-bench-mineru", modal_class="MinerUInference",
        modal_app_file="modal_apps/mineru_app.py",
        gpu_type="A100-80GB", batch_size=8,
        # geometry added 2026-08-10: every ContentBlock carries a bbox already
        # normalised 0-1 plus type, angle and reading order. Same single
        # two_step_extract call — nothing extra is asked for.
        native_legs=("text", "markdown", "tables", "geometry"),
        leg_notes="fixed internal task routing; no caller-supplied prompt. "
                  "Structured extraction is not a documented MinerU2.5 task",
        # Native table content IS HTML — the model emits OTSL and the wrapper
        # converts it. Declaring "tsv" pointed the extractor at the plain-text
        # rendering this app manufactures for the text leg, not at MinerU's own
        # table output.
        table_format="html",
        leg_prompts={
            # MinerU2.5 routes internally: "Layout Detection:" over the page,
            # then "Text Recognition:" / "Table Recognition:" /
            # "Formula Recognition:" per detected region. Those prompts exist
            # inside mineru-vl-utils; none is caller-supplied, which is why
            # accepts_task stays False. image_analysis stays at its default
            # False: the 2509 model card documents no chart/image task, and
            # that capability arrived with the later Pro models.
        },
    ),
    "glm_ocr": EngineSpec(
        name="glm_ocr", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.000617, cost_notes="Modal A10; measured batch estimate",
        modal_app="ocr-bench-glm-ocr", modal_class="GLMOCRInference",
        modal_app_file="modal_apps/glm_ocr_app.py",
        gpu_type="A10", batch_size=24,
        accepts_task=True,
        # markdown gets its own call rather than being read off the Text
        # Recognition output: that output is plain text and scored 0.325/0.800,
        # where an explicit markdown ask scores 0.388/0.943 on the same pages
        # (2026-08-09). The card documents no markdown task, so the generic leg
        # prompt is the ask — undocumented, and flagged as such — but it is the
        # engine's best available shot, which is what the leg must measure.
        native_legs=("text",),
        # REVERTED 2026-08-10. The official SDK document-parse path is the
        # documented way to get markdown/tables/geometry, but glmocr defaults to
        # Zhipu's hosted MaaS service and raises MissingApiKeyError in its
        # constructor; the self-hosted switch is not the kwargs its docs imply.
        # Routing four legs through it took glm from its baseline
        # (text 0.894, markdown 0.524, tables 0.596, schema 0.820 over 93 docs)
        # to errors on everything but text. The 2026-10-02 source audit now
        # reproduces the cause: ignored constructor kwargs. The pinned SDK
        # probe uses mode="selfhosted" plus ocr_api_host/ocr_api_port. Keep
        # direct vLLM routing until the controlled pipeline comparison passes.
        # key_value was dropped because the documented KIE task requires the
        # caller to state the target structure. RE-ENABLED 2026-09-30 by
        # decision (interface rule: it takes a prompt, so it competes). The
        # ask is the generic key_value prompt from evaluation/legs.py — the
        # card's KIE prompt needs a structure and this leg gives none. The
        # reference run 20260810T130609Z has no key_value rows for this
        # engine; the next run will.
        prompted_legs=("markdown", "tables", "key_value", "schema"),
        table_format="markdown",
        non_json_legs=("markdown", "tables"),
        # The card's information-extraction example is a filled JSON object,
        # not a JSON Schema document. Handed the schema document the model
        # echoes it back and scores 0.000; handed an empty instance of it, it
        # fills every field. Verified 2026-08-09 on invoice_000.
        schema_render="instance_template",
        leg_notes="card documents Text/Table/Formula Recognition and an "
                  "extraction prompt — no coordinate task exists to ask for",
        leg_prompts={
            # Model card (huggingface.co/zai-org/GLM-OCR) documents four task
            # prefixes. Using the right one per leg matters: this model is a task-
            # prefixed OCR fine-tune, not a general instruction follower.
            "text": "Text Recognition:",
            # The card documents no markdown task, so this generic ask is
            # undocumented and flagged as such — but it is the engine's best
            # available shot and it produced real results (0.524 over 57 docs).
            # The SDK path is repaired locally but its controlled quality
            # comparison has not run; see the note above.
            "tables": "Table Recognition:",
            # The card's own information-extraction prompt, given in Chinese, and
            # it states the output must strictly follow the supplied JSON schema.
            "schema": "请按下列JSON格式输出图中信息:\n\n{schema}",
            # No key_value override: the card's prompt above presupposes a
            # supplied format ("output in the following JSON format:") and
            # the key_value leg supplies none, so the generic leg ask applies.
        },
    ),
    "paddleocr_vl": EngineSpec(
        name="paddleocr_vl", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.001411, cost_notes="Modal A100-80GB; measured batch estimate",
        modal_app="ocr-bench-paddleocr-vl", modal_class="PaddleOCRVLInference",
        modal_app_file="modal_apps/paddleocr_vl_app.py",
        gpu_type="A100-80GB", batch_size=24,
        # geometry added 2026-08-10: the pipeline natively returns
        # overall_ocr_res.rec_texts paired with rec_boxes/rec_polys, plus
        # block_bbox per layout block. It was marked ineligible only because the
        # app reduced all of that to a block count. Same single page call.
        native_legs=("text", "markdown", "tables", "geometry"),
        leg_notes="fixed pipeline, no prompt interface for extraction legs "
                  "(PP-ChatOCR is the separate Paddle product for those)",
        table_format="html",
        leg_prompts={
            # Our deployment runs the page-level doc_parser PIPELINE, which takes
            # no prompt (modal_apps/paddleocr_vl_app.py exposes no prompt param).
            # Its raw_text IS the pipeline's own markdown rendering
            # (res.markdown["markdown_texts"]), with tables embedded as HTML —
            # which the markdown scorer normalises, so text, markdown and tables
            # are all scored from this single default output. The model card's
            # element-level prefixes ("OCR:", "Table Recognition:") apply to raw
            # transformers usage, not to this pipeline, and are not sent.
            # key_value/schema stay out: the pipeline has no extraction interface
            # (PP-ChatOCR is a separate product pairing it with an external LLM).
            #
            # THE ASK FOR THIS ENGINE IS ITS PIPELINE CONFIGURATION, not a prompt.
            # Capability flags default to False in paddleocr, so accepting the
            # defaults silently narrows what the model is allowed to do — the
            # equivalent of sending a worse prompt. Currently set in
            # modal_apps/paddleocr_vl_app.py:
            #   use_chart_recognition=True  — was default-False, which made the
            #     layout stage label each chart a figure, crop it to
            #     imgs/img_in_chart_box_*.jpg and drop it; the crop is never
            #     returned, so the whole region was lost on all 5 chart
            #     documents. Fires only on chart-labelled regions (the corpus
            #     has 6, all in chart_000-004), so other pages are unaffected.
            #   use_seal_recognition=True — also enabled in the app (an
            #     earlier version of this comment said it was still off).
            #     sroie_002 contains a seal region (img_in_seal_box).
            # Still at their defaults, and each one narrows the ask:
            #   use_doc_orientation_classify=False, use_doc_unwarping=False —
            #     the corpus contains photographed pages (bank_*, photo_receipt_*,
            #     scanned_*) where a real deployment would enable both.
            # Geometry caveat (found 2026-09-30 from the saved outputs): the
            # note above says rec_texts/rec_boxes are returned, but PaddleX 3.7's
            # res.json has no overall_ocr_res, so none of the 93 saved v1
            # outputs carries text_regions and the geometry leg scored this
            # engine from its raw markdown text (unparseable -> 0.0). The v1
            # app is left as it ran; paddleocr_vl_1_6 pairs layout-block boxes
            # with their VLM text instead.
        },
    ),
    "marker": EngineSpec(
        name="marker", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.001420, cost_notes="Modal A100-80GB; measured batch estimate",
        modal_app="ocr-bench-marker", modal_class="MarkerInference",
        modal_app_file="modal_apps/marker_app.py",
        gpu_type="A100-80GB", batch_size=24, supports_pdf=True,
        accepts_task=True,
        # Marker 1.8.2 ships a converter per task, so this is not one fixed
        # pipeline. Treating it as one meant text was scored from flattened
        # document markdown instead of OCRConverter, tables were scraped out of
        # that markdown instead of TableConverter, and geometry was marked
        # ineligible although every JSON block carries a polygon.
        native_legs=(),
        prompted_legs=("text", "markdown", "tables", "geometry"),
        # Text and markdown share the document converter; geometry uses the
        # separate OCR converter. These are converter selectors, not prompts.
        non_json_legs=("text", "markdown", "tables", "geometry"),
        leg_notes="structured extraction excluded: Marker's ExtractionConverter "
                  "requires an external LLM, which would make this a hybrid engine",
        table_format="markdown",
        leg_prompts={
            # Sentinels selecting a converter — see modal_apps/marker_app.py.
            # Preserve the measured text/document route. The 2026-10-02
            # pinned-source audit found the walker handles OCRJSONRenderer's
            # children correctly; sparse output needs a pipeline probe, not
            # an assumed tree-parser fix. See the handoff.
            "text": "__marker_pdf_converter__",
            "markdown": "__marker_pdf_converter__",
            # Native OCR line polygons are the geometry evidence. The
            # explicit all-blocks probe is not silently selected here.
            "geometry": "__marker_ocr_converter__",
            "tables": "__marker_table_converter__",
        },
    ),
    "deepseek_ocr": EngineSpec(
        name="deepseek_ocr", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.000975, cost_notes="Modal A100-80GB; measured batch estimate",
        modal_app="ocr-bench-deepseek-ocr", modal_class="DeepSeekOCRInference",
        modal_app_file="modal_apps/deepseek_ocr_app.py",
        gpu_type="A100-80GB", batch_size=24,
        accepts_task=True,
        native_legs=("text",),
        # The card documents no extraction prompt, and a 2026-08-09
        # prompt-sensitivity check over four phrasings (grounding prefix with a
        # schema, the same without it, a plain "return JSON" instruction, and a
        # label->value ask) produced an empty string, an empty string, "No line
        # breaks detected.", and a markdown conversion of the page. The model
        # can be prompted, so it competes and whatever it scores is its score —
        # what the check bought is the certainty that a zero here is the model
        # failing, not us asking badly.
        # key_value dropped 2026-08-10: no documented extraction prompt, no
        # instruction/chat training stage per the authors, and two phrasings
        # over 65 documents never returned JSON. RE-ENABLED 2026-09-30 by
        # decision: the model takes a prompt, so under the interface rule in
        # docs/scoring-design.md it competes and its score is its score —
        # withdrawing it while keeping schema (where it does return JSON on
        # 59 of 65 pages) was inconsistent. The reference run 20260810T130609Z
        # has no key_value rows for this engine; the next run will.
        prompted_legs=("markdown", "tables", "key_value", "schema", "geometry"),
        # The grounding conversion emits real <table> markup. Declared "tsv"
        # until 2026-08-09, which sent the HTML through the tab-splitter and
        # scored every table zero — the app's own cleaner only flattens tables
        # to text on the plain-text pass, and the markdown pass keeps them.
        table_format="html",
        # Geometry is answered by the grounding pass's own markup, not JSON —
        # which also lets it share that one call with markdown and tables.
        non_json_legs=("geometry",),
        leg_notes="OCR fine-tune; the card documents no structured-extraction "
                  "prompt, so the extraction legs use our own ask",
        leg_prompts={
            # Model card (huggingface.co/deepseek-ai/DeepSeek-OCR) documents two
            # prompts. "Free OCR." is the plain-text task; the grounding prompt is
            # for markdown conversion. The image is supplied as a separate message
            # part by the app, so the card's literal "<image>\n" prefix is omitted.
            "text": "Free OCR.",
            "markdown": "<|grounding|>Convert the document to markdown.",
            # Tables are part of the documented document-parsing task, not a
            # missing one: the training data contains HTML-formatted tables,
            # OmniDocBench scores DeepSeek on a dedicated table metric, and the
            # official inference config special-cases the <td>/</td> tokens in
            # its repetition control — which this app inherits verbatim as
            # whitelist_token_ids=[128821, 128822]. So the grounding conversion
            # IS the right ask for a printed table, and it shares markdown's
            # call rather than paying twice. (An earlier note here read "no
            # table task documented", which was misleading.)
            "tables": "<|grounding|>Convert the document to markdown.",
            # Geometry rides the SAME grounding pass — no third call. Grounding
            # mode annotates each region it reads as
            #   <|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
            # with coordinates normalised 0-999. The app's cleaner deleted those
            # spans, which is why the registry used to claim this model had no
            # coordinate task; it has one, and we were discarding its output.
            "geometry": "<|grounding|>Convert the document to markdown.",
            # NOT on the card — it documents no extraction prompt at all. Kept so
            # the engine competes rather than being excluded by us.
            #
            # No <|grounding|> prefix. The repo README documents eight prompts
            # (github.com/deepseek-ai/DeepSeek-OCR) and uses that token on ONLY
            # the two layout tasks — "Convert the document to markdown." and
            # "OCR this image." The non-layout ones ("Parse the figure.",
            # "Describe this image in detail.") carry no prefix. Sending it with
            # a JSON instruction put the ask off-distribution twice and told the
            # model to do grounded OCR; 58 of 65 schema replies came back empty.
            # Whatever it scores without the prefix is a fair result; with it,
            # we were measuring our own ask.
            "schema": "Extract the values from this document as JSON "
                      "conforming exactly to this schema. Output only JSON.\n\n{schema}",
        },
    ),
    "qwen3_vl": EngineSpec(
        name="qwen3_vl", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.004113, cost_notes="Modal A100-80GB; measured batch estimate",
        modal_app="ocr-bench-qwen3-vl", modal_class="Qwen3VLInference",
        modal_app_file="modal_apps/qwen3_vl_app.py",
        gpu_type="A100-80GB", batch_size=24,
        accepts_task=True,
        # No native pass: reading the text and formatting the document are two
        # different documented Qwen tasks, so they get separate calls. Sharing
        # one made the text leg measure how much text survived a markdown
        # conversion rather than what the model reads.
        native_legs=(),
        prompted_legs=("text", "markdown", "tables", "key_value", "schema",
                       "geometry"),
        # geometry is excluded from JSON enforcement on purpose: the documented
        # spotting answer is a JSON ARRAY, and vLLM's json_object mode
        # constrains generation to an object, which would break the very format
        # the prompt asks for. The array is parsed from the text instead.
        non_json_legs=("text", "markdown", "tables", "geometry"),
        # A general VLM like gemini and claude, and its conversion emits pipe
        # tables. Left at None it was classed a flat-text engine, so its tables
        # were never extracted and it scored 0.000 while returning a correct
        # markdown table.
        table_format="markdown",
        leg_prompts={
            # The bare model card is sparse, but the Qwen3-VL repo ships
            # cookbooks for OCR/KIE, document parsing and 2D grounding. Two legs
            # have a documented ask and were previously using our generic one.
            #
            # Direct OCR, per the official vLLM Qwen3-VL recipe. Previously the
            # text leg was scored from the MARKDOWN pass, which measures how
            # much text survives a formatting task rather than what the model
            # reads.
            "text": "Read all the text in the image.",
            # tables is given markdown's exact wording so the two resolve to one
            # ask and the planner folds them into a single call — the table is
            # part of the document formatting, and Qwen documents no separate
            # table task.
            "tables": (
                "Convert this document to Markdown. Preserve reading order, "
                "headings, lists and tables. Output only the Markdown."
            ),
            # markdown/tables keep the generic instruction: Qwen's native
            # documented document representation is Qwen HTML with layout
            # positions, so there is no official markdown ask, and a general
            # instruction-following VLM is legitimately asked this way.
            #
            # Word-level text spotting, verbatim from the OCR cookbook. Its
            # bbox_2d values are RELATIVE 0-1000, not pixels — the app divides
            # by 1000 before handing them to the scorer.
            "geometry": (
                "Spotting all the text in the image with word-level, and output "
                "in JSON format as [{'bbox_2d': [x1, y1, x2, y2], "
                "'text_content': 'text'}, ...]."
            ),
        },
    ),
    # =======================================================================
    # Expansion, 2026-09-30. Every engine below is default=False: none has
    # been run yet, so none has a measured cost, and none is part of the
    # reference run 20260810T130609Z. They are opted into with --engines and
    # collected into an extension run (--collect-only --extends <baseline>).
    # Integration guide: docs/engine-integrations.md; contracts live as
    # comments at the top of each app/runner file.
    # =======================================================================
    # --- Cloud APIs (expansion) ----------------------------------------------
    "mistral_ocr_4_1": EngineSpec(
        name="mistral_ocr_4_1", kind="cloud",
        runner="runners.mistral_ocr_4_1_runner:MistralOCR41Runner",
        required_env=("MISTRAL_API_KEY",),
        cost_per_page=0.004,
        cost_notes="$4/1K OCR pages, $5/1K annotated (docs.mistral.ai/models/ocr-4-1, "
                   "2026-09-30); billed from usage_info.pages_processed",
        default=False,
        accepts_task=True,
        # A distinct row from mistral_ocr_4: the saved baseline records
        # `mistral-ocr-latest` for every call, and per Mistral's changelog
        # `mistral-ocr-4-0` retires on 2026-09-30, so the old row cannot be
        # re-run as what it was. This row pins `mistral-ocr-4-1` and requests
        # HTML tables + include_blocks; the runner re-inserts the returned
        # table HTML at each placeholder so text/markdown keep the cells.
        # geometry: blocks are paragraph boxes (no word boxes exist) ->
        # text_regions, granularity "block".
        native_legs=("text", "markdown", "tables", "geometry"),
        # schema via the vendor's own Annotations feature, like mistral_ocr_4.
        prompted_legs=("schema",),
        table_format="html",
        leg_notes="OCR endpoint takes no prompt; geometry is paragraph/block "
                  "boxes; key_value excluded: annotation needs a caller-supplied "
                  "json_schema, there is no label-discovery contract",
    ),
    "google_document_ocr": EngineSpec(
        name="google_document_ocr", kind="cloud",
        runner="runners.google_document_ocr_runner:GoogleDocumentOCRRunner",
        required_env=("GOOGLE_DOCAI_PROJECT", "GOOGLE_DOCAI_LOCATION",
                      "GOOGLE_DOCAI_PROCESSOR_ID"),
        cost_per_page=0.0015,
        cost_notes="Enterprise Document OCR $1.50/1K pages (cloud.google.com/"
                   "document-ai/pricing, 2026-09-30); premium add-ons off; "
                   "first 1K pages/month free",
        default=False,
        # Document AI tokens are genuine word boxes (normalized_vertices) ->
        # "words". The runner refuses a processor whose type is not OCR so
        # Form Parser / Layout Parser output can never be attributed here.
        native_legs=("text", "geometry"),
        leg_notes="OCR processor only: no markdown, tables or extraction "
                  "(Form Parser and Layout Parser are separate products)",
    ),
    "mathpix": EngineSpec(
        name="mathpix", kind="cloud",
        runner="runners.mathpix_runner:MathpixRunner",
        required_env=("MATHPIX_APP_ID", "MATHPIX_APP_KEY"),
        cost_per_page=0.002,
        cost_notes="$0.002/image, >12 text rows may bill $0.005 (mathpix.com/"
                   "pricing/api, 2026-09-30); geometry is a second billed call",
        default=False,
        accepts_task=True,
        native_legs=("text", "markdown", "tables"),
        # Word boxes need their own call: include_word_data cannot be combined
        # with enable_document_layout (docs.mathpix.com/reference/post-v3-text),
        # and the default pass uses layout mode. The "prompt" below is a
        # configuration selector the runner switches on, never sent to
        # Mathpix — the same device marker uses for its converters.
        prompted_legs=("geometry",),
        non_json_legs=("geometry",),
        # raw_text is Mathpix Markdown; tables are read from the `data`
        # items the runner stores in structured_output["tables"], so no
        # text-format converter is declared (same as textract_structured).
        table_format=None,
        leg_prompts={
            "geometry": "__mathpix_word_data__",
        },
        leg_notes="no prompt/extraction interface; geometry is a separately "
                  "configured word_data call. Three corpus images exceed the "
                  "5 MB request limit and will error until a policy is chosen",
    ),
    "datalab_ocr": EngineSpec(
        name="datalab_ocr", kind="cloud",
        runner="runners.datalab_runner:DatalabRunner",
        required_env=("DATALAB_API_KEY",),
        cost_per_page=0.01,
        cost_notes="list Convert balanced $4/1K pages, rounded UP per request to "
                   "the cent = $0.01 per single-image request (documentation."
                   "datalab.to/platform/billing, 2026-09-30); schema via Extract "
                   "$15/1K + parsing; runs record the returned cost_breakdown",
        default=False, supports_pdf=True,
        # schema only, through the vendor's Extract API (a documented
        # structured path, like Mistral Annotations). Any other task raises.
        accepts_task=True,
        # geometry deliberately NOT declared yet: the JSON tree's block boxes
        # are parsed into text_regions, but the shape is unverified until two
        # paid conversions are inspected. Add "geometry" after that probe.
        native_legs=("text", "markdown", "tables"),
        prompted_legs=("schema",),
        table_format="markdown",
        leg_notes="Convert takes no prompt; schema via Datalab Extract; "
                  "key_value not offered (Extract needs a schema, and inventing "
                  "one would leak or fake the ask); word boxes are a paid add-on "
                  "left off; hosted model versions cannot be pinned",
    ),
    # --- Self-hosted (expansion) --------------------------------------------
    "paddleocr_vl_1_6": EngineSpec(
        name="paddleocr_vl_1_6", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal A10 GPU-seconds (fit probe; L40S next). "
                   "Cost/page is comparable to paddleocr_vl (billed per actual "
                   "GPU); seconds/page are not (A100 vs A10)",
        modal_app="ocr-bench-paddleocr-vl-1-6", modal_class="PaddleOCRVL16Inference",
        modal_app_file="modal_apps/paddleocr_vl_1_6_app.py",
        gpu_type="A10", batch_size=24,
        default=False,
        # A separate row, never a MODEL_ID swap on paddleocr_vl: the saved
        # baseline is v1 (PP-DocLayoutV2 + PaddleOCR-VL-0.9B @7fa00a8c). This
        # is the full v1.6 pipeline (PP-DocLayoutV3 + PaddleOCR-VL-1.6-0.9B),
        # both artifacts pinned by revision and hash in the app.
        accepts_task=False,
        # geometry is BLOCK granularity: each layout box paired with the VLM
        # text of that crop (PaddleX 3.7 returns no overall_ocr_res). v1.6's
        # line-level "spotting" is a separate use_layout_detection=False pass,
        # i.e. a different configuration, not run under this row.
        native_legs=("text", "markdown", "tables", "geometry"),
        table_format="html",
        leg_notes="fixed v1.6 pipeline, no extraction interface; geometry is "
                  "layout-block boxes, not lines or words",
        leg_prompts={
            # None: the pipeline owns its per-region prompts (paddlex 3.7.2
            # paddleocr_vl/pipeline.py); the app raises on a caller prompt.
        },
    ),
    "opendoc": EngineSpec(
        name="opendoc", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal CPU container (4 cores + 8 GiB) per "
                   "occupied second; an L4 probe is a one-line GPU_TYPE change",
        modal_app="ocr-bench-opendoc", modal_class="OpenDocInference",
        modal_app_file="modal_apps/opendoc_app.py",
        # openocr-python 0.1.5's task='doc' is an ONNX Runtime CPU pipeline
        # (PP-DocLayoutV2 + UniRec-0.1B); its decoder is a per-token Python
        # loop, so CPU is the honest first baseline. Pages run sequentially,
        # so the outer batch is kept small to stay inside the call timeout.
        gpu_type="CPU", batch_size=8,
        default=False,
        accepts_task=False,
        native_legs=("text", "markdown", "tables", "geometry"),
        table_format="html",
        leg_notes="fixed OpenOCR doc pipeline; UniRec takes no prompt; geometry "
                  "is layout-block boxes; upstream drops charts from markdown",
    ),
    "deepseek_ocr_2": EngineSpec(
        name="deepseek_ocr_2", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal L4 GPU-seconds, 4 active requests",
        modal_app="ocr-bench-deepseek-ocr-2", modal_class="DeepSeekOCR2Inference",
        modal_app_file="modal_apps/deepseek_ocr_2_app.py",
        gpu_type="L4", batch_size=24,
        default=False,
        accepts_task=True,
        native_legs=("text",),
        # key_value POLICY (decided 2026-09-30, not inherited from v1): the
        # app forwards any caller prompt, so by the interface-based rule in
        # docs/scoring-design.md the engine competes and whatever it scores
        # is its score. v1 dropped key_value after 0/65 JSON replies; v2 is a
        # different checkpoint (retrained decoder), so that is a prior, not a
        # measurement. The paper describes no instruction/chat stage, so the
        # expectation is low — that is a result to publish, not a reason to
        # decline the ask. 65 extra passes on L4.
        prompted_legs=("markdown", "tables", "key_value", "schema", "geometry"),
        # Geometry rides the grounding pass (markup, not JSON) and so merges
        # with markdown/tables into one call, as for v1.
        non_json_legs=("geometry",),
        table_format="html",
        leg_notes="OCR model with two documented prompts; extraction legs use "
                  "the generic asks. Geometry = grounding boxes with the text "
                  "that follows each tag (block granularity). The v1 app's "
                  "'Parse the figure.' second stage is not documented for v2 "
                  "and is not run",
        leg_prompts={
            # huggingface.co/deepseek-ai/DeepSeek-OCR-2 "Main Prompts" and
            # DeepSeek-OCR2-vllm/config.py @2f3699eb. The image is a separate
            # message part; vLLM's deepseek_ocr2 template renders "<image>\n".
            "text": "Free OCR.",
            "markdown": "<|grounding|>Convert the document to markdown.",
            # Same documented ask -> one pass for markdown+tables+geometry.
            "tables": "<|grounding|>Convert the document to markdown.",
            "geometry": "<|grounding|>Convert the document to markdown.",
            # key_value / schema: no documented prompt -> generic leg prompts.
        },
    ),
    "dots_mocr": EngineSpec(
        name="dots_mocr", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal L4 GPU-seconds, 4 active requests",
        modal_app="ocr-bench-dots-mocr", modal_class="DotsMOCRInference",
        modal_app_file="modal_apps/dots_mocr_app.py",
        gpu_type="L4", batch_size=24,
        default=False,
        accepts_task=True,
        # Default pass = the author's prompt_layout_all_en: ordered layout
        # JSON (Markdown text, HTML tables, LaTeX, pixel boxes) that the app
        # converts to Markdown + text_regions (block granularity). Headers and
        # footers are kept; the author's "nohf" deletion is not used.
        native_legs=("text", "markdown", "tables", "geometry"),
        # The card documents general QA ("interactive dialogue", DocVQA); any
        # non-layout prompt goes through the author's prompt_general route
        # with its required system prompt. Generic leg asks; no extraction
        # prompt is documented.
        prompted_legs=("key_value", "schema"),
        table_format="html",
        leg_notes="layout-JSON conversion for the default pass; extraction via "
                  "the documented general-QA route with the generic asks",
        leg_prompts={
            # Default pass sends no prompt: the app uses prompt_layout_all_en
            # verbatim from dots_mocr/utils/prompts.py @23f3e561
            # (github.com/studio-dots-ai/dots.mocr).
        },
    ),
    "surya_ocr_2": EngineSpec(
        name="surya_ocr_2", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal L4 GPU-seconds; Surya's own client pool "
                   "(SURYA_INFERENCE_PARALLEL=8) feeds vLLM",
        modal_app="ocr-bench-surya-ocr-2", modal_class="SuryaOCR2Inference",
        modal_app_file="modal_apps/surya_ocr_2_app.py",
        gpu_type="L4", batch_size=24,
        default=False,
        accepts_task=True,
        # Default = the official full-page RecognitionPredictor path (the
        # surya_ocr CLI): one HIGH_ACCURACY_BBOX_PROMPT per page returning
        # <div data-label data-bbox> HTML blocks -> text/markdown/tables plus
        # block geometry. Layout+block mode is a later probe.
        native_legs=("text", "markdown", "tables", "geometry"),
        # BatchInputItem.prompt + guided_json are forwarded by
        # surya/inference/backends/openai_client.py (v0.22.1).
        prompted_legs=("key_value", "schema"),
        table_format="html",
        leg_notes="geometry = layout-block boxes; markdown is markdownify over "
                  "Surya's block HTML (Surya emits HTML only); extraction "
                  "prompts are outside the model's four trained prompts",
        leg_prompts={
            # No extraction prompt documented (surya/inference/prompts.py
            # lists layout/block/table_rec/high_accuracy_bbox) -> generic asks.
        },
    ),
    "chandra_ocr_2": EngineSpec(
        name="chandra_ocr_2", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal A10 GPU-seconds at active concurrency 1 "
                   "(5.3B weights = 10.6 GB bf16; fit probe first)",
        modal_app="ocr-bench-chandra-ocr-2", modal_class="ChandraOCR2Inference",
        modal_app_file="modal_apps/chandra_ocr_2_app.py",
        # Concurrency 1 for the fit probe; raise the outer batch with the sweep.
        gpu_type="A10", batch_size=8,
        default=False,
        accepts_task=True,
        native_legs=("text", "markdown", "tables", "geometry"),
        # BatchInputItem.prompt is forwarded by chandra/model/vllm.py (v0.2.0),
        # but that route sends no response_format, so extraction runs WITHOUT
        # JSON-constrained decoding (recorded per item as
        # response_format_applied=False). Not equivalent to Surya's route.
        prompted_legs=("key_value", "schema"),
        table_format="html",
        leg_notes="geometry = ocr_layout block boxes; image/figure descriptions "
                  "kept separately and excluded from geometry; extraction runs "
                  "without constrained decoding",
        leg_prompts={
            # Chandra documents only ocr_layout/ocr (chandra/prompts.py) ->
            # generic extraction prompts from evaluation/legs.py.
        },
    ),
    "mineru_4_standard": EngineSpec(
        name="mineru_4_standard", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal A10 fit probe, whole Standard pipeline "
                   "(Torch small models + vLLM VLM) on one GPU; L40S next",
        modal_app="ocr-bench-mineru-4-standard", modal_class="MinerU4StandardInference",
        modal_app_file="modal_apps/mineru_4_standard_app.py",
        gpu_type="A10", batch_size=24,
        default=False,
        # A different pipeline from the `mineru` row (MinerU2.5-2509-1.2B via
        # mineru-vl-utils two_step_extract): mineru 4.0.10, tier=standard,
        # PP-DocLayoutV2 layout + MinerU2.5-Pro-2605-1.2B recognition + the
        # PP-OCRv6/formula small models, all pinned by revision in the app.
        # Licence: MinerU code is under the custom "MinerU Open Source
        # License" (Apache-2.0 plus a commercial threshold and an attribution
        # duty); weights are Apache-2.0.
        accepts_task=False,
        # raw_text is FULL-mode Markdown (keeps header/footer/page_number,
        # which the DEFAULT view drops); tables from table_body HTML;
        # geometry from MiddleJson block bbox+content (block granularity).
        native_legs=("text", "markdown", "tables", "geometry"),
        table_format="html",
        leg_notes="fixed pipeline (tier=standard), no caller prompt; geometry "
                  "is block-level; charts/images are analysed only in the "
                  "Advanced tier, not here",
        leg_prompts={
            # No caller prompts. Region prompts ("\nText Recognition:",
            # "\nTable Recognition:", "\nFormula Recognition:") are internal
            # to mineru-vl-utils 2.0.5 DEFAULT_PROMPTS.
        },
    ),
    "teleocr": EngineSpec(
        name="teleocr", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal A10 probe (then L4, L40S); one 1.2B VLM "
                   "does both layout and crop recognition",
        modal_app="ocr-bench-teleocr", modal_class="TeleOCRInference",
        modal_app_file="modal_apps/teleocr_app.py",
        gpu_type="A10", batch_size=24,
        default=False,
        # The official pipeline: the VLM is asked "\nAnalyze the image
        # layout." on a 1036x1036 render, then each region crop gets a fixed
        # per-type prompt (text/table/equation/code/seal/chart) and OTSL
        # tables are converted to HTML. There is no whole-document
        # caller-prompt route, so no extraction legs.
        accepts_task=False,
        native_legs=("text", "markdown", "tables", "geometry"),
        table_format="html",
        leg_notes="fixed layout->crop pipeline (LAYOUT_MODE=Detection, "
                  "vllm-async-engine); no caller prompt; geometry is "
                  "block-level layout regions (model 0-1000 boxes / 1000)",
        leg_prompts={
            # No caller prompts. Layout and per-type crop prompts are internal:
            # github.com/caipeng328/TeleOCR@1e71f4fe
            # TeleOCR/vlm_utils/TeleOCR_client.py LAYOUT_PROMPTS/DEFAULT_PROMPTS.
        },
    ),
    "rapidocr": EngineSpec(
        name="rapidocr", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal CPU container (4 cores + 8 GiB) per "
                   "occupied second — not a GPU rate",
        modal_app="ocr-bench-rapidocr", modal_class="RapidOCRInference",
        modal_app_file="modal_apps/rapidocr_app.py",
        # "CPU" is a real rate in modal_apps/_shared.py, not a GPU. It runs on
        # Modal rather than this machine because no model inference runs
        # locally in this project.
        gpu_type="CPU", batch_size=24,
        default=False,
        # Native word results (CTC column positions inside each detected line)
        # -> "words" when every line has them, else line boxes -> text_regions.
        native_legs=("text", "geometry"),
        leg_notes="detection+recognition toolkit: flat lines, no markdown or "
                  "table structure, no prompt",
    ),
    # Added 2026-10-01 for geometry: Paddle's and MinerU's OTHER pipelines, the
    # ones with a text-line detector. paddleocr_vl and mineru (the VLM
    # backends) stay as they are; their finest geometry is a layout block.
    "paddleocr_ppocrv5": EngineSpec(
        name="paddleocr_ppocrv5", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal A10 GPU-seconds, sequential pages",
        modal_app="ocr-bench-paddleocr-ppocrv5", modal_class="PaddleOCRv5Inference",
        modal_app_file="modal_apps/paddleocr_ppocrv5_app.py",
        gpu_type="A10", batch_size=24,
        default=False,
        # PP-OCRv5_server_det polygons per text line -> text_regions (line);
        # words when the pipeline's return_word_box fields are present.
        native_legs=("text", "geometry"),
        leg_notes="PP-OCRv5 det+rec (text-line OCR, no layout): flat lines, "
                  "no markdown or table structure, no prompt",
    ),
    "mineru_pipeline": EngineSpec(
        name="mineru_pipeline", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal T4 GPU-seconds, one pipeline_doc_analyze "
                   "call per chunk",
        modal_app="ocr-bench-mineru-pipeline", modal_class="MinerUPipelineInference",
        modal_app_file="modal_apps/mineru_pipeline_app.py",
        gpu_type="T4", batch_size=24,
        default=False,
        # middle.json lines (bbox + span text) -> text_regions (line);
        # markdown and HTML tables come from the same pass.
        native_legs=("text", "markdown", "tables", "geometry"),
        table_format="html",
        leg_notes="MinerU 2.5 pipeline backend (layout + text-line OCR + "
                  "formula + table models), lang=en, method=ocr; no prompt",
    ),
    "ovisocr2": EngineSpec(
        name="ovisocr2", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal A10 GPU-seconds from one batched "
                   "LLM.generate per call",
        modal_app="ocr-bench-ovisocr2", modal_class="OvisOCR2Inference",
        modal_app_file="modal_apps/ovisocr2_app.py",
        gpu_type="A10", batch_size=24,
        default=False,
        # The app forwards a caller prompt through the general Qwen3.5 chat
        # template (chat_template.jinja @ 1fc9221b). The authors claim no
        # instruction following beyond the card's one OCR prompt, so the
        # extraction legs are measured as candidates — whatever they score.
        accepts_task=True,
        # The card's single conversion prompt (the app default) yields
        # Markdown with HTML tables: text/markdown/tables from one pass.
        native_legs=("text", "markdown", "tables"),
        # No documented extraction prompt: the generic leg asks are sent.
        # No geometry: neither the card nor the report (arxiv 2607.13639)
        # documents text grounding; its <img> bbox tags locate figures only.
        # (The leg matrix lists geometry as a prompted candidate — that would
        # be an invented ask, so it is not declared.)
        prompted_legs=("key_value", "schema"),
        table_format="html",
        # UNVERIFIED which render this model fills better; GLM needed the
        # instance template. Compare on the first schema probe.
        schema_render="json_schema",
        leg_notes="0.8B page-to-Markdown model with one documented prompt; "
                  "extraction legs use the generic asks through the Qwen3.5 "
                  "chat template; img-tag boxes locate figures, not text",
        leg_prompts={
            # None: the conversion prompt is the app's default, copied
            # verbatim from huggingface.co/ATH-MaaS/OvisOCR2 (README @
            # 1fc9221b); the card documents no other prompt.
        },
    ),
    "lighton_ocr_2": EngineSpec(
        name="lighton_ocr_2", kind="modal",
        runner="runners.modal_runner:ModalRunner",
        cost_per_page=0.0,
        cost_notes="unmeasured; Modal A10 GPU-seconds, vLLM server, 16 active "
                   "requests",
        modal_app="ocr-bench-lighton-ocr-2", modal_class="LightOnOCR2Inference",
        modal_app_file="modal_apps/lighton_ocr_2_app.py",
        gpu_type="A10", batch_size=24,
        default=False,
        # Image-only by design ("OCR without task prompts at inference time",
        # arxiv 2601.14251); the app raises on any prompt rather than
        # ignoring it.
        accepts_task=False,
        native_legs=("text", "markdown", "tables"),
        table_format="html",  # paper: HTML targets for tables
        leg_notes="image-only OCR model: no prompt route, so no key_value/"
                  "schema; the bbox variants box embedded images only, so no "
                  "geometry",
    ),
    # --- Local (expansion; opt-in) -------------------------------------------
    "apple_vision": EngineSpec(
        name="apple_vision", kind="local",
        runner="runners.apple_vision_runner:AppleVisionRunner",
        cost_notes="on-device Apple Vision (Neural Engine), no API charge. "
                   "OPT-IN: it is local inference on this Mac",
        default=False,
        # VNRecognizeTextRequest lines plus per-word boxes from
        # boundingBox(for: wordRange) -> "words".
        native_legs=("text", "geometry"),
        leg_notes="lines + word boxes only; RecognizeDocumentsRequest (tables) "
                  "needs macOS 26, this host is 15.6",
    ),
}

# Derived views — import these instead of re-declaring engine lists anywhere.
DEFAULT_ENGINES: list[str] = [n for n, s in ENGINES.items() if s.default]
MODAL_ENGINE_NAMES: frozenset[str] = frozenset(
    n for n, s in ENGINES.items() if s.kind == "modal"
)
LOCAL_CPU_ENGINES: frozenset[str] = frozenset(
    n for n, s in ENGINES.items() if s.kind == "local"
)


def engines_for_leg(leg: str, engines: list[str] | None = None) -> list[str]:
    """Engine names eligible for `leg`, in registry order."""
    names = engines if engines is not None else list(ENGINES)
    return [n for n in names if n in ENGINES and ENGINES[n].answers(leg)]


def build_runner(engine: str):
    """Instantiate the runner for an engine name (lazy imports, so a missing
    optional dependency only breaks the engine that needs it)."""
    spec = ENGINES.get(engine)
    if spec is None:
        raise ValueError(f"Unknown engine: '{engine}'. Known: {', '.join(ENGINES)}")
    module_name, _, class_name = spec.runner.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    if spec.kind == "modal":
        return cls(engine)
    return cls()
