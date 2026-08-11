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
        native_legs=("text", "tables", "key_value", "schema", "geometry"),
        leg_notes=(
            "no prompt; schema leg judged by LLM against FORMS output "
            "(same judge as all other engines)"
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
        native_legs=_CONVERSION_LEGS,
        prompted_legs=("key_value", "schema", "geometry"),
        table_format="markdown",
    ),
    "claude_sonnet_4_6": EngineSpec(
        name="claude_sonnet_4_6", kind="cloud",
        runner="runners.claude_runner:ClaudeRunner",
        required_env=("ANTHROPIC_API_KEY",),
        cost_per_page=0.012,
        cost_notes="variable — token-based",
        accepts_task=True,
        native_legs=_CONVERSION_LEGS,
        prompted_legs=("key_value", "schema", "geometry"),
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
        # to errors on everything but text. Back to direct vLLM calls with the
        # card's task prefixes until the correct self-hosted switch is read off
        # the installed package.
        # key_value stays dropped: the documented KIE task requires the caller
        # to state the target structure, so generic "find every label" is not a
        # GLM task, and supplying the keys duplicates the schema leg.
        prompted_legs=("markdown", "tables", "schema"),
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
            # The SDK path that would replace it is blocked on glmocr's
            # self-hosted switch; see the note above.
            "tables": "Table Recognition:",
            # The card's own information-extraction prompt, given in Chinese, and
            # it states the output must strictly follow the supplied JSON schema.
            "schema": "请按下列JSON格式输出图中信息:\n\n{schema}",
            "key_value": "请按下列JSON格式输出图中信息:",
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
            # Still at their defaults, and each one narrows the ask:
            #   use_doc_orientation_classify=False, use_doc_unwarping=False —
            #     the corpus contains photographed pages (bank_*, photo_receipt_*,
            #     scanned_*) where a real deployment would enable both.
            #   use_seal_recognition=False — sroie_002 contains a seal region
            #     (img_in_seal_box), currently cropped and discarded.
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
        # markdown and geometry share one pass: build_document() runs the
        # pipeline once and markdown/JSON are two renderings of that result.
        non_json_legs=("text", "markdown", "tables", "geometry"),
        leg_notes="structured extraction excluded: Marker's ExtractionConverter "
                  "requires an external LLM, which would make this a hybrid engine",
        table_format="markdown",
        leg_prompts={
            # Sentinels selecting a converter — see modal_apps/marker_app.py.
            # text shares the document pass with markdown: OCRConverter was the
            # right idea (dedicated OCR mode) but its OCRJSONRenderer tree has a
            # different schema from PdfConverter's block tree, and
            # _block_geometry() finds only a "Page" node with text=None —
            # yielding near-empty output. The baseline scored text=0.851 from
            # PdfConverter markdown; the text scorer strips formatting during
            # normalisation, so the result is the same.
            "text": "__marker_pdf_converter__",
            "markdown": "__marker_pdf_converter__",
            # geometry rides the OCR pass: OCRConverter DOES return per-line
            # polygons, but its tree parser is broken (see above), so geometry
            # also gets near-empty output. Left in place so the leg is tested
            # rather than silently dropped — a fix to the tree walker would
            # unblock both.
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
        # key_value dropped 2026-08-10. Not a task this model has: no documented
        # extraction prompt, and the authors state it never had an
        # instruction/chat training stage. Two different phrasings were tried
        # (with and without the <|grounding|> prefix) across 65 documents and it
        # never once returned JSON — it OCRs the page and hands back text or a
        # TSV table. Asking anyway measured our prompt, not the engine.
        prompted_legs=("markdown", "tables", "schema", "geometry"),
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
