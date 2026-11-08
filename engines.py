"""Single engine registry — the ONE place an engine is declared."""

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
    # What reads the page, for grouping results (the viewer's engine-type filter):
    # "general_vlm"  — a general-purpose vision-language model.
    # "ocr_model"    — a model trained for document OCR (some run a layout model first).
    # "pipeline"     — separate text detection and recognition models, plus layout/table
    #                  models where the pipeline has them; no language model reads the page.
    # "document_api" — a hosted document service that does not publish what it runs.
    engine_type: str = ""
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
    # GPU_TYPE in the app/response is authoritative; this registry value is a cost fallback.
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
    # Table format and per-leg prompt overrides; model-specific task recipes take precedence.
    leg_prompts: dict = field(default_factory=dict)
    table_format: str | None = None
    # Schema document or empty instance template, according to the engine's task recipe.
    schema_render: str = "json_schema"
    # Disable provider JSON mode for native non-JSON formats such as grounding markup.
    non_json_legs: tuple[str, ...] = ()
    # Native legs declared after the published runs were collected and served
    # by the stored default conversion (its reply already carried the data):
    # a rescore scores them from the saved reply instead of needing a new call.
    # Every other stored pass keeps exactly the legs it was saved for.
    default_pass_added_legs: tuple[str, ...] = ()
    # Default account limits that cap how many pages run at once, as each
    # provider documents them for a standard paid account with no requested
    # increase (checked 2026-11-02). Only used to project time at volume:
    # requests_per_second, requests/input_tokens/output_tokens_per_minute,
    # requests_per_day, concurrent_jobs or concurrent_gpus, with the basis and
    # its source; a basis starting "estimate:" marks a limit not published.
    default_limits: dict | None = None

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

# Account-level limits shared by several engines (see EngineSpec.default_limits).
_MODAL_STARTER = {"concurrent_gpus": 10, "basis": "Modal Starter plan: 10 GPUs at once, one per container", "source": "https://modal.com/pricing"}
_LLAMACLOUD_STARTER = {"concurrent_jobs": 5, "basis": "LlamaCloud Starter plan: 5 parse and 5 extract jobs at once", "source": "https://www.llamaindex.ai/pricing"}
_LOCAL = {"basis": "local: as many pages at once as your CPU cores allow"}

ENGINES: dict[str, EngineSpec] = {
    # --- Local (free, no keys) --------------------------------------------
    "tesseract": EngineSpec(
        name="tesseract", kind="local", engine_type="pipeline",
        runner="runners.tesseract_runner:TesseractRunner",
        default_limits=_LOCAL,
        cost_notes="local, free",
        native_legs=("text", "geometry"),
        leg_notes="flat text only: no markdown, no table structure, no prompt",
    ),
    "docling": EngineSpec(
        name="docling", kind="local", engine_type="pipeline",
        runner="runners.docling_runner:DoclingRunner",
        default_limits=_LOCAL,
        cost_notes="local, free",
        # geometry added 2026-10-20: every export carries prov[0].bbox per
        # text item and table (bottom-left origin, page size recorded);
        # scored as layout_block, never turned into word boxes.
        native_legs=_CONVERSION_LEGS + ("geometry",),
        default_pass_added_legs=("geometry",),
        leg_notes="fixed pipeline: cannot be given a prompt or a schema; "
                  "geometry is element boxes (layout_block)",
        table_format="markdown",
    ),
    # --- Cloud APIs --------------------------------------------------------
    "textract_text": EngineSpec(
        name="textract_text", kind="cloud", engine_type="document_api",
        runner="runners.textract_text_runner:TextractTextRunner",
        default_limits={"requests_per_second": 25, "basis": "AWS default quota for DetectDocumentText, US East (N. Virginia)", "source": "https://docs.aws.amazon.com/general/latest/gr/textract.html"},
        required_env=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        cost_per_page=0.0015,
        native_legs=("text", "geometry"),
        leg_notes="DetectDocumentText returns lines only: no structure, no prompt",
    ),
    "textract_structured": EngineSpec(
        name="textract_structured", kind="cloud", engine_type="document_api",
        runner="runners.textract_structured_runner:TextractStructuredRunner",
        default_limits={"requests_per_second": 10, "basis": "AWS default quota for AnalyzeDocument, US East (N. Virginia)", "source": "https://docs.aws.amazon.com/general/latest/gr/textract.html"},
        required_env=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        cost_per_page=0.065,
        cost_notes="AnalyzeDocument TABLES+FORMS",
        # Textract FORMS/TABLES supports KV, but accepts no supplied schema. Queries is a separate API.
        native_legs=("text", "tables", "key_value", "geometry"),
        leg_notes=(
            "no prompt; key_value judged from its FORMS pairs; schema not "
            "applicable (no schema/queries interface in this integration)"
        ),
    ),
    "mistral_ocr_4_1": EngineSpec(
        name="mistral_ocr_4_1", kind="cloud", engine_type="ocr_model",
        runner="runners.mistral_runner:MistralOCRRunner",
        default_limits={"requests_per_second": 1, "basis": "estimate: Mistral publishes no default; accounts start on the lowest limits and prepaying credits does not raise them, assumed 1 request per second (a third-party guide reports 5)", "source": "https://help.mistral.ai/en/articles/698531-why-am-i-hitting-api-rate-limits-and-how-do-i-increase-them"},
        required_env=("MISTRAL_API_KEY",),
        cost_per_page=0.004,
        native_legs=_CONVERSION_LEGS,
        # All passes pin `mistral-ocr-4-1` (runners/mistral_runner.py). schema
        # and key_value go through the vendor's Annotations feature, not a
        # prompt we invented: schema sends the page's JSON schema only;
        # key_value sends the generic leg prompt as the documented
        # `document_annotation_prompt` with an object schema that names no
        # fields (additionalProperties string|null), so the printed labels are
        # the model's own. Probed 2026-10-17 on funsd_003: 18 labels returned.
        prompted_legs=("key_value", "schema"),
        accepts_task=True,
        leg_notes="OCR endpoint takes no free-text prompt: markdown/tables from "
                  "default conversion; key_value and schema via Annotations",
        table_format="markdown",
    ),
    "llamaparse": EngineSpec(
        name="llamaparse", kind="cloud", engine_type="document_api",
        runner="runners.llamaparse_runner:LlamaParseRunner",
        default_limits=_LLAMACLOUD_STARTER,
        required_env=("LLAMA_CLOUD_API_KEY",),
        cost_per_page=0.00375,
        cost_notes="cost_effective tier: 3 credits × $1.25/1K; free tier: 10K credits/mo",
        # geometry added 2026-10-20: every parse reply carries one bBox per
        # layout item (heading/text/table) in the source image's pixel frame;
        # scored as layout_block, never turned into word boxes. Word and line
        # boxes are the separate `llamaparse_granular` configuration.
        native_legs=_CONVERSION_LEGS + ("geometry",),
        default_pass_added_legs=("geometry",),
        # schema via the vendor's own extraction feature (Mistral Annotations /
        # LlamaExtract), not a prompt. UNVALIDATED until the 2-doc check runs.
        prompted_legs=("schema",),
        accepts_task=True,
        leg_notes="parse endpoint takes no prompt in the cost_effective tier; "
                  "geometry is item boxes (layout_block); LlamaExtract runs with "
                  "cite_sources; the citations are saved with the call, not scored",
        table_format="markdown",
    ),
    "llamaparse_granular": EngineSpec(
        name="llamaparse_granular", kind="cloud", engine_type="document_api",
        runner="runners.llamaparse_runner:LlamaParseGranularRunner",
        default_limits=_LLAMACLOUD_STARTER,
        required_env=("LLAMA_CLOUD_API_KEY",),
        cost_per_page=0.00375,
        cost_notes="cost_effective tier: 3 credits × $1.25/1K; granular_bboxes "
                   "lists no extra credits (confirm on the job's credit usage)",
        default=False,
        # A measured geometry configuration of llamaparse (the pattern of
        # textract_text / textract_structured): the parse endpoint's documented
        # output_options.granular_bboxes=[word, line], scored on word boxes.
        # Text/markdown/tables stay with `llamaparse`; this row answers
        # geometry only, so the saved conversion run is untouched.
        native_legs=("geometry",),
        leg_notes="geometry only: word and line boxes from the grounded_items "
                  "sidecar; same parse tier and price as llamaparse",
    ),
    "gemini_flash": EngineSpec(
        name="gemini_flash", kind="cloud", engine_type="general_vlm",
        runner="runners.gemini_runner:GeminiRunner",
        default_limits={"requests_per_minute": 4000, "input_tokens_per_minute": 4000000, "requests_per_day": 150000, "basis": "estimate: Tier 1 values for Gemini 3.1 Flash-Lite from an undated third-party table; Google shows each project's limits only in AI Studio", "source": "https://www.scriptbyai.com/gemini-api-rate-limits/"},
        required_env=("GOOGLE_API_KEY",),
        cost_per_page=0.0007,
        cost_notes="variable — token-based",
        accepts_task=True,
        # Text uses transcription; Markdown and tables use their dedicated task prompts.
        native_legs=("text",),
        prompted_legs=("markdown", "tables", "key_value", "schema", "geometry"),
        table_format="markdown",
    ),
    "claude_sonnet_4_6": EngineSpec(
        name="claude_sonnet_4_6", kind="cloud", engine_type="general_vlm",
        runner="runners.claude_runner:ClaudeRunner",
        default_limits={"requests_per_minute": 1000, "input_tokens_per_minute": 2000000, "output_tokens_per_minute": 400000, "basis": "Anthropic Start tier, Sonnet 4.x (spend capped at $500 a month)", "source": "https://platform.claude.com/docs/en/api/rate-limits"},
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
        name="mineru", kind="modal", engine_type="ocr_model",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
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
            # MinerU2.5 chooses region prompts internally; this pipeline accepts no caller prompt or image-analysis task.
        },
    ),
    "glm_ocr": EngineSpec(
        name="glm_ocr", kind="modal", engine_type="ocr_model",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
        cost_per_page=0.000617, cost_notes="Modal A10; measured batch estimate",
        modal_app="ocr-bench-glm-ocr", modal_class="GLMOCRInference",
        modal_app_file="modal_apps/glm_ocr_app.py",
        gpu_type="A10", batch_size=24,
        accepts_task=True,
        # GLM Markdown uses a dedicated generic prompt; its card documents no Markdown task.
        native_legs=("text",),
        # Keep direct vLLM routing until the repaired selfhosted SDK passes a controlled comparison.
        # KV remains disabled after extraction failures; schema uses the card's instance-template recipe.
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
            # Generic Markdown prompt; the repaired SDK pipeline remains unmeasured.
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
        name="paddleocr_vl", kind="modal", engine_type="ocr_model",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
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
            # PaddleOCR-VL uses its fixed page pipeline: native Markdown, HTML tables and geometry.
            # Element-level model prompts do not apply; the pipeline has no KV/schema extraction interface.
        },
    ),
    "marker": EngineSpec(
        name="marker", kind="modal", engine_type="pipeline",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
        cost_per_page=0.001420, cost_notes="Modal A100-80GB; measured batch estimate",
        modal_app="ocr-bench-marker", modal_class="MarkerInference",
        modal_app_file="modal_apps/marker_app.py",
        gpu_type="A100-80GB", batch_size=24, supports_pdf=True,
        accepts_task=True,
        # Marker selects OCR, document or table conversion through task-specific sentinels.
        native_legs=(),
        prompted_legs=("text", "markdown", "tables", "geometry"),
        # Text and markdown share the document converter; geometry uses the
        # separate OCR converter. These are converter selectors, not prompts.
        non_json_legs=("text", "markdown", "tables", "geometry"),
        leg_notes="structured extraction excluded: Marker's ExtractionConverter "
                  "requires an external LLM, which would make this a hybrid engine",
        table_format="markdown",
        leg_prompts={
            # Converter sentinels are interpreted in modal_apps/marker_app.py.
            "text": "__marker_pdf_converter__",
            "markdown": "__marker_pdf_converter__",
            # Native OCR line polygons are the geometry evidence. The
            # explicit all-blocks probe is not silently selected here.
            "geometry": "__marker_ocr_converter__",
            "tables": "__marker_table_converter__",
        },
    ),
    "deepseek_ocr": EngineSpec(
        name="deepseek_ocr", kind="modal", engine_type="ocr_model",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
        cost_per_page=0.000975, cost_notes="Modal A100-80GB; measured batch estimate",
        modal_app="ocr-bench-deepseek-ocr", modal_class="DeepSeekOCRInference",
        modal_app_file="modal_apps/deepseek_ocr_app.py",
        gpu_type="A100-80GB", batch_size=24,
        accepts_task=True,
        native_legs=("text",),
        # No documented extraction recipe; retain prompted participation rather than excluding poor results.
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
            # DeepSeek tables share the grounding conversion pass and its native table-token repetition exceptions.
            "tables": "<|grounding|>Convert the document to markdown.",
            # Geometry shares the grounding pass; retain region text and boxes in its native coordinate frame.
            "geometry": "<|grounding|>Convert the document to markdown.",
            # Experimental JSON extraction prompt. Omit the grounding prefix, which requests layout conversion.
            "schema": "Extract the values from this document as JSON "
                      "conforming exactly to this schema. Output only JSON.\n\n{schema}",
        },
    ),
    "qwen3_vl": EngineSpec(
        name="qwen3_vl", kind="modal", engine_type="general_vlm",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
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
            # Use Qwen's OCR/KIE and grounding cookbook recipes; text has a dedicated OCR pass.
            "text": "Read all the text in the image.",
            # tables is given markdown's exact wording so the two resolve to one
            # ask and the planner folds them into a single call — the table is
            # part of the document formatting, and Qwen documents no separate
            # table task.
            "tables": (
                "Convert this document to Markdown. Preserve reading order, "
                "headings, lists and tables. Output only the Markdown."
            ),
            # Qwen Markdown/tables use generic instructions; geometry follows the text-spotting recipe.
            "geometry": (
                "Spotting all the text in the image with word-level, and output "
                "in JSON format as [{'bbox_2d': [x1, y1, x2, y2], "
                "'text_content': 'text'}, ...]."
            ),
        },
    ),
    # Additional measured geometry configurations.
    "paddleocr_ppocrv5": EngineSpec(
        name="paddleocr_ppocrv5", kind="modal", engine_type="pipeline",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
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
        name="mineru_pipeline", kind="modal", engine_type="pipeline",
        runner="runners.modal_runner:ModalRunner",
        default_limits=_MODAL_STARTER,
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
