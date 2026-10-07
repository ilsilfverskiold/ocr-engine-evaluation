"""Invariants that keep every engine measured with the ask its model card
prescribes, on every leg it competes in.

These exist because each of the following actually happened during development
and none of them announced itself: an engine's tables were scored from a
text-recognition call; a model-card prompt sat in the registry but the runtime
sent the app's stale default instead; a general VLM was silently excluded from
a leg it could answer. If any of these tests fails, the benchmark is measuring
our configuration, not the engines — do not run it until they pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines import ENGINES
from evaluation.legs import ALL_LEGS, GEOMETRY, KEY_VALUE, LEGS, plan_passes

DUMMY_SCHEMA = {"type": "object", "properties": {"total": {"type": "string"}}}


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_marker_worker_isolates_config_and_preserves_native_geometry(monkeypatch, probe, fails):
    import ast
    from types import SimpleNamespace

    path = Path(__file__).parent.parent / "modal_apps/marker_app.py"
    tree = ast.parse(path.read_text())
    names = {"_marker_pass_config", "_block_geometry", "_marker_worker_convert"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    original = {"nested": {"untouched": True}}
    ns = {"OCR_PASS": "original", "OCR_AUDIT_PASS": "probe", "TABLE_PASS": "table",
          "DOC_PASS": "document", "GPU_TYPE": "A100-80GB", "MARKER_WORKERS": 11,
          "_marker_config": original, "_marker_model_dict": {},
          "_marker_page_size": lambda _: (200, 100)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    raw = {"block_type": "Document", "children": [{"block_type": "Page", "children": [
        {"block_type": "Line", "html": "Hi", "bbox": [20, 10, 60, 30], "children": [
            {"block_type": "Char", "text": "H", "bbox": [20, 10, 40, 30]},
            {"block_type": "Char", "text": "i", "bbox": [40, 10, 60, 30]},
        ]},
    ]}]}

    class FakeConverter:
        def __init__(self, artifact_dict, config):
            # The pinned OCRConverter mutates the caller's dict in this way.
            config["format_lines"] = True
            config["nested"]["untouched"] = False
            assert config.get("keep_chars", False) == probe
            assert ("skip_ocr_blocks" in config) == probe
            if probe:
                assert config["skip_ocr_blocks"] == []
        def __call__(self, filename):
            if fails:
                raise TypeError("fixture pipeline failure")
            return SimpleNamespace(model_dump=lambda: raw)

    monkeypatch.setitem(sys.modules, "marker.converters.ocr", SimpleNamespace(OCRConverter=FakeConverter))
    result = ns["_marker_worker_convert"](("fixture.png", "fixture.png", "probe" if probe else "original"))
    assert original == {"nested": {"untouched": True}}
    assert result["prompt_used"] == ("probe" if probe else "original")
    so = result["structured_output"]
    assert so["converter_config"]["format_lines"] is True
    if fails:
        assert result["error"] == "TypeError: fixture pipeline failure"
        assert "__call__" in so["error_traceback"]
        assert so["pass_requested"] == result["prompt_used"]
    else:
        assert result["raw_text"] == "Hi"  # Characters are not duplicated as lines.
        assert so["text_regions"] == [{"text": "Hi", "bbox": [0.1, 0.1, 0.3, 0.3]}]
        assert so["geometry_granularity"] == "ocr_line"
        assert so["ocr_renderer_output"] == raw
        assert len(so["ocr_lines"]) == 3  # Native character evidence is retained.


@pytest.mark.parametrize("wrong_endpoint", [False, True])
def test_glm_sdk_probe_requires_local_endpoint_and_retains_coarse_output(monkeypatch, wrong_endpoint):
    import ast
    import importlib.metadata
    from types import SimpleNamespace

    # Load only pure helpers/method body, without importing Modal or models.
    path = Path(__file__).parent.parent / "modal_apps/glm_ocr_app.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GLMOCRInference")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_document_parse")
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sdk_kwargs")
    ns = {"GLMOCR_VERSION": "0.1.5", "SERVER_HOST": "127.0.0.1", "SERVER_PORT": 8001,
          "SERVED_MODEL_NAME": "glm-ocr"}
    exec(compile(ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"), ns)
    calls = []
    raw = [[{"label": "text", "content": "two words", "bbox_2d": [100, 200, 400, 300]}]]
    config = SimpleNamespace(pipeline=SimpleNamespace(
        maas=SimpleNamespace(enabled=False),
        ocr_api=SimpleNamespace(api_host="127.0.0.1", api_port=8001,
                               api_url="https://unexpected.example" if wrong_endpoint else None,
                               api_path="/v1/chat/completions", model="glm-ocr"),
        page_loader=SimpleNamespace(model_dump=lambda: {"max_tokens": 8192}),
    ))

    def load_config(**kwargs):
        assert kwargs == {"mode": "selfhosted", "ocr_api_host": "127.0.0.1",
                          "ocr_api_port": 8001, "model": "glm-ocr", "layout_device": "cpu"}
        return config

    class FakeOcr:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            calls.append("closed")
        def parse(self, path, **kwargs):
            return SimpleNamespace(markdown_result="two words", json_result=raw)

    monkeypatch.setitem(sys.modules, "glmocr", SimpleNamespace(GlmOcr=FakeOcr))
    monkeypatch.setitem(sys.modules, "glmocr.config", SimpleNamespace(load_config=load_config))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.5")
    if wrong_endpoint:
        with pytest.raises(RuntimeError, match="endpoint/model differs"):
            ns["_document_parse"](None, "mock.png")
        assert calls == []  # fail before constructing any SDK/network client
    else:
        result = ns["_document_parse"](None, "mock.png")
        so = result["structured_output"]
        assert calls[-1] == "closed"
        assert so["sdk_json_result"] == raw
        assert so["geometry_granularity"] == "layout_block" and "words" not in so
        assert so["layout_blocks"][0]["bbox"] == [0.1, 0.2, 0.4, 0.3]


@pytest.mark.parametrize("engine", ["gemini", "claude"])
@pytest.mark.parametrize("prompted", [False, True])
def test_api_prompt_sent_is_persisted_for_default_and_task_calls(monkeypatch, tmp_path, engine, prompted):
    from types import SimpleNamespace
    import run_benchmark as rb
    from runners.base import OCRTask
    from runners import claude_runner, gemini_runner

    captured = {}
    task = OCRTask(leg="markdown", prompt="Preserve headings as Markdown.") if prompted else None
    document = tmp_path / "sample.png"
    document.write_bytes(b"mock image")
    usage = SimpleNamespace(input_tokens=1, output_tokens=1, prompt_token_count=1, candidates_token_count=1)
    response = SimpleNamespace(text="answer", content=[SimpleNamespace(text="answer")],
                               usage=usage, usage_metadata=usage, stop_reason="end_turn")

    def generate(**kwargs):
        captured.update(kwargs)
        return response

    if engine == "gemini":
        module, cls = gemini_runner, gemini_runner.GeminiRunner
        fake_types = SimpleNamespace(Part=SimpleNamespace(from_bytes=lambda **kwargs: kwargs))
        monkeypatch.setitem(sys.modules, "google.genai", SimpleNamespace(types=fake_types))
        runner = cls.__new__(cls)
        runner._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        runner._model_name = "mock-model"
    else:
        module, cls = claude_runner, claude_runner.ClaudeRunner
        monkeypatch.setattr(module, "_prepare_image", lambda path: (b"mock image", "image/png"))
        runner = cls.__new__(cls)
        runner._client = SimpleNamespace(messages=SimpleNamespace(create=generate))
        runner._model = "mock-model"
    monkeypatch.setattr(module, "provider_payload", lambda response: {})
    result = runner.run(document, task)
    sent = captured["contents"][1] if engine == "gemini" else captured["messages"][0]["content"][1]["text"]
    assert sent == (task.prompt if task else cls._PROMPT)
    monkeypatch.setattr(rb, "OCR_OUTPUTS_DIR", tmp_path / "outputs")
    gt = SimpleNamespace(tier="easy", doc_type="invoice")
    rb._write_ocr_output(runner.name, document, gt, result.raw_text, "test", "default",
                         result=result, task=task, legs=("markdown",))
    import json
    saved = json.loads(next((tmp_path / "outputs").rglob("*.call.json")).read_text())
    assert saved["prompt_sent"] == sent


def test_native_and_prompted_never_overlap():
    for name, spec in ENGINES.items():
        both = set(spec.native_legs) & set(spec.prompted_legs)
        assert not both, f"{name}: legs declared both native and prompted: {both}"


def test_unpromptable_engines_declare_no_prompted_legs():
    # A prompted leg on an engine that cannot receive a task would silently
    # run as a default conversion and be scored as if it had been asked.
    for name, spec in ENGINES.items():
        if not spec.accepts_task:
            assert not spec.prompted_legs, (
                f"{name} cannot accept a task but declares prompted legs "
                f"{spec.prompted_legs} — those calls would never carry the prompt"
            )


def test_every_prompted_leg_resolves_to_a_prompt():
    for name, spec in ENGINES.items():
        for leg in spec.prompted_legs:
            resolved = spec.prompt_for(leg, LEGS[leg].prompt)
            assert resolved, (
                f"{name}/{leg} is prompted but resolves to no prompt — "
                f"the engine would be sent nothing and scored anyway"
            )


def test_schema_prompts_carry_the_schema_placeholder():
    for name, spec in ENGINES.items():
        if "schema" in spec.prompted_legs:
            resolved = spec.prompt_for("schema", LEGS["schema"].prompt)
            assert "{schema}" in resolved, (
                f"{name}: schema prompt has no {{schema}} placeholder — every "
                f"document would be asked without its schema"
            )


def test_no_orphan_prompt_overrides():
    # A leg_prompts entry for a leg the engine does not answer is a prompt
    # that never gets sent — almost always a sign of a mis-declared leg.
    # glm_ocr's key_value override was removed on 2026-09-30 when key_value
    # was re-enabled for it with the generic ask (the card's KIE prompt needs
    # a supplied structure, which this leg does not give).
    KNOWN_ORPHANS: dict[str, set[str]] = {}
    for name, spec in ENGINES.items():
        answered = set(spec.native_legs) | set(spec.prompted_legs)
        orphans = set(spec.leg_prompts) - answered
        expected = KNOWN_ORPHANS.get(name, set())
        unexpected = orphans - expected
        assert not unexpected, (
            f"{name}: prompt overrides for legs it does not answer: {unexpected}"
        )


def test_text_override_requires_task_support():
    # The default pass sends the registry's text prompt explicitly; an engine
    # that cannot accept a task would silently fall back to its app default.
    for name, spec in ENGINES.items():
        if spec.leg_prompts.get("text"):
            assert spec.accepts_task, (
                f"{name} declares a text prompt but accepts_task=False — "
                f"the runtime could never send it"
            )


def test_every_pass_task_matches_the_registry_prompt():
    # End to end: the task actually built for each planned pass must carry the
    # engine-resolved prompt, not the generic one and not None.
    for name, spec in ENGINES.items():
        for plan in plan_passes(name, list(ALL_LEGS)):
            if plan.leg_for_task is None:
                continue
            leg_spec = LEGS[plan.leg_for_task]
            task = leg_spec.task_for(
                DUMMY_SCHEMA if leg_spec.needs_schema else None, engine=name
            )
            assert task is not None, f"{name}/{plan.name}: no task built"
            expected = spec.prompt_for(plan.leg_for_task, leg_spec.prompt)
            if leg_spec.needs_schema:
                assert expected.split("{schema}")[0] in task.prompt, (
                    f"{name}/{plan.name}: task prompt does not start from the "
                    f"registry prompt"
                )
            else:
                assert task.prompt == expected, (
                    f"{name}/{plan.name}: task prompt differs from the registry "
                    f"prompt.\n  task:     {task.prompt!r}\n  registry: {expected!r}"
                )


def test_merged_passes_share_one_identical_ask():
    # Legs folded into one call must resolve to the same prompt and format;
    # otherwise one of them is being scored on another leg's output.
    # The format is determined by the engine's response_format_for(), which
    # honours non_json_legs — e.g. deepseek_ocr's geometry shares the
    # grounding call with markdown/tables but must not force JSON mode.
    for name, spec in ENGINES.items():
        for plan in plan_passes(name, list(ALL_LEGS)):
            if plan.leg_for_task is None or len(plan.legs) == 1:
                continue
            asks = {
                (
                    spec.prompt_for(leg, LEGS[leg].prompt),
                    spec.response_format_for(leg, LEGS[leg].needs_schema),
                )
                for leg in plan.legs
            }
            assert len(asks) == 1, (
                f"{name}/{plan.name}: merged legs {plan.legs} resolve to "
                f"different asks — they must not share a call"
            )


def test_known_model_card_asks_are_what_runs():
    """Pin the model-card prompts we researched, so a silent edit fails loudly.

    Sources: huggingface.co model cards for zai-org/GLM-OCR,
    deepseek-ai/DeepSeek-OCR, PaddlePaddle/PaddleOCR-VL (2026-08).
    """
    cases = {
        ("glm_ocr", "text"): "Text Recognition:",
        ("glm_ocr", "tables"): "Table Recognition:",
        ("deepseek_ocr", "text"): "Free OCR.",
        ("deepseek_ocr", "markdown"): "<|grounding|>Convert the document to markdown.",
    }
    for (engine, leg), expected in cases.items():
        got = ENGINES[engine].prompt_for(leg, LEGS[leg].prompt)
        assert got == expected, f"{engine}/{leg}: {got!r} != model card {expected!r}"
    assert ENGINES["glm_ocr"].prompt_for("schema", LEGS["schema"].prompt).startswith(
        "请按下列JSON格式输出图中信息"
    )
    # PaddleOCR-VL runs as the pipeline: no prompt interface, four legs from
    # one default output (geometry added 2026-08-10 — the pipeline natively
    # returns rec_boxes/rec_polys), and never the extraction legs.
    paddle = ENGINES["paddleocr_vl"]
    assert not paddle.accepts_task
    assert set(paddle.native_legs) == {"text", "markdown", "tables", "geometry"}
    assert not paddle.prompted_legs


def test_pass_counts_are_minimal():
    # gemini and claude are general VLMs: text from the runner's transcription
    # prompt, markdown and tables each from the leg's own ask (the 2026-08 run
    # scored both from the transcription pass, which never asks for headings,
    # lists or tables-only output), so the conversion legs are three passes.
    for engine in ("gemini_flash", "claude_sonnet_4_6"):
        plans = plan_passes(engine, ["text", "markdown", "tables"])
        assert {p.name for p in plans} == {"default", "markdown", "tables"}, f"{engine}: {[p.name for p in plans]}"
    # qwen3_vl splits text into its own pass (the card documents a dedicated OCR
    # task distinct from the markdown conversion), so text+markdown+tables is 2.
    plans = plan_passes("qwen3_vl", ["text", "markdown", "tables"])
    assert len(plans) == 2, f"qwen3_vl: {len(plans)} passes for the conversion legs"
    # deepseek's markdown and tables share the grounding call.
    plans = plan_passes("deepseek_ocr", ["markdown", "tables"])
    assert len(plans) == 1 and set(plans[0].legs) == {"markdown", "tables"}


@pytest.mark.parametrize('engine', ['claude', 'gemini'])
@pytest.mark.parametrize('leg', ['schema', 'markdown'])
def test_api_existing_caps_and_billed_thinking(monkeypatch, tmp_path, engine, leg):
    from types import SimpleNamespace as NS
    from runners.base import OCRTask
    from runners import claude_runner, gemini_runner
    captured = {}
    usage = NS(input_tokens=100, output_tokens=200, prompt_token_count=100,
               candidates_token_count=200, thoughts_token_count=50)
    response = NS(text='{}', content=[NS(text='{}')], usage=usage,
                  usage_metadata=usage, stop_reason='end_turn')
    def generate(**kwargs):
        captured.update(kwargs)
        return response
    document = tmp_path / 'page.png'
    document.write_bytes(b'mocked image')
    task = OCRTask(leg=leg, prompt='Exact benchmark prompt',
                   json_schema={'type': 'object'} if leg == 'schema' else None)
    if engine == 'claude':
        module = claude_runner
        runner = module.ClaudeRunner.__new__(module.ClaudeRunner)
        runner._client = NS(messages=NS(create=generate))
        runner._model = 'claude-sonnet-4-6'
    else:
        module = gemini_runner
        runner = module.GeminiRunner.__new__(module.GeminiRunner)
        runner._client = NS(models=NS(generate_content=generate))
        runner._model_name = 'gemini-3.1-flash-lite'
        types = NS(Part=NS(from_bytes=lambda **kw: kw), GenerateContentConfig=lambda **kw: kw)
        monkeypatch.setitem(sys.modules, 'google.genai', NS(types=types))
    monkeypatch.setattr(module, 'provider_payload', lambda r: {})
    result = runner.run(document, task)
    if engine == 'claude':
        assert captured['max_tokens'] == 4096
        assert result.metadata['max_output_tokens'] == captured['max_tokens']
    else:
        if leg == 'schema':
            assert captured['config']['max_output_tokens'] == 4096
            assert captured['config']['response_mime_type'] == 'application/json'
        else:
            assert 'config' not in captured  # Preserve prior Markdown control settings.
        assert result.metadata['thinking_tokens'] == 50
        assert result.cost_per_page == pytest.approx(100 * .25e-6 + 250 * 1.5e-6)
    assert result.metadata['prompt_used'] == task.prompt


def test_llama_schema_submits_once_and_polls_existing_job(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    from llama_cloud import StatusEnum
    from runners.llamaparse_runner import LlamaParseRunner
    import llama_cloud.client
    calls = []
    def submit(**kwargs):
        calls.append(kwargs)
        return NS(id='saved-job')
    endpoint = NS(extract_stateless=submit,
                  get_job=lambda job_id: NS(status=StatusEnum.SUCCESS),
                  get_run_by_job_id=lambda job_id: NS(data={'Total': '100'}))
    monkeypatch.setattr(llama_cloud.client, 'LlamaCloud', lambda **kw: NS(llama_extract=endpoint))
    monkeypatch.setenv('LLAMA_CLOUD_API_KEY', 'fixture')
    runner = LlamaParseRunner.__new__(LlamaParseRunner)
    path = tmp_path / 'page.png'
    path.write_bytes(b'image')
    schema = {'type': 'object', 'properties': {'Total': {'type': ['string', 'null']}},
              'required': ['Total'], 'additionalProperties': False}
    result = runner._run_extract(path, schema)
    assert len(calls) == 1 and calls[0]['data_schema'] == schema
    assert calls[0]['config'].extraction_mode.value == 'MULTIMODAL'
    assert result.metadata['job_id'] == 'saved-job'
    assert result.raw_text == '{"Total": "100"}'
    assert result.cost_per_page == .025
    def failed_poll(job_id):
        raise RuntimeError('poll unavailable')
    endpoint.get_job = failed_poll
    result = runner._run_extract(path, schema)
    assert len(calls) == 2  # One submission each; failed polling did not resubmit.
    assert 'saved-job' in result.error and result.raw_response['job']['id'] == 'saved-job'


def test_deepseek_grounding_labels_are_not_recognized_words():
    import ast
    path = Path(__file__).parent.parent / 'modal_apps/deepseek_ocr_app.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_clean_deepseek_output', '_html_table_to_text')]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    raw = '<|ref|>title<|/ref|><|det|>[[0, 0, 999, 99]]<|/det|>\n# Actual title'
    text, so = ns['_clean_deepseek_output'](raw, preserve_markdown=True)
    assert text == '# Actual title'
    assert 'words' not in so
    assert so['layout_blocks'][0]['content'] == '# Actual title'
    assert so['regions'][0]['label'] == 'title'


def test_sonnet_geometry_records_sent_frame_and_normalizes_pixels(monkeypatch, tmp_path):
    import math
    from types import SimpleNamespace
    from PIL import Image
    from runners.base import OCRTask
    from runners import claude_runner as module
    path = tmp_path / 'page.png'
    Image.new('RGB', (2400, 3200), 'white').save(path)
    sent = {}
    def generate(**kw):
        sent.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(text='[{"text":"Word","bbox":[100,200,300,240]}]')],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason='end_turn')
    runner = module.ClaudeRunner.__new__(module.ClaudeRunner)
    runner._client = SimpleNamespace(messages=SimpleNamespace(create=generate))
    runner._model = 'claude-sonnet-4-6'
    monkeypatch.setattr(module, 'provider_payload', lambda _: {})
    result = runner.run(path, OCRTask(leg='geometry', prompt='old normalized prompt'))
    w, h = result.metadata['geometry_frame']['sent_size']
    assert max(w,h) <= 1568 and math.ceil(w/28)*math.ceil(h/28) <= 1568
    assert 'absolute pixel' in result.metadata['prompt_used']
    assert result.structured_output['words'][0]['bbox'] == [100/w,200/h,300/w,240/h]
    assert sent['max_tokens'] == 32768
    import json
    import run_benchmark as rb
    monkeypatch.setattr(rb, 'OCR_OUTPUTS_DIR', tmp_path / 'saved')
    rb._write_ocr_output(runner.name, path, SimpleNamespace(tier='hard', doc_type='form'), result.raw_text,
                         'test', 'geometry', result=result,
                         task=OCRTask(leg='geometry', prompt='old normalized prompt'), legs=('geometry',))
    saved = json.loads(next((tmp_path / 'saved').rglob('*.call.json')).read_text())
    assert saved['prompt_sent'] == result.metadata['prompt_used']
