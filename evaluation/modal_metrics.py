"""Audit and summarize Modal batch timing records.

Modal models reserve one GPU for the complete ``run_ocr_batch`` method.  The
method timer starts after ``@modal.enter`` has loaded and warmed the model, so:

    effective GPU seconds/document = batch_gpu_seconds / batch_size
    cost/document = effective GPU seconds/document * GPU rate
    documents/second = 1 / effective GPU seconds/document

This module verifies those identities from the persisted JSONL records.  Keeping
the audit independent from the report aggregation prevents a display-only mean
from silently becoming the benchmark's billing formula.
"""

from __future__ import annotations

from collections import defaultdict

from modal_apps._shared import GPU_SECOND_RATES_USD
from runners.modal_runner import MODAL_ENGINES


def audit_modal_records(records: list[dict]) -> tuple[list[dict], list[str], bool]:
    """Return ``(engine summaries, errors, used_batch_measurements)``.

    Historical runs used per-request timing and have no ``measurement_scope``.
    They are left reportable, but are not treated as audited batch measurements.
    If a run contains any new batch records, every successful Modal record in
    that run must use the same auditable schema.
    """

    modal_rows = [
        row
        for row in records
        if row.get("engine_name") in MODAL_ENGINES and not row.get("error")
    ]
    if not modal_rows:
        return [], [], False

    batch_rows = [
        row for row in modal_rows if row.get("measurement_scope") == "modal_batch"
    ]
    if not batch_rows:
        return [], [], False

    errors: list[str] = []
    if len(batch_rows) != len(modal_rows):
        errors.append(
            f"mixed Modal timing schemas: {len(batch_rows)} batch rows and "
            f"{len(modal_rows) - len(batch_rows)} legacy rows"
        )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in batch_rows:
        engine = row["engine_name"]
        batch_seconds = _positive_float(row.get("batch_gpu_seconds"))
        batch_size = _positive_int(row.get("batch_size"))
        batch_id = row.get("batch_id")
        if not batch_id:
            batch_id = f"legacy:{batch_seconds!r}:{batch_size!r}"
        grouped[(engine, batch_id)].append(row)

    engine_batches: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for (engine, batch_id), rows in grouped.items():
        first = rows[0]
        batch_seconds = _positive_float(first.get("batch_gpu_seconds"))
        batch_size = _positive_int(first.get("batch_size"))
        if batch_seconds is None or batch_size is None:
            errors.append(f"{engine}/{batch_id}: invalid batch timer or batch size")
            continue

        if len(rows) > batch_size:
            errors.append(
                f"{engine}/{batch_id}: {len(rows)} successful records for "
                f"batch_size={batch_size} (more results than inputs)"
            )

        expected_gpu_type = MODAL_ENGINES[engine][2].upper()
        expected_rate = GPU_SECOND_RATES_USD[expected_gpu_type]
        expected_seconds = batch_seconds / batch_size
        expected_cost = expected_seconds * expected_rate

        for row in rows:
            prefix = f"{engine}/{batch_id}/{row.get('document_path', '?')}"
            if row.get("startup_excluded") is not True:
                errors.append(f"{prefix}: startup_excluded is not true")

            gpu_type = str(row.get("gpu_type") or expected_gpu_type).upper()
            if gpu_type != expected_gpu_type:
                errors.append(
                    f"{prefix}: gpu_type={gpu_type}, expected {expected_gpu_type}"
                )

            stored_rate = _float(row.get("gpu_rate_usd_per_second"))
            if stored_rate is not None and abs(stored_rate - expected_rate) > 1e-12:
                errors.append(
                    f"{prefix}: GPU rate {stored_rate}, expected {expected_rate}"
                )

            _check_close(
                errors,
                prefix,
                "effective_gpu_seconds_per_document",
                row.get("effective_gpu_seconds_per_document"),
                expected_seconds,
                5.1e-7,
            )
            _check_close(
                errors,
                prefix,
                "latency_seconds",
                row.get("latency_seconds"),
                expected_seconds,
                5.1e-4,
            )
            _check_close(
                errors,
                prefix,
                "steady_latency_seconds",
                row.get("steady_latency_seconds"),
                expected_seconds,
                5.1e-4,
            )
            _check_close(
                errors,
                prefix,
                "cost_per_page",
                row.get("cost_per_page"),
                expected_cost,
                5.1e-7,
            )
            _check_close(
                errors,
                prefix,
                "steady_cost_per_page",
                row.get("steady_cost_per_page"),
                expected_cost,
                5.1e-7,
            )

        engine_batches[engine].append((batch_seconds, batch_size))

    summaries = []
    for engine, batches in sorted(engine_batches.items()):
        total_seconds = sum(seconds for seconds, _ in batches)
        documents = sum(size for _, size in batches)
        seconds_per_document = total_seconds / documents
        gpu_type = MODAL_ENGINES[engine][2].upper()
        rate = GPU_SECOND_RATES_USD[gpu_type]
        summaries.append(
            {
                "engine": engine,
                "gpu_type": gpu_type,
                "documents": documents,
                "batches": len(batches),
                "batch_gpu_seconds": total_seconds,
                "gpu_seconds_per_document": seconds_per_document,
                "documents_per_second": documents / total_seconds,
                "cost_per_document": seconds_per_document * rate,
            }
        )

    return summaries, errors, True


def _float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _positive_float(value) -> float | None:
    parsed = _float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _positive_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _check_close(
    errors: list[str],
    prefix: str,
    field: str,
    actual_value,
    expected: float,
    tolerance: float,
) -> None:
    actual = _float(actual_value)
    if actual is None or abs(actual - expected) > tolerance:
        errors.append(
            f"{prefix}: {field}={actual_value!r}, expected {expected:.12f}"
        )
