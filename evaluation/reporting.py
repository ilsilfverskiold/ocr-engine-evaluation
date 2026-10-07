"""Shared, deterministic reporting diagnostics; never changes saved scores."""
from collections import defaultdict
from pathlib import Path
import math

LOW_THRESHOLD = 0.25
MIN_ENGINES = 3


def valid(row):
    value = row.get('score')
    return bool(row.get('applicable') and not row.get('error') and isinstance(value, (int, float)) and math.isfinite(value))


def case_key(row):
    return row['document_path'], row['leg']


def low_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['leg'] != 'judge':
            groups[case_key(row)].append(row)
    result = []
    for (document, leg), records in sorted(groups.items()):
        scored = [row for row in records if valid(row)]
        engines = {row['engine_name'] for row in scored}
        # A failed or pending participant cannot establish universal poor performance.
        unresolved = any(row.get('error') or (row.get('applicable') and not valid(row)) or (row.get('metrics') or {}).get('pending', 0) or 'pending' in str(row.get('reason', '')).lower() for row in records)
        if not unresolved and len(engines) >= MIN_ENGINES and max(row['score'] for row in scored) <= LOW_THRESHOLD:
            result.append({'document_path': document, 'page': Path(document).stem, 'leg': leg,
                           'doc_type': scored[0]['doc_type'], 'engines': len(engines),
                           'min': min(row['score'] for row in scored), 'max': max(row['score'] for row in scored)})
    return result


def comparison(rows):
    cases = low_cases(rows)
    excluded = {(c['document_path'], c['leg']) for c in cases}
    buckets = defaultdict(list)
    for row in rows:
        if row['leg'] != 'judge':
            buckets[(row['engine_name'], row['leg'])].append(row)
    output = []
    def stats(records):
        groups = defaultdict(list)
        for row in records:
            if valid(row):
                groups[(row.get('tier'), row['doc_type'])].append(row['score'])
        means = [round(sum(v)/len(v), 4) for v in groups.values()]
        return {'score': sum(means)/len(means) if means else None,
                'n': sum(map(len, groups.values())), 'types': len(groups)}
    for (engine, leg), records in sorted(buckets.items()):
        full = stats(records)
        remaining = stats([row for row in records if case_key(row) not in excluded])
        output.append({'engine': engine, 'leg': leg, 'full': full, 'without_low': remaining,
                       'removed': full['n'] - remaining['n'], 'types_removed': full['types'] - remaining['types']})
    return {'threshold': LOW_THRESHOLD, 'minimum_engines': MIN_ENGINES, 'cases': cases, 'scores': output}


SUPPORT = {'text': ['cer', 'wer', 'word_f1'], 'markdown': ['markdown_cer', 'markdown_wer', 'element_f1'],
           'tables': ['cell_f1', 'extra_cell_rate', 'structure_similarity'],
           'geometry': ['geometry_precision', 'geometry_f1', 'geometry_cer', 'geometry_wer']}


def print_report(rows, inference, log, legs):
    import run_benchmark as rb
    selected = [row for row in rows if row['leg'] in legs]
    analysis = comparison(rows)  # Always use the full engine cohort, before display filters.
    fmt = lambda n: 'N/A' if n is None else f'{n:.4f}'
    money = lambda n: 'N/A' if n is None else f'{n:.6f}'
    def table(headers, records):
        strings = [[str(value) for value in record] for record in records]
        widths = [max(len(head), *(len(row[i]) for row in strings)) if strings else len(head) for i, head in enumerate(headers)]
        log.info('  '.join(head.ljust(widths[i]) for i, head in enumerate(headers)))
        for row in strings:
            log.info('  '.join(value.ljust(widths[i]) for i, value in enumerate(row)))
    log.info('\nSCORES — full corpus is primary; means give each document type equal weight')
    table(['Engine', 'Leg', 'Full', 'N', 'Without low', 'N', 'Types lost'],
          [[x['engine'], x['leg'], fmt(x['full']['score']), x['full']['n'], fmt(x['without_low']['score']), x['without_low']['n'], x['types_removed']]
           for x in analysis['scores'] if x['leg'] in legs])
    log.info('\nCONSISTENTLY LOW CASES — every scored engine ≤0.25; minimum 3 engines; errors/pending block selection')
    cases = [c for c in analysis['cases'] if c['leg'] in legs]
    table(['Page', 'Leg', 'Type', 'Engines', 'Min', 'Max'], [[c['page'], c['leg'], c['doc_type'], c['engines'], fmt(c['min']), fmt(c['max'])] for c in cases])
    log.info('These may be hard pages or reference/scorer defects. Exclusion is a sensitivity check, not a corrected ranking. Costs are unchanged.')
    log.info('\nDOCUMENT TYPE COVERAGE — per-engine/type scores are available with --report-details')
    coverage = defaultdict(list)
    for row in selected:
        coverage[(row['leg'], row.get('tier'), row['doc_type'])].append(row)
    table(['Leg', 'Tier', 'Document type', 'Pages', 'Scored engine/pages', 'Errors', 'N/A/pending'],
          [[leg, tier, kind, len({r['document_path'] for r in records}), sum(valid(r) for r in records),
            sum(bool(r.get('error')) for r in records), sum(not valid(r) and not r.get('error') for r in records)]
           for (leg, tier, kind), records in sorted(coverage.items())])
    log.info('\nSUPPORTING METRICS — same document-type weighting; CER/WER lower is better')
    supporting = []
    for engine, leg in sorted({(r['engine_name'], r['leg']) for r in selected}):
        for metric in SUPPORT.get(leg, []):
            groups = defaultdict(list)
            for row in selected:
                value = (row.get('metrics') or {}).get(metric)
                if row['engine_name'] == engine and row['leg'] == leg and valid(row) and isinstance(value, (int, float)) and math.isfinite(value):
                    groups[(row.get('tier'), row['doc_type'])].append(value)
            means = [sum(v)/len(v) for v in groups.values()]
            if means:
                supporting.append([engine, leg, metric, fmt(sum(means)/len(means)), sum(map(len, groups.values()))])
    table(['Engine', 'Leg', 'Metric', 'Value', 'N'], supporting)
    log.info('\nCOST AND LATENCY — successful saved primary pass, steady-state where recorded; excludes judge/setup spend')
    ops = rb.engine_operational(inference)
    table(['Engine', 'Primary $/page', 'Primary seconds/page', 'All-pass $/page', 'Pages'],
          [[e, money(v['cost_per_page']), fmt(v['latency_s']), money(v['cost_all_passes']), v['documents']] for e, v in sorted(ops.items())])
    log.info('\nPER-PASS COST AND LATENCY — a shared pass serves several legs without paying again')
    served = defaultdict(set)
    for r in inference:
        served[(r['engine_name'], r.get('pass_name') or 'default')].update(r.get('legs') or [])
    table(['Engine', 'Pass', 'Legs served', '$/page', 'Seconds/page', 'Pages'],
          [[e, p, ','.join(sorted(served[e,p])), money(v['cost_per_page']), fmt(v['latency_s']), v['documents']] for e, op in sorted(ops.items()) for p, v in sorted(op['per_pass'].items())])
    log.info(f'\nSelected records: {len(selected)}; scored: {sum(valid(r) for r in selected)}; errors: {sum(bool(r.get("error")) for r in selected)}; N/A/pending: {sum(not valid(r) and not r.get("error") for r in selected)}.')
    rb._print_inference_errors(inference, log)
    log.info('Inspect per-document values, returned outputs and references in the local HTML report. Gold/reference labels do not guarantee correctness.')
