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


# Label, saved metric (or derived diagnostic), direction. These never blend into a score.
METRICS = {
    'text': [('CER', 'cer', False), ('WER', 'wer', False), ('Word F1', 'word_f1', True)],
    'markdown': [('Text CER', 'markdown_cer', False), ('Text WER', 'markdown_wer', False), ('Block F1', 'element_f1', True), ('Order', 'order_score', True)],
    'tables': [('Strict F1', 'cell_f1', True), ('Extra rate', 'extra_cell_rate', False), ('Structure', 'structure_similarity', True)],
    'geometry': [('Precision', 'geometry_precision', True), ('F1', 'geometry_f1', True), ('Box CER', 'geometry_cer', False), ('Box WER', 'geometry_wer', False)],
    'key_value': [('Missing', 'missing_rate', False), ('Wrong', 'wrong_rate', False), ('Unmapped', 'kv_unresolved_keys', False)],
    'schema': [('Missing', 'missing_rate', False), ('Wrong', 'wrong_rate', False), ('Extra/page', 'schema_extra_fields', False), ('Valid JSON', 'schema_valid_json', True)],
}
EXTRA_METRICS = {
    'key_value': [('Populated fields', 'kv_populated_accuracy', True), ('Blank fields', 'kv_blank_accuracy', True)],
    'text': [('Word P', 'word_precision', True), ('Word R', 'word_recall', True)],
    'tables': [('Strict P', 'cell_precision', True), ('Strict R', 'cell_recall', True)],
    'markdown': [('Block P', 'element_precision', True), ('Block R', 'element_recall', True)],
}
METHOD = {
    'text': 'Distinct-token coverage; CER/WER measure sequence errors separately.',
    'markdown': 'Correct structure + half partial credit per reference element; text fidelity is separate.',
    'tables': 'Reference value credit, including blanks and partial text matches. Labels and headers align supported split, merged or reordered fields; identical physical grids are not required. Extras, strict F1 and structure are separate diagnostics.',
    'key_value': 'Correct values / printed-label fields. Unmapped is a count of returned keys per page.',
    'schema': 'Correct values / (requested fields + unexpected fields). Same frozen field sample.',
    'geometry': 'Located reference-unit recall. Word/line units and layout-only predictions remain distinct.',
}


def metric_value(row, key):
    if key == 'score':
        return row.get('score')
    m = row.get('metrics') or {}
    if key in ('missing_rate', 'wrong_rate'):
        prefix = 'kv' if row['leg'] == 'key_value' else 'schema'
        total = m.get(prefix + ('_pairs_total' if prefix == 'kv' else '_fields_total'))
        name = prefix + ('_pairs_' if prefix == 'kv' else '_fields_')
        count = m.get(name + ('missing' if key == 'missing_rate' else 'wrong_value'))
        return count / total if total and count is not None else None
    return m.get(key)


def mean_metric(records, key='score'):
    groups = defaultdict(list)
    for row in records:
        value = metric_value(row, key)
        if valid(row) and isinstance(value, (int, float)) and math.isfinite(value):
            groups[(row.get('tier'), row['doc_type'])].append(float(value))
    # Preserve the published headline's rounding and type weighting.
    means = [(round(sum(v)/len(v), 4) if key == 'score' else sum(v)/len(v)) for v in groups.values()]
    return sum(means)/len(means) if means else None


def _order(values, higher=True):
    return sorted(values, key=lambda e: (values[e] is None,
                  (-values[e] if higher else values[e]) if values[e] is not None else 0, e))


def _ranks(values, higher=True):
    return {e: 1 + sum((v > value if higher else v < value) for v in values.values() if v is not None)
            for e, value in values.items() if value is not None}


def _operations(records, inference):
    """Use the actual recorded pass serving each scored page, not today's planner."""
    index = {(r['engine_name'], r['document_path'], r.get('pass_name') or 'default'): r
             for r in inference if not r.get('error')}
    costs, times = defaultdict(list), defaultdict(list)
    for row in records:
        if not valid(row):
            continue
        engine = row['engine_name']
        call = index.get((engine, row['document_path'], row.get('pass_name') or 'default'))
        if call is None:
            continue
        for target, steady, fallback in [(costs, 'steady_cost_per_page', 'cost_per_page'),
                                         (times, 'steady_latency_seconds', 'latency_seconds')]:
            value = call.get(steady)
            if value is None:
                value = call.get(fallback)
            if isinstance(value, (int, float)) and math.isfinite(value):
                target[engine].append(value)
    return {e: {'cost': sum(costs[e])/len(costs[e]) if costs[e] else None,
                'latency': sum(times[e])/len(times[e]) if times[e] else None,
                'cost_n': len(costs[e]), 'latency_n': len(times[e])}
            for e in {r['engine_name'] for r in records}}


def print_report(rows, inference, log, legs, *, sort_by='score', details=False):
    """Engine comparison tables, grouped by leg, then difficulty tier."""
    analysis = comparison(rows)  # Never choose exclusions independently per engine or tier.
    excluded = {(c['document_path'], c['leg']) for c in analysis['cases']}
    selected = [r for r in rows if r['leg'] in legs]
    engines = sorted({r['engine_name'] for r in selected})
    fmt = lambda value: 'N/A' if value is None else f'{value:.4f}'

    def table(heads, data):
        data = [[str(v) for v in row] for row in data]
        widths = [max([len(h), *[len(row[i]) for row in data]]) for i, h in enumerate(heads)]
        def line(row):
            return '  ' + ' | '.join(v.ljust(widths[i]) if i <= 1 else v.rjust(widths[i]) for i, v in enumerate(row))
        log.info(line(heads))
        log.info('  ' + '-+-'.join('-' * w for w in widths))
        for row in data:
            log.info(line(row))

    def title(label):
        log.info('\n' + label)
        log.info('─' * min(len(label), 110))

    log.info('\nOCR BENCHMARK — ENGINE COMPARISON')
    log.info(f'{len(engines)} engines/configurations · {len({r["document_path"] for r in selected})} recorded pages · {len(legs)} leg{"s" if len(legs) != 1 else ""}')
    log.info('↑ higher is better; ↓ lower is better; * best value (ties included). N/A ranks last.')
    log.info('Full = all scored pages. No outl = same scores without the listed low-scoring page/leg cases.')
    log.info('Means weight document families equally. Read coverage alongside rankings; no cross-leg blended score.')

    # One overview matrix, as in the earlier report. Each leg is ranked independently below.
    means = {(e, leg): mean_metric([r for r in selected if r['engine_name'] == e and r['leg'] == leg])
             for e in engines for leg in legs}
    labels = {'text': 'Text', 'markdown': 'Markdown', 'tables': 'Tables', 'key_value': 'KV', 'schema': 'Schema', 'geometry': 'Geometry'}
    best = {l: max((means[e,l] for e in engines if means[e,l] is not None), default=None) for l in legs}
    if len(legs) > 1:
        title('AT A GLANCE — full scores; column winners marked *')
        overview_order = _order({e: means[e, legs[0]] for e in engines})
        table(['Engine', *[labels[l] + ' ↑' for l in legs]],
              [[e, *[fmt(means[e,l]) + ('*' if means[e,l] is not None and means[e,l] == best[l] else '') for l in legs]] for e in overview_order])
        log.info(f'  Ordered by {labels[legs[0]]} full score. Each leg is ranked independently below.')

    for leg in legs:
        records = [r for r in selected if r['leg'] == leg]
        if not records:
            continue
        scored_pages = {r['document_path'] for r in records if valid(r)}
        title(labels[leg].upper() + f' — {len(scored_pages)} scored pages')
        log.info('  ' + METHOD[leg])
        cases = [c for c in analysis['cases'] if c['leg'] == leg]
        if cases:
            log.info('  Outlier comparison excludes: ' + '; '.join(f"{c['page']} ({c['doc_type']}, {c['engines']} engines, max {c['max']:.3f})" for c in cases))
        else:
            log.info('  No outlier exclusions in this leg; Full and No outl are identical.')
        tiers = [t for t in ('easy', 'medium', 'hard') if any(valid(r) and r['tier'] == t for r in records)]
        split = len(scored_pages) >= 24 and len(tiers) > 1
        scopes = [('Overall', records)]
        if split:
            scopes += [(t.upper(), [r for r in records if r['tier'] == t]) for t in tiers]
        else:
            log.info('  Small/single-tier leg: shown together rather than split into sparse tier tables.')

        for scope, group in scopes:
            buckets = {e: [r for r in group if r['engine_name'] == e] for e in engines}
            # Keep all-error participants visible; omit engines with only N/A rows.
            participants = [e for e, rs in buckets.items() if any(valid(r) or r.get('error') for r in rs)]
            if not participants:
                log.info(f'\n  {scope}: no scored or failed participants.')
                continue
            absent = [e for e, rs in buckets.items() if rs and e not in participants]
            full = {e: mean_metric(buckets[e]) for e in participants}
            kept = {e: [r for r in buckets[e] if case_key(r) not in excluded] for e in participants}
            trimmed = {e: mean_metric(kept[e]) for e in participants}
            metrics = METRICS[leg] + (EXTRA_METRICS.get(leg, []) if details else [])
            values = {key: {e: mean_metric(buckets[e], key) for e in participants} for _, key, _ in metrics}
            ops = _operations(group, inference)
            sort_values, higher, sort_label = full, True, 'Full (highest first)'
            metric_alias = {'cer': next((key for _, key, _ in metrics if key.endswith('cer')), None),
                            'wer': next((key for _, key, _ in metrics if key.endswith('wer')), None),
                            'f1': next((key for _, key, _ in metrics if key.endswith('f1')), None)}
            if sort_by == 'without-outliers':
                sort_values, sort_label = trimmed, 'No outl (highest first)'
            elif sort_by in ('cost', 'latency'):
                sort_values = {e: ops[e][sort_by] for e in participants}
                higher, sort_label = False, sort_by + ' (lowest first)'
            elif metric_alias.get(sort_by):
                key = metric_alias[sort_by]
                sort_values, higher, sort_label = values[key], sort_by == 'f1', sort_by.upper() + (' (highest first)' if sort_by == 'f1' else ' (lowest first)')
            metric_counts = {key: {e: sum(valid(r) and isinstance(metric_value(r, key), (int, float))
                                                and math.isfinite(metric_value(r, key)) for r in buckets[e])
                                   for e in participants} for _, key, _ in metrics}
            ordered = _order(sort_values, higher)
            ranks = _ranks(sort_values, higher)
            def marked(value, all_values, direction=True):
                present = [v for v in all_values.values() if v is not None]
                top = (max(present) if direction else min(present)) if present else None
                return fmt(value) + ('*' if value is not None and value == top else '')
            log.info(f'\n  {scope} — ranked by {sort_label}')
            table(['#', 'Engine', 'Full ↑', 'No outl ↑', 'Pages→kept', *[label + (' ↑' if up else ' ↓') for label, _, up in metrics], 'Err'],
                  [[ranks.get(e, '—'), e, marked(full[e], full), marked(trimmed[e], trimmed),
                    f'{sum(valid(r) for r in buckets[e])}→{sum(valid(r) for r in kept[e])}',
                    *[marked(values[key][e], values[key], up) + (f' ({metric_counts[key][e]})' if 0 < metric_counts[key][e] < sum(valid(r) for r in buckets[e]) else '') for _, key, up in metrics],
                    sum(bool(r.get('error')) for r in buckets[e])] for e in ordered])
            if leg in ('schema', 'key_value'):
                log.info('  Missing/Wrong are field fractions; Unmapped/Extra are mean counts.')
            lost = []
            for e in participants:
                before = {(r['tier'],r['doc_type']) for r in buckets[e] if valid(r)}
                after = {(r['tier'],r['doc_type']) for r in kept[e] if valid(r)}
                if before-after:
                    lost.append(e + ': ' + ', '.join(sorted(t for _,t in before-after)))
            if lost:
                log.info('  Types removed entirely by exclusion: ' + '; '.join(lost))
            if scope == 'Overall':
                if absent:
                    log.info('  N/A only (not ranked): ' + ', '.join(absent))
                cost_values = {e: ops[e]['cost'] for e in participants}
                time_values = {e: ops[e]['latency'] for e in participants}
                cost_rank, time_rank = _ranks(cost_values, False), _ranks(time_values, False)
                order = _order(time_values if sort_by == 'latency' else cost_values, False)
                log.info('\n  Cost & latency — actual pass for this leg; ' + ('fastest' if sort_by == 'latency' else 'cheapest') + ' first')
                table(['Cost #', 'Engine', '$/page ↓', 'Time #', 's/page ↓', 'Cost/time N'],
                      [[cost_rank.get(e,'—'), e, 'N/A' if ops[e]['cost'] is None else f"{ops[e]['cost']:.6f}",
                        time_rank.get(e,'—'), 'N/A' if ops[e]['latency'] is None else f"{ops[e]['latency']:.2f}",
                        f"{ops[e]['cost_n']}/{ops[e]['latency_n']}"] for e in order])
            if details:
                kinds = sorted({r['doc_type'] for r in group if valid(r)})
                if len(kinds) > 1:
                    log.info('  Document families — full score (scored pages); same engine order')
                    # Wrap wide matrices instead of truncating document names.
                    for i in range(0, len(kinds), 4):
                        part = kinds[i:i+4]
                        table(['Engine', *part], [[e, *[fmt(mean_metric([r for r in buckets[e] if r['doc_type']==kind])) + f" ({sum(valid(r) and r['doc_type']==kind for r in buckets[e])})" for kind in part]] for e in ordered])

    title('HOW TO READ THIS COMPARISON')
    log.info('  Tiers split only with ≥24 scored pages and more than one represented tier. Families appear with --report-details.')
    log.info('  Outliers: ≥3 scored engines, all ≤0.25, no failed/pending participant. Same excluded cases for all engines.')
    log.info('  Low scores may reflect hard pages, reference errors or scorer limits; Full remains the primary result.')
    log.info('  Costs/times use successful scored pages and their saved serving pass; steady-state where recorded.')
    log.info('  Shared calls are paid once; do not sum costs across legs. Failed-call, startup and judge spend are excluded.')
    log.info('  Supporting metrics use Full pages; a parenthesized count indicates fewer metric-reference pages.')
    log.info('  No outl can decrease: exclusions change family means. Local $0 excludes machine expense.')
    log.info('  Errors and N/A are excluded from means, not scored as zeros. Unequal page coverage can change rankings.')
    log.info('  --report-sort cer|wer|f1|without-outliers|cost|latency changes ordering; unsupported metrics fall back to Full.')
    log.info('  --report-details adds family breakdowns, other metrics, diagnostics and local CSV exports.')
