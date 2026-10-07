"""Match native record-table fields by printed row identity and column header.

No reference value participates in field selection. The ordinary value checker
runs afterwards, including for blanks. Physical source coordinates are retained.
"""
from collections import defaultdict
from evaluation.shared_fields import reference_fields
from evaluation.structured_metrics import normalize_label


def native_field_matches(gt, pairs, contexts):
    fields = defaultdict(list)
    for field in reference_fields(gt, 'key_value'):
        if ' — ' in field.path:
            owner, label = field.path.rsplit(' — ', 1)
            fields[(normalize_label(owner), normalize_label(label))].append(field.path)
    # A first row that is itself a known record is not a column header.
    # Leave headerless question/answer grids on their existing KV path.
    owners = {owner for owner, _ in fields}
    headerless = {ctx['table'] for ctx in contexts if ctx
                  and normalize_label(ctx.get('row_header', '')) in owners}
    candidates = defaultdict(list)
    rows = defaultdict(set)
    for ctx in contexts:
        if ctx and ctx.get('row_key'):
            rows[(ctx['table'], normalize_label(ctx['row_key']))].add(ctx['row'])
    for index, ctx in enumerate(contexts):
        if not ctx or ctx['table'] in headerless or not ctx.get('row_key') or not ctx.get('column_header'):
            continue
        owner, header = normalize_label(ctx['row_key']), normalize_label(ctx['column_header'])
        if len(rows[(ctx['table'], owner)]) != 1:
            continue
        targets = fields.get((owner, header), [])
        if len(targets) == 1:
            candidates[targets[0]].append((index, ctx))
    matches = {}
    for path, hits in candidates.items():
        # Mirrored native HTML views can describe the same physical cell.
        # Different rows/columns, or conflicting renderings, remain ambiguous.
        identities = {(c['table'], c['row'], c['column']) for _, c in hits}
        values = {str(pairs[i][1]) for i, _ in hits}
        if len(identities) != 1 or len(values) != 1:
            continue
        index, ctx = hits[0]
        matches[path] = {'index': index, 'source': ctx,
            'how': f"native table {ctx['table']+1}, row {ctx['row']+1}, column {ctx['column']+1}: "
                   f"printed row identity {ctx['row_key']!r} + header {ctx['column_header']!r}"}
    return matches
