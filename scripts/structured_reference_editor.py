"""Serve the schema/KV review editor and save to documents/structured_references/gold."""
from __future__ import annotations

import argparse
import copy
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.table_reference_editor import GoldStore, handler_for
from evaluation.structured_references import field_patterns, validate_structured_record


class StructuredGoldStore(GoldStore):
    def validate_record(self, record):
        validate_structured_record(record)

    def save(self, page, record, expected_revision):
        # Record original reference scope server-side: new targets must not be
        # mistaken for fields requested in historical engine calls.
        self.path(page)
        original = json.loads((self.versions / 'old' / f'{page}.json').read_text())
        record = copy.deepcopy(record)
        if 'true_json' in original and 'json_schema' in original:
            validate_structured_record(record)
            baseline = field_patterns(original)
            record['historical_reference_schema'] = original['json_schema']
            record['historical_field_patterns'] = sorted(baseline)
            record['new_schema_field_patterns'] = sorted(field_patterns(record) - baseline)
            record['schema_scope_note'] = (
                'Baseline is the original reference schema, not proof of every historical request. '
                'Check the actual saved request before scoring expanded targets.')
        return super().save(page, record, expected_revision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    versions = ROOT / 'documents/structured_references'
    store = StructuredGoldStore(versions, versions / 'gold')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(store))
    url = f'http://127.0.0.1:{args.port}/review.html'
    print(f'Review: {url}\nSave folder: {store.gold}\nNo internet required.', flush=True)
    if args.open:
        threading.Timer(.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
