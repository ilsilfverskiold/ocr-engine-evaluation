"""Local table review server. Saves gold in documents/table_references/gold; no browser file APIs."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
from urllib.parse import urlparse
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


class Conflict(ValueError):
    pass


class TableMarkup(HTMLParser):
    tags = {'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption',
            'br', 'strong', 'b', 'em', 'i', 'span', 'code', 'sub', 'sup'}

    def __init__(self):
        super().__init__()
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        if tag not in self.tags:
            raise ValueError(f'Unsupported table tag: {tag}')
        self.tables += tag == 'table'
        for key, value in attrs:
            if key not in {'rowspan', 'colspan'} or not value or not value.isdigit() or not 1 <= int(value) <= 10000:
                raise ValueError('Only positive rowspan/colspan attributes are allowed')


class GoldStore:
    def __init__(self, versions: Path, gold: Path):
        self.versions, self.gold = versions, gold
        self.manifest = json.loads((versions / 'manifest.json').read_text())
        self.pages = set(self.manifest['pages'])
        self.lock = threading.Lock()
        self.gold.mkdir(parents=True, exist_ok=True)

    def path(self, page):
        if page not in self.pages:
            raise ValueError('Unknown reference page')
        return self.gold / f'{page}.json'

    def read(self, page):
        path = self.path(page)
        if not path.exists():
            return None, None
        raw = path.read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    @staticmethod
    def atomic_write(path, raw):
        fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def validate_record(self, record):
        """Subclasses may review other reference types using the same save protocol."""
        markup = record.get('table_html')
        if not isinstance(markup, str):
            raise ValueError('table_html must be a string')
        validator = TableMarkup()
        validator.feed(markup)
        if markup.strip() and not validator.tables:
            raise ValueError('Nonempty reference must contain a table')

    def save(self, page, record, expected_revision):
        path = self.path(page)
        old = json.loads((self.versions / 'old' / f'{page}.json').read_text())
        if (record.get('page') != page or record.get('version') != 1 or
                record.get('dataset_id') != self.manifest['dataset_id'] or
                record.get('image_sha256') != old['image_sha256']):
            raise ValueError('Reference identity does not match the source page')
        self.validate_record(record)
        raw = (json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode()
        with self.lock:
            previous, revision = self.read(page)
            if revision != expected_revision:
                raise Conflict('Gold changed in another tab. Reload saved gold before saving your draft.')
            if previous is not None:
                history = self.gold / 'history'
                history.mkdir(exist_ok=True)
                self.atomic_write(history / f'{page}__{uuid.uuid4().hex}.json', path.read_bytes())
            self.atomic_write(path, raw)
            if path.read_bytes() != raw:
                raise OSError('Saved reference failed verification')
        return {'record': record, 'revision': hashlib.sha256(raw).hexdigest(), 'path': str(path), 'backed_up': previous is not None}


def handler_for(store):
    token = secrets.token_urlsafe(32)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(store.versions), **kwargs)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def reply(self, status, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if urlparse(self.path).path == '/api/state':
                try:
                    records, revisions = {}, {}
                    with store.lock:
                        for page in sorted(store.pages):
                            record, revision = store.read(page)
                            if record is not None:
                                records[page], revisions[page] = record, revision
                    self.reply(200, {'dataset_id': store.manifest['dataset_id'], 'gold': records,
                                     'revisions': revisions, 'folder': str(store.gold), 'token': token})
                except Exception as exc:
                    self.reply(500, {'error': str(exc)})
                return
            if urlparse(self.path).path in {'/', '/gold-review.html', '/table-reference-comparison.html'}:
                self.path = '/review.html'
            super().do_GET()

        def do_POST(self):
            if self.headers.get('X-Review-Token') != token:
                self.reply(403, {'error': 'Reload the review page before saving.'})
                return
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                self.reply(403, {'error': 'Save must come from the local review page.'})
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Expected JSON')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                if urlparse(self.path).path != '/api/save':
                    self.reply(404, {'error': 'Unknown action'})
                    return
                self.reply(200, store.save(data['page'], data['record'], data.get('expected_revision')))
            except Conflict as exc:
                self.reply(409, {'error': str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception as exc:
                self.reply(500, {'error': str(exc)})

    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    store = GoldStore(ROOT / 'documents/table_references', ROOT / 'documents/table_references/gold')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(store))
    url = f'http://127.0.0.1:{args.port}/review.html'
    print(f'Review: {url}\nSave folder: {store.gold}\nNo internet required.', flush=True)
    if args.open:
        threading.Timer(.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
