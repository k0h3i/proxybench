"""Download a bounded SEC archive selection through one declared client."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener

GIB = 1024 ** 3


def validate_url(url):
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.hostname not in {'www.sec.gov', 'data.sec.gov'}
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443)):
        raise ValueError('Only official SEC HTTPS URLs without credentials or unsafe ports are allowed')
    return url


class SecRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Validate before urllib can forward the declared client identity.
        validate_url(newurl)
        time.sleep(0.5)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(request, *, timeout):
    return build_opener(SecRedirects()).open(request, timeout=timeout)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def fetch(url, output, state_path, user_agent, *, kind='filing',
          max_bytes=4 * GIB, max_document_bytes=100 * 1024 ** 2, max_filings=60):
    validate_url(url)
    if kind not in {'index', 'filing'} or not user_agent.strip():
        raise ValueError('A declared client identity and download kind are required')
    output, state_path = Path(output), Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            schema='sec-download-ledger-v1', bytes=0, filings=0, events=[], last_request=0, blocked=False)
        if state['blocked']:
            raise RuntimeError('SEC acquisition is stopped after a block response')
        cached = next((e for e in state['events'] if e.get('status') == 'complete' and e['url'] == url
                       and e['output'] == str(output)), None)
        if output.exists():
            if cached and hashlib.sha256(output.read_bytes()).hexdigest() == cached['sha256']:
                return cached
            raise FileExistsError('An existing output does not match the download cache')
        if state['bytes'] >= max_bytes or (kind == 'filing' and state['filings'] >= max_filings):
            raise RuntimeError('SEC acquisition reached its byte or filing ceiling')
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        if available < 4 * GIB:
            raise RuntimeError('Available host memory is below 4 GiB')
        if shutil.disk_usage(state_path.parent).free < max_bytes - state['bytes'] + 2 * GIB:
            raise RuntimeError('Free disk space is below the remaining allowance plus 2 GiB')
        time.sleep(max(0, 0.5 - (time.time() - state['last_request'])))
        event = dict(url=url, output=str(output), kind=kind, started_at=now(), status='started', bytes=0)
        state['events'].append(event)
        state['last_request'] = time.time()
        save(state_path, state)
        response = None
        partial = None
        try:
            try:
                response = urlopen(Request(url, headers={'User-Agent': user_agent,
                                                        'Accept-Encoding': 'identity'}), timeout=45)
            except HTTPError as error:
                response = error
            status = response.status
            event['http_status'] = status
            event['content_type'] = response.headers.get('Content-Type', '')
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                event['retry_after'] = retry_after
            blocked = status in {403, 429} or bool(retry_after)
            if blocked:
                state['blocked'] = True
                event['status'] = 'blocked'
                save(state_path, state)
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(output.name + f'.partial-{len(state["events"]):04d}')
            event['partial_path'] = str(partial)
            digest = hashlib.sha256()
            first = True
            with partial.open('xb') as stream:
                while True:
                    allowance = min(max_document_bytes - event['bytes'], max_bytes - state['bytes'])
                    # Stop at the ceiling without consuming an extra byte.
                    # A response that exactly fills the allowance is deferred.
                    if allowance <= 0:
                        if not blocked:
                            event['status'] = 'byte_limit'
                        raise RuntimeError('Download reached the document or total byte ceiling')
                    block = response.read(min(1024 * 1024, allowance))
                    if not block:
                        break
                    if first and kind == 'filing' and status == 200:
                        state['filings'] += 1
                    first = False
                    event['bytes'] += len(block)
                    state['bytes'] += len(block)
                    stream.write(block)
                    digest.update(block)
                    save(state_path, state)
            event['sha256'] = digest.hexdigest()
            if blocked:
                raise RuntimeError(f'SEC acquisition stopped on HTTP {status}')
            if status != 200:
                event['status'] = 'http_error'
                raise RuntimeError(f'SEC returned HTTP {status}')
            if first:
                event['status'] = 'empty'
                raise RuntimeError('SEC returned an empty document')
            # Recognize a block page even when a proxy preserves HTTP 200.
            with partial.open('rb') as stream:
                beginning = stream.read(65536).lower()
            if any(marker in beginning for marker in (b'undeclared automated tool', b'request rate threshold exceeded', b'file unavailable')):
                state['blocked'] = True
                event['status'] = 'blocked'
                raise RuntimeError('SEC acquisition stopped on a block page')
            if output.exists():
                raise FileExistsError(output)
            partial.rename(output)
            event['status'] = 'complete'
            event.pop('partial_path')
            return event
        except Exception as error:
            if event['status'] == 'started':
                event['status'] = 'transport_error'
            if partial is not None and partial.exists():
                partial.unlink()
            event.pop('partial_path', None)
            # Do not record request headers or contact details.
            event['error_type'] = type(error).__name__
            raise
        finally:
            if response is not None:
                response.close()
            event['ended_at'] = now()
            save(state_path, state)


def index_rows(raw):
    """Read SEC master-index entries without treating accession prefixes as families."""
    for line in raw.decode('latin-1').splitlines():
        parts = line.split('|')
        if len(parts) == 5 and parts[0].strip().isdigit() and parts[2] in {'N-PX', 'N-PX/A'}:
            cik, registrant, form, filed, path = parts
            yield dict(cik=cik.strip(), registrant=registrant, form=form, filed=filed,
                       archive_path=path, accession=Path(path).stem)
