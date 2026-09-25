"""Import byte-preserved sources into the flat private inventory."""

import fcntl
import json
from pathlib import Path
import re

from proxybench.sources.sec import validate_url
from proxybench.training.labels import sha


def import_source(path, root, *, sec_url, accession, friendly_filename,
                  retrieval_date=None, complete=False):
    """Require a reviewed completeness decision before admitting a local file."""
    validate_url(sec_url)
    if complete is not True:
        raise ValueError('Source import requires explicit confirmation that the document is complete')
    if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
        raise ValueError('A filing accession is required')
    if (not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,219}', friendly_filename)
            or '..' in friendly_filename or friendly_filename.endswith('.')
            or friendly_filename.split('.')[0].upper() in {'CON','PRN','AUX','NUL',
                *(f'COM{i}' for i in range(1,10)), *(f'LPT{i}' for i in range(1,10))}):
        raise ValueError('Use a portable lowercase filename without directories')
    source = Path(path)
    if source.stat().st_size > 100 * 1024 ** 2:
        raise ValueError('Source exceeds the 100 MiB input limit')
    raw = source.read_bytes()
    beginning = raw[:65536].lower()
    if not raw or any(marker in beginning for marker in
                      (b'file unavailable', b'undeclared automated tool', b'request rate threshold exceeded')):
        raise ValueError('An empty document or SEC failure page is not a source')
    if b'<sec-document>' in beginning and b'</sec-document>' not in raw[-65536:].lower():
        raise ValueError('SEC submission is truncated')
    root = Path(root).resolve()
    folder = root/'data/raw'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'data/source-manifest.json'
    digest = sha(raw)
    location = dict(sec_url=sec_url,original_filename=source.name,accession=accession,
                    retrieval_date=retrieval_date,filing_date=None,form=None,cik=None)
    with (root/'data/.source-import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(schema='source-inventory-v1',sources=[])
        entry = next((item for item in manifest['sources'] if item['sha256']==digest),None)
        if entry is None:
            destination = folder/friendly_filename
            if any(p.name.casefold()==friendly_filename.casefold() for p in folder.iterdir()):
                raise FileExistsError('Source filename already exists')
            with destination.open('xb') as stream:
                stream.write(raw)
            pending = dict(status='held',reason='Release review pending',reviewer=None,review_date=None,supporting_source=None)
            entry = dict(kind='filing',path=destination.relative_to(root).as_posix(),
                         friendly_filename=friendly_filename,sha256=digest,bytes=len(raw),locations=[location],
                         rights_review=dict(pending),privacy_review=dict(pending))
            manifest['sources'].append(entry)
        else:
            retained=(root/entry['path']).resolve()
            if not retained.is_relative_to(folder.resolve()) or sha(retained.read_bytes()) != digest:
                raise ValueError('Retained source identity changed')
            if location not in entry['locations']:
                entry['locations'].append(location)
        temporary=manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        temporary.replace(manifest_path)
        return entry
