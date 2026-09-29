"""Keep reserved test sources out of preparation and training."""

from pathlib import Path
import re

from proxybench.training.labels import read_json, sha

LEDGER = 'data/test-source-ledger.json'
SCHEMA = 'test-source-ledger-v1'


def source_cik(selection):
    value = selection.get('cik')
    match = re.search(r'/[Dd]ata/(\d+)/', selection.get('source_url', ''))
    if value is not None and match and int(value) != int(match[1]):
        raise ValueError('Registrant identity differs from its source URL')
    if value is None:
        value = match[1] if match else None
    return str(int(value)) if value is not None else None


def load_ledger(root, *, initialize=False):
    root = Path(root)
    path = root / LEDGER
    if not path.exists():
        inventory_path = root / 'data/source-manifest.json'
        inventory = read_json(inventory_path.read_bytes()) if inventory_path.exists() else {'sources': []}
        if not initialize and any(Path(s['path']).is_relative_to('data/raw/test') for s in inventory['sources']):
            raise ValueError('Protected test-source ledger is missing')
        return dict(schema=SCHEMA, sources=[])
    ledger = read_json(path.read_bytes())
    if ledger.get('schema') != SCHEMA or not isinstance(ledger.get('sources'), list):
        raise ValueError('Invalid protected test-source ledger')
    for source in ledger['sources']:
        if (not source.get('accession') or not source.get('group_id')
                or not isinstance(source.get('sha256'), list) or not source['sha256']):
            raise ValueError('Incomplete protected test-source identity')
    return ledger


def reject_protected_sources(root, selections, assignments):
    ledger = load_ledger(root)
    sources = ledger['sources']
    accessions = {s['accession'] for s in sources}
    accessions.update(alias for s in sources for alias in s.get('aliases', []))
    groups = {s['group_id'] for s in sources}
    groups.update(s['family_id'] for s in sources if s.get('family_id'))
    hashes = {h for s in sources for h in s['sha256']}
    ciks = {s['cik'] for s in sources if s.get('cik')}
    for accession, assignment in assignments.items():
        if assignment['split'] in {'training', 'development'} and (
                accession in accessions or assignment['group_id'] in groups):
            raise ValueError('Protected test source cannot enter training or development')
    inventory_path = Path(root) / 'data/source-manifest.json'
    inventory = read_json(inventory_path.read_bytes()) if inventory_path.exists() else {'sources': []}
    locations = {source['path']: source['locations'] for source in inventory['sources']}
    documents = []
    for selection in selections:
        documents.append(selection)
        documents.extend(dict(dict(selection, **context), split=selection['split'])
                         for context in selection.get('context_sources', []))
    for selection in documents:
        if selection['split'] not in {'training', 'development'}:
            continue
        path = selection['source_path']
        candidate_ciks = {source_cik(selection)}
        for location in locations.get(path, []):
            candidate_ciks.add(source_cik(dict(cik=location.get('cik'), source_url=location.get('sec_url', ''))))
        if (selection['accession'] in accessions or selection['group_id'] in groups or selection.get('family_id') in groups
                or selection['source_sha256'] in hashes or candidate_ciks & ciks
                or Path(path).is_relative_to('data/raw/test')):
            raise ValueError('Protected test source cannot enter training or development')


def protect_sources(root, selections):
    """Persist reservations before exposing test packets for review."""
    from proxybench.runstate import atomic_json
    root = Path(root)
    ledger = load_ledger(root, initialize=True)
    known = {s['accession']: s for s in ledger['sources']}
    inventory_path = root / 'data/source-manifest.json'
    inventory = read_json(inventory_path.read_bytes()) if inventory_path.exists() else {'sources': []}
    for selection in selections:
        proposed = dict(accession=selection['accession'], group_id=selection['group_id'],
                        cik=source_cik(selection), sha256=[selection['source_sha256']], aliases=[])
        if selection.get('family_id'):
            proposed['family_id'] = selection['family_id']
        for entry in inventory['sources']:
            if any(loc['accession'] == proposed['accession'] for loc in entry['locations']):
                if entry['sha256'] not in proposed['sha256']:
                    proposed['sha256'].append(entry['sha256'])
        previous = known.get(proposed['accession'])
        if previous:
            if previous['group_id'] != proposed['group_id'] or previous.get('cik') != proposed['cik']:
                raise ValueError('Protected test identity changed')
            if proposed.get('family_id'):
                if previous.get('family_id') not in (None, proposed['family_id']):
                    raise ValueError('Protected test provider family changed')
                previous['family_id'] = proposed['family_id']
            previous['sha256'] = sorted(set(previous['sha256']) | set(proposed['sha256']))
        else:
            ledger['sources'].append(proposed)
            known[proposed['accession']] = proposed
    atomic_json(root / LEDGER, ledger)
    return dict(path=LEDGER, sha256=sha((root / LEDGER).read_bytes()))
