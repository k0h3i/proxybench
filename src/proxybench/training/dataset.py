"""Read a self-contained dataset and rebuild every source message."""

from pathlib import Path

from proxybench.annotation.bindings import checked_file
from proxybench.annotation.historical import VIEW_VERSION, literal_support, prepare_historical
from proxybench.training.labels import dumps, read_json, sha, validate

SCHEMA = 'training-dataset-v1'
COORDINATES = 'zero-based-half-open-source-bytes'
FILES = {'training': 'training-examples.jsonl', 'development': 'development-examples.jsonl'}


def check_assignments(selections, assignments):
    """Enforce source groups and all known exposure, including unused sources."""
    groups = {}
    exposed = {a['group_id'] for a in assignments.values() if a['development_exposed']}
    for identity, assignment in assignments.items():
        split = assignment['split']
        if split not in {'training', 'development', 'test', 'excluded'}:
            raise ValueError('Unknown source assignment partition')
        if assignment['group_id'] in exposed and split == 'training':
            raise ValueError('Development-exposed source group cannot enter training')
        if split != 'excluded':
            if groups.setdefault(assignment['group_id'], split) != split:
                raise ValueError('A source group crosses partitions')
    targets = {}
    for selection in selections:
        assignment = assignments[selection['accession']]
        if any(selection[k] != assignment[k] for k in ('split', 'group_id')):
            raise ValueError('Selection differs from its source assignment')
        a, b = selection['target']
        prior = targets.setdefault(selection['source_sha256'], [])
        if any(a < y and x < b for x, y in prior):
            raise ValueError('Duplicate or overlapping source targets')
        prior.append((a, b))


def preserve_exposure(root, assignments):
    """Carry the existing exposure ledger into every future preparation."""
    from proxybench.sources.holdout import reject_protected_sources
    reject_protected_sources(root, [], assignments)
    path = Path(root)/'data/training-dataset/dataset-manifest.json'
    if not path.exists():
        check_assignments([], assignments)
        return assignments
    known = read_json(path.read_bytes())['assignments']
    merged = {key: dict(value) for key, value in known.items()}
    exposed_groups = {a['group_id'] for a in known.values() if a['development_exposed'] or a['split'] == 'development'}
    for identity, proposed in assignments.items():
        previous = known.get(identity)
        if previous and previous['group_id'] != proposed['group_id']:
            raise ValueError('Known source groups cannot be changed during preparation')
        value = dict(proposed)
        if value['group_id'] in exposed_groups:
            value['development_exposed'] = True
        merged[identity] = value
    check_assignments([], merged)
    return merged


def project_file(root, binding):
    path = Path(binding['path'])
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Dataset paths must be project-relative')
    return checked_file(root, str(path), binding['sha256'])


def read_release(path, *, project_root=None):
    """Return exact rows and source-checked, hydrated example metadata."""
    return _read_release(path, project_root=project_root)


def _read_release(path, *, project_root=None, schema=SCHEMA, files=FILES):
    """Share source reconstruction without admitting test data to training."""
    path = Path(path).resolve()
    root = Path(project_root or Path.cwd()).resolve()
    manifest = read_json((path / 'dataset-manifest.json').read_bytes())
    if manifest.get('schema') != schema or manifest.get('status') != 'COMPLETE':
        raise ValueError('Incomplete or unsupported dataset')
    if manifest.get('coordinate_convention') != COORDINATES or manifest.get('rendering_version') != VIEW_VERSION:
        raise ValueError('Unsupported source coordinates or rendering version')
    prompt = project_file(root, manifest['system_prompt']).decode('utf-8')
    project_file(root, manifest['label_contract'])
    if set(manifest['files']) != set(files):
        raise ValueError('Dataset files differ from its declared schema')
    rows, hydrated = {}, []
    for split, name in files.items():
        file = manifest['files'][split]
        if file['path'] != name:
            raise ValueError('Unexpected dataset filename')
        raw = checked_file(path, name, file['sha256'])
        rows[split] = [read_json(line) for line in raw.splitlines()]
        if len(rows[split]) != file['examples']:
            raise ValueError('Dataset row count differs')
    seen = set()
    for meta in manifest['examples']:
        split, position, selection = meta['split'], meta['position'], meta['selection']
        if split not in rows or not isinstance(position, int) or not 0 <= position < len(rows[split]):
            raise ValueError('Invalid dataset row position')
        if (split, position) in seen:
            raise ValueError('Duplicate dataset row position')
        seen.add((split, position))
        row = rows[split][position]
        if sha(dumps(row).encode()) != meta['row_sha256']:
            raise ValueError('Dataset row content changed')
        if selection['split'] != split:
            raise ValueError('Source and row partitions differ')
        packet = prepare_historical(root, selection)
        messages = row['messages']
        if ([m['role'] for m in messages] != ['system', 'user', 'assistant']
                or messages[0]['content'].encode() != prompt.encode()
                or messages[1]['content'] != packet['model_input']):
            raise ValueError('Dataset messages differ from the prompt or original source')
        literal_support(validate(read_json(messages[2]['content'])), packet)
        hydrated.append(dict(meta, packet=packet))
    if len(seen) != sum(map(len, rows.values())):
        raise ValueError('Missing dataset row metadata')
    check_assignments([e['selection'] for e in manifest['examples']], manifest['assignments'])
    if schema == SCHEMA:
        from proxybench.sources.holdout import reject_protected_sources
        reject_protected_sources(root, [e['selection'] for e in manifest['examples']], manifest['assignments'])
    manifest['examples'] = hydrated
    return rows, manifest
