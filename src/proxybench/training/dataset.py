"""Admit reviewed historical labels and publish a complete local release."""

import math
import os
from collections import Counter
from pathlib import Path
import tempfile

from proxybench.annotation.bindings import checked_file, review_binding
from proxybench.annotation.historical import check_packet, literal_support
from proxybench.training.labels import dumps, from_review, read_json, sha, validate
from proxybench.training.sequences import sequence


def draft_gate(outcomes, *, calibration=False):
    """Count affected first attempts, including attempts without usable output."""
    n = len(outcomes)
    if not n or (calibration and n != 8) or (not calibration and n > 12):
        raise ValueError('Expected eight calibration targets or one to twelve production targets')
    counts = {key: sum(bool(row.get(key)) for row in outcomes)
              for key in ('source_value_error', 'wording_origin_error')}
    counts['unusable'] = sum(not row.get('usable', False) for row in outcomes)
    thresholds = {'source_value_error': 3 if calibration else math.ceil(n / 6),
                  'wording_origin_error': 4 if calibration else math.ceil(n / 3),
                  'unusable': 1 if calibration else math.ceil(n / 6)}
    failed = [k for k, v in counts.items() if v >= thresholds[k]]
    if calibration and sum(row.get('accepted', False) for row in outcomes) != 8:
        failed.append('accepted_count')
    if any(row.get('unresolved_policy', False) for row in outcomes):
        failed.append('unresolved_policy')
    return dict(passed=not failed, scheduled=n, counts=counts, thresholds=thresholds, failed=failed)


def length_report(tokenizer, prompt, response):
    lengths = sequence(tokenizer, prompt, response, context_cap=5120)
    result = {key: lengths[key] for key in ('input_tokens', 'response_tokens', 'combined_tokens')}
    if result['input_tokens'] > 3328 or result['response_tokens'] > 1792:
        raise ValueError(f'Sequence exceeds a prompt or response limit: {result}')
    return result


def check_assignments(packets, assignments):
    """Mandatory source links must already be resolved into frozen groups."""
    group_splits, identities, targets = {}, set(), {}
    filing_counts = Counter()
    for packet in packets:
        m = packet['manifest']
        filing_counts[m['accession']] += 1
        if filing_counts[m['accession']] > 12:
            raise ValueError('A filing exceeds the 12-target selection cap')
        assignment = assignments[m['accession']]
        if assignment['split'] != m['split'] or assignment['group_id'] != m['group_id']:
            raise ValueError('Packet differs from its frozen source assignment')
        if assignment.get('development_exposed') and m['split'] != 'legacy_development':
            raise ValueError('Development exposure cannot enter a new partition')
        prior = group_splits.setdefault(m['group_id'], m['split'])
        if prior != m['split']:
            raise ValueError('A source group crosses partitions')
        identity = (m['accession'], m['source_sha256'], tuple(m['target']))
        if identity in identities:
            raise ValueError('Duplicate source target')
        identities.add(identity)
        previous = targets.setdefault(m['source_sha256'], [])
        a, b = m['target']
        if any(a < y and x < b for x, y in previous):
            raise ValueError('Overlapping source targets')
        previous.append((a, b))
    # Check the complete supplied exposure ledger, including unselected sources.
    group_splits = {}
    exposed = {a['group_id'] for a in assignments.values() if a.get('development_exposed')}
    for a in assignments.values():
        if group_splits.setdefault(a['group_id'], a['split']) != a['split']:
            raise ValueError('A source group crosses partitions in the assignment ledger')
        if a['group_id'] in exposed and a['split'] != 'legacy_development':
            raise ValueError('A linked source inherits development exposure')


def publish(root, packets, policy, review_path, approval_path, output, *, assignments, evidence, tokenizer):
    """Write only exact accepted exports. Preserve incomplete staging on failure.

    Evidence binds raw drafts, corrections, audits, and the release gate report.
    Semantic review and quality decisions remain coordinator responsibilities.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    review_raw, approval_raw = Path(review_path).read_bytes(), Path(approval_path).read_bytes()
    review, approval = read_json(review_raw), read_json(approval_raw)
    if (approval.get('decision') != 'ACCEPTED' or approval.get('export_sha256') != sha(review_raw)
            or not approval.get('reviewer') or not approval.get('accepted_at')
            or not approval.get('delegation') or review.get('schema') != 'training-review-v1'):
        raise ValueError('Acceptance must bind this exact review and identify its reviewer and authority')
    ids = approval.get('accepted_packet_ids')
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError('Acceptance requires distinct packet IDs')
    by_id = {p['manifest']['packet_id']: p for p in packets}
    if len(by_id) != len(packets) or not set(ids) <= by_id.keys():
        raise ValueError('Duplicate or unknown packet identity')
    check_assignments(packets, assignments)
    rows, metadata, blobs = {}, [], {}
    for packet_id in ids:
        packet = by_id[packet_id]
        check_packet(root, packet, policy)
        answer = review['packets'][packet_id]
        if answer.get('reviewed') is not True or answer.get('input_binding') != review_binding(packet):
            raise ValueError('Accepted answer was not reviewed against this exact packet')
        label = from_review(answer['fields'])
        literal_support(label, packet)
        response = dumps(label)
        lengths = length_report(tokenizer, packet['model_input'], response)
        sidecars = evidence[packet_id]
        if not {'draft', 'review_history', 'quality_gate'} <= sidecars.keys():
            raise ValueError('Missing draft, review history, or quality gate evidence')
        for name, binding in sidecars.items():
            raw = checked_file(root, binding['path'], binding['sha256'])
            if name == 'quality_gate' and read_json(raw).get('passed') is not True:
                raise ValueError('A failed quality gate blocks publication')
            blobs[binding['sha256']] = raw
        split = packet['manifest']['split']
        row = dict(messages=[dict(role='user', content=packet['model_input']), dict(role='assistant', content=response)])
        rows.setdefault(split, []).append(row)
        metadata.append(dict(packet=packet, lengths=lengths, evidence=sidecars,
                             row_sha256=sha(dumps(row).encode()), split=split))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + '.publishing')
    lock.mkdir()  # A concurrent or interrupted publisher needs a separate output version.
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.incomplete-', dir=output.parent))
    try:
        (stage / 'evidence').mkdir()
        for digest, raw in blobs.items():
            (stage / 'evidence' / digest).write_bytes(raw)
        (stage / 'review.json').write_bytes(review_raw)
        (stage / 'acceptance.json').write_bytes(approval_raw)
        (stage / 'policy.md').write_text(policy, encoding='utf-8')
        hashes = {}
        for split, items in rows.items():
            raw = ''.join(dumps(row) + '\n' for row in items).encode()
            (stage / (split + '.jsonl')).write_bytes(raw)
            hashes[split] = sha(raw)
        manifest = dict(schema='historical-dataset-v1', status='COMPLETE', examples=metadata,
                        assignments=assignments, dataset_hashes=hashes,
                        policy_sha256=sha(policy.encode()), review_sha256=sha(review_raw),
                        acceptance_sha256=sha(approval_raw))
        (stage / 'manifest.json').write_text(dumps(manifest) + '\n', encoding='utf-8')
        read_release(stage)
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    except BaseException:
        # Preserve staging and its lock as observable incomplete publication.
        raise
    else:
        lock.rmdir()
    return manifest


def read_release(path):
    path = Path(path)
    manifest = read_json((path / 'manifest.json').read_bytes())
    if manifest.get('schema') == 'historical-dataset-system-v1':
        from proxybench.training.system_release import read_system_release
        return read_system_release(path, manifest)
    if manifest.get('schema') != 'historical-dataset-v1' or manifest.get('status') != 'COMPLETE':
        raise ValueError('Incomplete or unsupported dataset')
    for name, key in [('policy.md', 'policy_sha256'), ('review.json', 'review_sha256'), ('acceptance.json', 'acceptance_sha256')]:
        checked_file(path, name, manifest[key])
    rows = {}
    for split, digest in manifest['dataset_hashes'].items():
        if split not in {'training', 'development', 'legacy_development'}:
            raise ValueError('Unknown dataset partition')
        raw = checked_file(path, split + '.jsonl', digest)
        rows[split] = [read_json(line) for line in raw.splitlines()]
        expected = [m for m in manifest['examples'] if m['split'] == split]
        if len(rows[split]) != len(expected):
            raise ValueError('Dataset row count differs from the manifest')
        for row, meta in zip(rows[split], expected):
            if sha(dumps(row).encode()) != meta['row_sha256']:
                raise ValueError('Dataset row identity differs from the manifest')
            messages = row['messages']
            if (len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']
                    or messages[0]['content'] != meta['packet']['model_input']):
                raise ValueError('Dataset messages differ from the frozen packet')
            label = validate(read_json(messages[1]['content']))
            literal_support(label, meta['packet'])
            for binding in meta['evidence'].values():
                checked_file(path, 'evidence/' + binding['sha256'], binding['sha256'])
    check_assignments([m['packet'] for m in manifest['examples']], manifest['assignments'])
    return rows, manifest
