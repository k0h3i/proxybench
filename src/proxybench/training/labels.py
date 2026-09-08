"""Adapt direct agent labels to editable reviews and export accepted examples."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

from proxybench.annotation.bindings import review_binding
from proxybench.annotation.packets import FIELDS
from proxybench.annotation.review import write_review
from proxybench.annotation.xml_packets import build
from proxybench.annotation.xml_view import passage

AVAILABILITY = {'PRESENT', 'ABSENT_IN_CONTEXT', 'AMBIGUOUS', 'UNREADABLE', 'CONFLICTING', 'NOT_APPLICABLE'}
DIRECTIONS = ('FOR', 'AGAINST', 'ABSTAIN', 'WITHHOLD', 'OTHER')
IDENTIFIER = {'source_label': str, 'value': str}
TYPES = dict.fromkeys(FIELDS, str)
TYPES.update(reporting_scope={'name': str, 'scope_type': ('INDIVIDUAL_FUND', 'FUND_GROUP'), 'members': [str]},
             series_identifiers=[IDENTIFIER], security_identifiers=[IDENTIFIER],
             participation=('VOTED', 'DID_NOT_VOTE', 'OTHER'),
             management_recommendation=(*DIRECTIONS, 'NONE'),
             vote_components=[{'direction': DIRECTIONS, 'quantity': {'amount': str, 'unit': str},
                               'disclosed_management_alignment': ('WITH_MANAGEMENT', 'AGAINST_MANAGEMENT', 'NOT_APPLICABLE', 'OTHER')}])


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Invalid JSON constant: {value}')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def exact(value, keys, path):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{path}: unexpected or missing keys')


def check_value(value, kind, path):
    if kind is str:
        if not isinstance(value, str):
            raise ValueError(f'{path}: expected a string')
    elif isinstance(kind, tuple):
        if value not in kind:
            raise ValueError(f'{path}: unknown value')
    elif isinstance(kind, dict):
        exact(value, kind, path)
        for key, child in kind.items():
            check_field(value[key], child, f'{path}.{key}')
    else:
        if not isinstance(value, list) or not value:
            raise ValueError(f'{path}: expected a nonempty list')
        for index, child in enumerate(value):
            function = check_value if isinstance(kind[0], dict) else check_field
            function(child, kind[0], f'{path}[{index}]')


def check_field(field, kind, path):
    exact(field, ('value', 'raw_text', 'availability', 'origin'), path)
    value, state, origin, raw = (field[k] for k in ('value', 'availability', 'origin', 'raw_text'))
    if state not in AVAILABILITY or (raw is not None and not isinstance(raw, str)):
        raise ValueError(f'{path}: invalid availability or raw wording')
    if path == 'vote_components' and value == [] and state == 'NOT_APPLICABLE' and origin == 'DERIVED':
        return
    if state != 'PRESENT':
        if value is not None or origin is not None:
            raise ValueError(f'{path}: unresolved value and origin must be null')
        return
    if origin not in ('EXTRACTED', 'DERIVED'):
        raise ValueError(f'{path}: present value requires an origin')
    check_value(value, kind, path)
    if value == 'OTHER' and not raw:
        raise ValueError(f'{path}: OTHER requires original wording')


def validate(label):
    exact(label, ('fields',), 'label')
    exact(label['fields'], TYPES, 'fields')
    fields = label['fields']
    for key, kind in TYPES.items():
        check_field(fields[key], kind, key)
    if fields['meeting_date']['availability'] == 'PRESENT':
        value = fields['meeting_date']['value']
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError('meeting_date: expected YYYY-MM-DD')
    nonvote = fields['participation']['value'] == 'DID_NOT_VOTE'
    if nonvote != (fields['vote_components']['value'] == []):
        raise ValueError('Explicit nonvoting requires empty vote components, and conversely')
    return label


def to_review(label, notes=None):
    # Preserve draft errors for user correction. Final export performs strict validation.
    exact(label, ('fields',), 'label')
    exact(label['fields'], TYPES, 'fields')
    result = {}
    for key, field in label['fields'].items():
        exact(field, ('value', 'raw_text', 'availability', 'origin'), key)
        result[key] = dict(value=dumps(field['value']), raw_text=dumps(field['raw_text']),
                           availability=field['availability'], origin=field['origin'] or '',
                           evidence_input=(notes or {}).get(key, ''), note='')
    return result


def from_review(fields):
    exact(fields, TYPES, 'review fields')
    label = {'fields': {key: dict(value=read_json(field['value']), raw_text=read_json(field['raw_text']),
                                  availability=field['availability'], origin=field['origin'] or None)
                         for key, field in fields.items()}}
    return validate(label)


def source_text(primary, votes, *, index, input_id):
    packet = build(primary, votes, primary_id='primary_doc.xml', votes_id='votes.xml', index=index, input_id=input_id)
    blocks = packet['bundle']['blocks']
    return ('PRIMARY ATTACHMENT (reporting-series context)\n' + blocks[0]['original_text'] +
            '\nVOTE ATTACHMENT ROOT\n' + blocks[1]['original_text'] + '\nBEGIN MARKED TARGET\n' +
            blocks[2]['original_text'] + '\nEND MARKED TARGET\n' + blocks[3]['original_text'])


def prepare_review(run, primary_path, votes_path, indices, *, accession):
    run = Path(run)
    primary, votes = Path(primary_path).read_bytes(), Path(votes_path).read_bytes()
    contract = (run / 'label-contract.md').read_text()
    packets, drafts = [], {}
    for index in indices:
        packet_id = f'modern-{index:03d}'
        directory = run / packet_id
        source = source_text(primary, votes, index=index, input_id=packet_id)
        model_input = contract + '\n\nExtract only the marked target from this source:\n\n' + source
        if (directory / 'input.txt').read_text() != model_input or (directory / 'source.txt').read_text() != source:
            raise ValueError('Agent source input differs from review source')
        view = passage(primary, votes, index=index)
        manifest = dict(packet_id=packet_id, packet_version='training-label-v1', split='development',
                        accession=accession, source_sha256=sha(source.encode()), selected_index=index,
                        primary_path=str(primary_path), primary_sha256=sha(primary), votes_path=str(votes_path),
                        votes_sha256=sha(votes), blocks=[dict(block_id='source', line_start=1, line_end=len(source.splitlines()))])
        packet = dict(manifest=manifest, source_view=view, model_input=model_input)
        packets.append(packet)
        raw = (directory / 'draft.json').read_bytes()
        notes = read_json((directory / 'notes.json').read_bytes())
        drafts[packet_id] = dict(source_sha256=manifest['source_sha256'], packet_version=manifest['packet_version'],
            packet_fingerprint=sha(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()),
            input_binding=review_binding(packet), fields=to_review(read_json(raw), notes))
    review_dir = run / 'review'
    review_dir.mkdir()
    (review_dir / 'packet-set.json').write_text(dumps(packets) + '\n')
    draft_path = review_dir / 'suggestions.json'
    draft_path.write_text(dumps(dict(draft_set_id=run.name, packets=drafts)) + '\n')
    return write_review(Path.cwd(), packet_directory=review_dir.resolve(), draft_path=draft_path,
                        training_contract=contract)


def export_accepted(run, review_path, approval_path, output):
    run, output = Path(run), Path(output)
    raw = Path(review_path).read_bytes()
    review, approval = read_json(raw), read_json(Path(approval_path).read_bytes())
    if (approval.get('decision') != 'ACCEPTED' or approval.get('export_sha256') != sha(raw)
            or not approval.get('reviewer') or not approval.get('accepted_at')):
        raise ValueError('Explicit acceptance must identify this exact review export')
    ids = approval.get('accepted_packet_ids')
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError('Acceptance requires distinct packet IDs')
    if review.get('schema') != 'training-review-v1':
        raise ValueError('Expected a training review export')
    packets = {p['manifest']['packet_id']: p for p in read_json((run / 'review/packet-set.json').read_bytes())}
    rows, metadata = [], []
    for packet_id in ids:
        if packet_id not in packets or packet_id not in review.get('packets', {}):
            raise ValueError('Unknown accepted packet')
        packet, answer = packets[packet_id], review['packets'][packet_id]
        if answer.get('reviewed') is not True or answer.get('input_binding') != review_binding(packet):
            raise ValueError('Accepted packet must be reviewed against the exact source')
        label = from_review(answer['fields'])
        rows.append(dict(messages=[dict(role='user', content=packet['model_input']),
                                   dict(role='assistant', content=dumps(label))]))
        metadata.append(dict(packet_id=packet_id, source=packet['manifest'], group_id=packet['manifest']['accession'],
                             split='development', raw_draft_sha256=sha((run / packet_id / 'draft.json').read_bytes())))
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise FileExistsError(output)
    output.write_text(''.join(dumps(row) + '\n' for row in rows))
    output.with_suffix('.manifest.json').write_text(dumps(dict(approval=approval, examples=metadata)) + '\n')
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(export_accepted(args.run, args.review, args.approval, args.output))
