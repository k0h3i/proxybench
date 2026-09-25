"""Adapt direct agent labels to editable reviews and export accepted examples."""
from datetime import date
import hashlib
import json

FIELDS = (
    'reporting_scope', 'series_identifiers', 'issuer_name', 'security_identifiers',
    'ticker', 'meeting_date', 'meeting_type', 'proposal_number', 'raw_description',
    'separate_subject', 'proposal_source', 'participation', 'vote_components',
    'management_recommendation',
)

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
