"""Compact conversion must retain source meaning and reject unsupported outputs."""

from copy import deepcopy
import hashlib
import unittest

from proxybench.schemas.records import FIELD_TYPES, D2_RULE
from proxybench.training.compact import (
    DEFAULT, compact_source, encode_record, decode_record, encode_response, decode_response, dumps,
)


def example():
    source = '<p>Élan Fund | 00123 | FOR | NONE | Élan Fund</p>'
    size = len(source.encode())
    span = dict(document_id='doc', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                start_byte=100, end_byte=100 + size)
    bundle = dict(input_version='fragment-input-v1', input_id='example', track='fragment', view_id='view',
                  document_id='doc', source_sha256=span['source_sha256'], encoding='utf-8',
                  target=dict(start_byte=100, end_byte=100 + size, sha256=span['source_sha256']),
                  blocks=[dict(block_id='b', span=span, original_text=source, prepared_text='Élan Fund | 00123 | FOR | NONE | Élan Fund')])
    evidence = dict(**span, view_id='view', block_id='b', quote='Élan Fund', quote_basis='PREPARED_VIEW')
    fields = {k: deepcopy(DEFAULT) for k in FIELD_TYPES}
    fields['issuer_name'] = dict(value='Élan Fund', raw_text='Élan Fund', availability='PRESENT',
                                 origin='EXTRACTED', evidence=[evidence], reason=None, rule_id=None)
    record = dict(record_id='administrative', source_anchor=dict(subject_spans=[span], scope_spans=[]),
                  fields=fields, enrichments=[])
    return dumps(bundle), record


class CompactTests(unittest.TestCase):
    def roundtrip(self, bundle, record):
        encoded = encode_record(record, bundle)
        restored = decode_record(dumps(encoded), bundle, record_id=record['record_id'])
        self.assertEqual(restored, record)
        return encoded

    def test_source_and_roundtrip_unicode_duplicate_text(self):
        bundle, record = example()
        source = compact_source(bundle)
        self.assertEqual(source['blocks'][0]['source'], '<p>Élan Fund | 00123 | FOR | NONE | Élan Fund</p>')
        self.assertNotIn('fields', source)
        self.assertEqual(source['target'], [0, 0, len(source['blocks'][0]['source'].encode())])
        self.roundtrip(bundle, record)

    def test_all_unresolved_states_and_explicit_none(self):
        for state in ('ABSENT_IN_CONTEXT', 'AMBIGUOUS', 'UNREADABLE', 'CONFLICTING', 'NOT_APPLICABLE'):
            bundle, record = example()
            f = record['fields']['issuer_name']
            f.update(value=None, origin=None, availability=state, reason='The supplied source does not resolve this value.')
            if state == 'UNREADABLE':
                f['evidence'][0].update(quote_basis='UNREADABLE_REGION', quote=None)
            self.roundtrip(bundle, record)
        bundle, record = example()
        record['fields']['management_recommendation'] = deepcopy(record['fields']['issuer_name'])
        f = record['fields']['management_recommendation']
        f.update(value='NONE', raw_text='NONE')
        f['evidence'][0]['quote'] = 'NONE'
        compact = self.roundtrip(bundle, record)
        self.assertEqual(compact['fields']['management_recommendation']['v'], 'NONE')
        self.assertIsNone(compact['fields']['ticker'])

    def test_nonvoting_empty_components_and_derived_origin(self):
        bundle, record = example()
        participation = deepcopy(record['fields']['issuer_name'])
        participation.update(value='DID_NOT_VOTE', raw_text=None)
        record['fields']['participation'] = participation
        components = deepcopy(participation)
        components.update(value=[], availability='NOT_APPLICABLE', origin='DERIVED', rule_id=D2_RULE, reason='Explicit nonvoting.')
        record['fields']['vote_components'] = components
        self.roundtrip(bundle, record)

    def test_nested_uncertainty_and_leading_zero_identifier(self):
        bundle, record = example()
        base = record['fields']['issuer_name']
        name = deepcopy(base)
        name.update(value=None, origin=None, availability='AMBIGUOUS', reason='Two fund interpretations.')
        kind = deepcopy(base); kind['value'] = 'INDIVIDUAL_FUND'
        members = deepcopy(name); members['availability'] = 'NOT_APPLICABLE'
        scope = deepcopy(base); scope['value'] = dict(name=name, scope_type=kind, members=members)
        record['fields']['reporting_scope'] = scope
        number = deepcopy(base); number.update(value='00123', raw_text='00123')
        number['evidence'][0]['quote'] = '00123'
        record['fields']['proposal_number'] = number
        self.roundtrip(bundle, record)

    def test_bad_citations_and_extra_fields_fail(self):
        bundle, record = example()
        original = encode_record(record, bundle)
        for mutate in (
            lambda c: c['citations'][0].update(span=[0, 0, 99999]),
            lambda c: c['citations'][0].update(span=[True, 0, 10]),
            lambda c: c['citations'][0].update(quote='invented'),
            lambda c: c['fields']['issuer_name'].update(e=[-1]),
            lambda c: c['fields']['issuer_name'].update(surprise='value'),
            lambda c: c['fields'].pop('ticker'),
        ):
            compact = deepcopy(original); mutate(compact)
            with self.assertRaises(ValueError):
                decode_record(dumps(compact), bundle, record_id='x')

    def test_split_components_keep_order_and_repeated_evidence(self):
        bundle, record = example()
        present = deepcopy(record['fields']['issuer_name'])
        direction = deepcopy(present)
        direction.update(value='FOR', raw_text='FOR')
        direction['evidence'][0]['quote'] = 'FOR'
        component = dict(direction=direction, quantity=deepcopy(DEFAULT),
                         disclosed_management_alignment=deepcopy(DEFAULT))
        components = [deepcopy(component), deepcopy(component)]
        components[1]['direction'].update(value='OTHER', raw_text='NONE')
        components[1]['direction']['evidence'][0]['quote'] = 'NONE'
        wrapper = deepcopy(present)
        wrapper.update(value=components, raw_text=None)
        record['fields']['vote_components'] = wrapper
        participation = deepcopy(present)
        participation.update(value='VOTED', raw_text=None, origin='DERIVED', rule_id=D2_RULE)
        record['fields']['participation'] = participation
        compact = self.roundtrip(bundle, record)
        self.assertEqual(len(compact['fields']['vote_components']['v']), 2)
        self.assertEqual(len(compact['citations']), 3)

    def test_valid_citation_does_not_prove_semantic_support(self):
        bundle, record = example()
        compact = encode_record(record, bundle)
        compact['fields']['issuer_name']['v'] = 'Unsupported Corporation'
        decoded = decode_record(dumps(compact), bundle, record_id='x')
        # The codec validates syntax and quotation, not semantic equivalence.
        self.assertEqual(decoded['fields']['issuer_name']['value'], 'Unsupported Corporation')

    def test_duplicate_keys_extra_records_truncation_and_abstention(self):
        bundle, record = example()
        with self.assertRaises(ValueError):
            decode_record('{"fields":{},"fields":{}}', bundle, record_id='x')
        response = dict(schema_version='benchmark-v1', input_id='example', status='COMPLETE', records=[record], failure=None)
        compact = encode_response(response, bundle)
        restored = decode_response(dumps(compact), bundle)
        self.assertEqual(restored['records'][0]['fields'], record['fields'])
        compact['records'].append(compact['records'][0])
        with self.assertRaises(ValueError):
            decode_response(dumps(compact), bundle)
        with self.assertRaises(ValueError):
            decode_response(dumps(compact)[:-8], bundle)
        failure = dict(code='NO_RECORD', message='No recoverable target.', stage='extraction', truncated=False)
        abstain = dict(status='ABSTAINED', records=[], failure=failure)
        self.assertEqual(decode_response(dumps(abstain), bundle)['failure'], failure)

    def test_source_boundary_and_unsupported_enrichment_fail(self):
        bundle, record = example()
        record['fields']['issuer_name']['evidence'][0]['start_byte'] = 99
        with self.assertRaises(ValueError):
            encode_record(record, bundle)
        bundle, record = example()
        compact = encode_record(record, bundle)
        compact['enrichments'] = [{'field_path': '/fields/issuer_name', 'field': {}}]
        with self.assertRaises(ValueError):
            decode_record(dumps(compact), bundle, record_id='x')
