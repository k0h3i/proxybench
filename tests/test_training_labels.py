"""Direct labels retain edits and require acceptance of exact review exports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from proxybench.annotation.bindings import review_binding
from proxybench.annotation.review import training_template
from proxybench.training.labels import (TYPES, validate, to_review, from_review, source_text,
                                        export_accepted, dumps, sha)
from test_local_inputs import PRIMARY, ROW, votes


def field(value=None, *, state=None, origin=None):
    return dict(value=value, raw_text=value if isinstance(value, str) else None,
                availability=state or ('ABSENT_IN_CONTEXT' if value is None else 'PRESENT'),
                origin=origin or ('EXTRACTED' if value is not None else None))


def label():
    fields = {key: field() for key in TYPES}
    fields['issuer_name'] = field('A & B')
    fields['ticker'] = field('')
    fields['reporting_scope'] = field(dict(name=field('Fund'), scope_type=field('FUND_GROUP'),
                                         members=field([field('First'), field(state='AMBIGUOUS')])))
    fields['vote_components'] = field([dict(direction=field('FOR'), quantity=field(state='UNREADABLE'),
                                          disclosed_management_alignment=field('WITH_MANAGEMENT'))])
    fields['participation'] = field('VOTED', origin='DERIVED')
    return {'fields': fields}


class LabelTests(unittest.TestCase):
    def test_roundtrip_edits_nested_uncertainty_empty_and_null(self):
        original = label()
        review = to_review(original)
        self.assertEqual(from_review(review), original)
        self.assertEqual(review['ticker']['value'], '""')
        self.assertEqual(review['proposal_number']['value'], 'null')
        review['issuer_name']['value'] = '"Corrected issuer"'
        corrected = from_review(review)
        self.assertEqual(corrected['fields']['issuer_name']['value'], 'Corrected issuer')
        self.assertEqual(corrected['fields']['issuer_name']['raw_text'], 'A & B')
        for invalid in ('', 'NaN', '{"value":1,"value":2}'):
            review['issuer_name']['value'] = invalid
            with self.assertRaises(ValueError):
                from_review(review)

    def test_invalid_states_and_nonvoting_exception(self):
        value = label()
        value['fields']['proposal_number']['origin'] = 'EXTRACTED'
        with self.assertRaises(ValueError):
            validate(value)
        value = label()
        value['fields']['participation'] = field('DID_NOT_VOTE')
        value['fields']['vote_components'] = field([], state='NOT_APPLICABLE', origin='DERIVED')
        self.assertEqual(validate(value), value)
        value['fields']['participation'] = field('VOTED')
        with self.assertRaises(ValueError):
            validate(value)

    def test_source_selects_target_and_keeps_primary_context(self):
        source = source_text(PRIMARY, votes(ROW + ROW.replace('Élan', 'Neighbor')), index=0, input_id='one')
        self.assertIn(PRIMARY.decode(), source)
        self.assertIn('BEGIN MARKED TARGET\n' + ROW, source)
        self.assertNotIn('Neighbor', source)
        page = training_template('Contract <script>data</script>', 4)
        self.assertIn('4 source packets', page)
        self.assertNotIn("'INFERRED'", page)
        self.assertNotIn('Use block numbers', page)
        self.assertIn('&lt;script&gt;data&lt;/script&gt;', page)
        self.assertIn('JSON.parse(v.value)', page)
        self.assertIn('trainingControl(f,prop)', page)
        self.assertIn('trainingDisplay(draft[prop],prop)', page)
        self.assertNotIn('Values and original wording use JSON.', page)

    def test_only_reviewed_explicitly_accepted_export_becomes_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'review').mkdir()
            (root/'one').mkdir()
            (root/'one/draft.json').write_text(dumps(label()))
            packet = dict(manifest=dict(packet_id='one', accession='filing', split='development'),
                          source_view='<pre>source</pre>', model_input='contract + exact source')
            (root/'review/packet-set.json').write_text(dumps([packet]))
            answer = dict(reviewed=True, input_binding=review_binding(packet), fields=to_review(label()))
            review = dict(schema='training-review-v1', packets={'one': answer})
            path, receipt, output = root/'review.json', root/'approval.json', root/'labels.jsonl'
            def save():
                path.write_text(dumps(review))
                approval = dict(decision='ACCEPTED', export_sha256=sha(path.read_bytes()),
                                accepted_packet_ids=['one'], reviewer='user', accepted_at='2026-09-08')
                receipt.write_text(dumps(approval))
            save()
            answer['reviewed'] = False
            save()
            with self.assertRaises(ValueError):
                export_accepted(root, path, receipt, output)
            self.assertFalse(output.exists())
            answer['reviewed'] = True
            save()
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):
                export_accepted(root, path, receipt, output)
            save()
            approval = json.loads(receipt.read_text())
            approval['decision'] = 'REJECTED'
            receipt.write_text(dumps(approval))
            with self.assertRaises(ValueError):
                export_accepted(root, path, receipt, output)
            save()
            self.assertEqual(export_accepted(root, path, receipt, output), 1)
            messages = json.loads(output.read_text())['messages']
            self.assertEqual(messages[0]['content'], packet['model_input'])
            self.assertEqual(json.loads(messages[1]['content']), label())
            self.assertNotIn('reviewer', messages[1]['content'])
            with self.assertRaises(FileExistsError):
                export_accepted(root, path, receipt, output)
