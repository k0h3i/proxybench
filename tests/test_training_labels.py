"""Direct labels retain edits and require acceptance of exact review exports."""
import unittest

from proxybench.training.labels import TYPES, validate, to_review, from_review


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
