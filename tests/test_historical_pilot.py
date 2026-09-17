"""Frozen historical recipe, source-value scoring, review gates, and admission."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.evaluation.pilot_review import accept_decisions, create_review, require_decisions, source_text
from proxybench.evaluation.training_labels import export_comparison, paired_report, score
from proxybench.training.historical import PHASES, admit_schedule, check_files, panel_indices, prepare_sequences, read_configuration
from proxybench.training.historical_engine import normalize_engine_length
from proxybench.training.historical_run import main, validate_transition
from proxybench.training.trajectory import sample_order
from test_training_labels import field, label


def answer(value, **kwargs):
    return dict(text=json.dumps(value), status='COMPLETE', prompt_token_ids=[1, 2], token_ids=[3, 99], **kwargs)


class PilotScoringTests(unittest.TestCase):
    def test_format_extra_records_missing_field_and_truncation(self):
        ref = label()
        for value in ([ref, ref], {'fields': {}}, {**ref, 'extra_record': ref}):
            self.assertFalse(score(ref, answer(value), '')['format_valid'])
        a = answer(ref)
        a['status'] = 'LENGTH_STOP'
        self.assertFalse(score(ref, a, '')['source_value_correct'])

    def test_origin_cannot_hide_wrong_vote_or_wrong_absence(self):
        ref, prediction = label(), label()
        vote = prediction['fields']['vote_components']['value'][0]['direction']
        vote.update(value='AGAINST', origin='DERIVED')
        result = score(ref, answer(prediction), '')
        self.assertTrue(result['format_valid'])
        self.assertFalse(result['source_value_correct'])
        prediction = label()
        prediction['fields']['issuer_name']['origin'] = 'DERIVED'
        result = score(ref, answer(prediction), '')
        self.assertTrue(result['source_value_correct'])
        self.assertEqual(result['origin_errors'], ['issuer_name'])
        prediction['fields']['management_recommendation'] = field('NONE')
        self.assertFalse(score(ref, answer(prediction), '')['source_value_correct'])

    def test_whitespace_case_identifiers_and_raw_description(self):
        ref = label()
        ref['fields']['proposal_number'] = field('001.')
        ref['fields']['raw_description'] = field('Elect Director Jane Doe')
        prediction = deepcopy(ref)
        prediction['fields']['issuer_name']['value'] = 'a  &\n b'
        prediction['fields']['reporting_scope']['value']['name']['value'] = '  FUND '
        self.assertTrue(score(ref, answer(prediction), '')['source_value_correct'])
        prediction['fields']['proposal_number']['value'] = '1'
        self.assertFalse(score(ref, answer(prediction), '')['source_value_correct'])
        prediction = deepcopy(ref)
        prediction['fields']['raw_description']['value'] = 'Jane Doe'
        self.assertFalse(score(ref, answer(prediction), '')['source_value_correct'])

    def test_subject_equivalence_requires_source_review(self):
        ref = label()
        ref['fields']['separate_subject'] = field('Elect Director Jane Doe')
        prediction = deepcopy(ref)
        prediction['fields']['separate_subject']['value'] = 'Jane Doe'
        self.assertFalse(score(ref, answer(prediction), '')['source_value_correct'])
        self.assertTrue(score(ref, answer(prediction), '', subject_equivalent=True)['source_value_correct'])
        self.assertEqual(export_comparison(answer(ref), answer(prediction))['status'], 'REVIEW_REQUIRED')

    def test_multisets_keep_duplicate_components_and_identifiers(self):
        ref = label()
        first = ref['fields']['vote_components']['value'][0]
        second = deepcopy(first)
        second['direction'] = field('WITHHOLD')
        ref['fields']['vote_components']['value'].append(second)
        prediction = deepcopy(ref)
        prediction['fields']['vote_components']['value'].reverse()
        self.assertTrue(score(ref, answer(prediction), '')['source_value_correct'])
        prediction['fields']['vote_components']['value'].append(deepcopy(first))
        self.assertFalse(score(ref, answer(prediction), '')['source_value_correct'])
        ref['fields']['security_identifiers'] = field([dict(source_label=field('CUSIP'), value=field('001234'))])
        prediction = deepcopy(ref)
        prediction['fields']['security_identifiers']['value'] *= 2
        self.assertFalse(score(ref, answer(prediction), '')['source_value_correct'])

    def test_derived_fields_nonvote_and_other_meaning(self):
        ref = label()
        prediction = deepcopy(ref)
        prediction['fields']['participation'] = field('OTHER')
        prediction['fields']['participation']['raw_text'] = 'source wording'
        prediction['fields']['reporting_scope']['value']['scope_type']['value'] = 'INDIVIDUAL_FUND'
        result = score(ref, answer(prediction), '')
        self.assertTrue(result['source_value_correct'])
        self.assertEqual(len(result['derivation_errors']), 2)
        ref['fields']['participation'] = field('DID_NOT_VOTE')
        ref['fields']['vote_components'] = field([], state='NOT_APPLICABLE', origin='DERIVED')
        self.assertFalse(score(ref, answer(label()), '')['source_value_correct'])
        ref = label()
        ref['fields']['vote_components']['value'][0]['direction'] = field('OTHER')
        ref['fields']['vote_components']['value'][0]['direction']['raw_text'] = 'No action'
        prediction = deepcopy(ref)
        prediction['fields']['vote_components']['value'][0]['direction']['raw_text'] = 'Declined'
        self.assertFalse(score(ref, answer(prediction), '')['source_value_correct'])

    def test_quotation_support_is_separate_from_semantics(self):
        ref = label()
        prediction = deepcopy(ref)
        prediction['fields']['issuer_name']['raw_text'] = 'Invented quotation'
        result = score(ref, answer(prediction), 'A & B FOR Fund First WITH_MANAGEMENT')
        self.assertTrue(result['source_value_correct'])
        self.assertIn('issuer_name', result['unsupported_quotes'])
        self.assertEqual(result['quotation_semantics'], 'REVIEW_REQUIRED')
        self.assertEqual(export_comparison(answer(ref), answer(prediction))['status'], 'REVIEW_REQUIRED')

    def test_export_malformed_length_and_changed_origin(self):
        a = dict(text='not JSON', status='COMPLETE', prompt_token_ids=[1], token_ids=[3])
        self.assertEqual(export_comparison(a, a)['status'], 'PASS')
        b = dict(a, text='different')
        self.assertEqual(export_comparison(a, b)['status'], 'FAILED')
        a['status'] = b['status'] = 'LENGTH_STOP'
        self.assertEqual(export_comparison(a, b)['status'], 'PASS')
        b['token_ids'] = [4]
        self.assertEqual(export_comparison(a, b)['status'], 'FAILED')
        ref, prediction = label(), label()
        prediction['fields']['issuer_name']['origin'] = 'DERIVED'
        self.assertEqual(export_comparison(answer(ref), answer(prediction))['status'], 'FAILED')

    def test_all_fourteen_field_examples_and_paired_counts(self):
        ref = label()
        good = score(ref, answer(ref), '')
        bad = score(ref, dict(text='', status='LENGTH_STOP'), '')
        self.assertEqual(len(good['field_correct']), 14)
        report = paired_report([dict(id='one', family='a', original=bad, trained=good),
                                dict(id='two', family='b', original=good, trained=bad),
                                dict(id='three', family='b', original=bad, trained=bad)])
        self.assertEqual([report['aggregate'][k] for k in ('targets', 'wins', 'losses', 'ties')], [3, 1, 1, 1])


class PilotAdmissionTests(unittest.TestCase):
    def test_exact_192_order_and_distinct_training_panel(self):
        order = sample_order()
        self.assertEqual(len(order), 192)
        self.assertEqual(order, sample_order())
        self.assertEqual(sorted(order[:96]), list(range(96)))
        self.assertEqual(sorted(order[96:]), list(range(96)))
        self.assertEqual(panel_indices([dict(input_tokens=10, response_tokens=10),
                                        dict(input_tokens=1, response_tokens=9), dict(input_tokens=2, response_tokens=1)]), [0, 1, 2])

    def test_answer_mask_keeps_eos_and_removes_only_template_suffix(self):
        class Tokenizer:
            eos_token_id = 99
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2] if len(messages) == 1 else [1, 2, 3, 99, 4]
            def decode(self, tokens, **kwargs):
                return 'answer' if tokens == [3] else 'wrong'
        config = dict(context_tokens=8, input_tokens=4, response_tokens=4)
        rows = {k: [dict(messages=[dict(content=k), dict(content='answer')])] for k in ('training', 'development')}
        items = prepare_sequences(Tokenizer(), rows, config)
        self.assertEqual(items['training'][0]['labels'], [-100, -100, 3, 99])
        self.assertEqual(items['development'][0]['response_tokens'], 2)
        with self.assertRaises(ValueError):
            prepare_sequences(Tokenizer(), rows, dict(config, response_tokens=1))

    def test_full_remaining_schedule_and_phase_consumption(self):
        state = dict(completed_phases={}, reservations={n: s for n, _, s in PHASES})
        empty = dict(cpu=[], gpu=[])
        result = admit_schedule(state, empty, phase='probe')
        self.assertEqual(sum(result['remaining'].values())+result['stop_reserve_seconds'], 7170)
        with self.assertRaises(ValueError):
            admit_schedule(state, dict(cpu=[], gpu=[dict(phase='probe', elapsed_seconds=31)]), phase='probe')
        state['completed_phases'] = {n: {} for n, _, _ in PHASES if n != 'final'}
        with self.assertRaises(ValueError):
            admit_schedule(state, dict(cpu=[], gpu=[dict(phase='final', elapsed_seconds=451)]), phase='final')

    def test_recipe_and_file_hash_refusals(self):
        config = read_configuration('configs/qwen35-4b-historical-pilot.json')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'configuration.json'
            for key, value in [('updates', 96), ('accumulation', 4), ('model_revision', 'different')]:
                path.write_text(json.dumps(dict(config, **{key: value})))
                with self.assertRaises(ValueError):
                    read_configuration(path)
            with self.assertRaises(ValueError):
                check_files({str(path): 'wrong'})

    def test_overwrite_and_state_transition_refusals(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileExistsError):
            main(['prepare', '--run', tmp])
        for action in ('run', 'continue', 'resume', 'repair'):
            with self.assertRaises(ValueError):
                validate_transition(action, dict(status='COMPLETE'))
        with self.assertRaises(ValueError):
            validate_transition('repair', dict(status='FAILED', failed_phase='training'))

    def test_review_bound_to_exact_evidence_before_reveal(self):
        meta = dict(packet=dict(manifest=dict(blocks=[dict(target=True, cells=[dict(text='Selected nominee')])])) )
        self.assertIn('BEGIN MARKED TARGET\nSelected nominee\nEND MARKED TARGET', source_text(meta))
        self.assertEqual(source_text(meta, marked=False), 'Selected nominee')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root/'a.json', root/'b.json'
            a.write_text(json.dumps(answer(label())))
            b.write_text(a.read_text())
            review = create_review(root/'review', [dict(id='one', source='<script>source</script>',
                                   answers=dict(original=str(a), trained=str(b)))], identity='run', kind='export')
            self.assertIn('&lt;script&gt;', (review/'index.html').read_text())
            self.assertNotIn('original', (review/'index.html').read_text())
            decisions = json.loads((review/'decision-template.json').read_text())
            decisions.update(reviewer='assistant', rationale='Reviewed the source')
            with self.assertRaises(ValueError):
                accept_decisions(review, decisions, 'run')
            self.assertFalse((review/'revealed-mapping.json').exists())
            decisions['cases']['one'] = dict(decision='PASS', reason='Same disclosed subject')
            accept_decisions(review, decisions, 'run')
            require_decisions(review, 'run')
            self.assertTrue((review/'revealed-mapping.json').exists())
            a.write_text('changed')
            with self.assertRaises(ValueError):
                require_decisions(review, 'run')

    def test_engine_length_requires_full_prompt_and_no_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'answer.json'
            a = dict(status='TERMINATION_MISMATCH', prompt_token_ids=[1], token_ids=[2, 3], max_new_tokens=2,
                     terminal=dict(stop_type='limit', stop=True, truncated=False, tokens_predicted=2, tokens_evaluated=1,
                                   timings=dict(cache_n=0, prompt_n=1)))
            self.assertEqual(normalize_engine_length(a, path)['status'], 'LENGTH_STOP')
            a['status'] = 'TERMINATION_MISMATCH'
            a['terminal']['timings']['cache_n'] = 1
            self.assertEqual(normalize_engine_length(a, path)['status'], 'TERMINATION_MISMATCH')
