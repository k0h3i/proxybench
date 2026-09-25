"""CPU scoring and review checks for the versioned batch 35 evaluation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from proxybench.evaluation.pilot_review import accept_decisions, create_review, finish_report
from proxybench.evaluation.system_admission import (admit_timeout_recovery, check_prefixes,
                                                   ledger_prefixes, saved_baseline_prefix, validate_evaluation)
from proxybench.evaluation.system_labels import paired_report_system, score_system
from proxybench.execution.resources import durable_json
from proxybench.training.historical import admit_schedule, binding, inventory
from proxybench.training.historical_engine import engine
from proxybench.training.historical_run import launch
from proxybench.training.smoke import digest
from test_training_labels import field, label


def answer(value):
    return dict(status='COMPLETE', text=json.dumps(value), token_ids=[1], prompt_token_ids=[2])


def meta(*cells):
    return dict(packet=dict(manifest=dict(blocks=[dict(cells=[dict(text=text) for text in cells])])))


class SystemEvaluationTests(unittest.TestCase):
    def test_description_identifier_and_nonvoting_remain_primary_errors(self):
        reference = label()
        reference['fields']['raw_description'] = field('Approve the three-year frequency of the advisory vote')
        reference['fields']['series_identifiers'] = field([dict(source_label=field('Series ID'), value=field('S0000123'))])
        reference['fields']['participation'] = field('DID_NOT_VOTE')
        reference['fields']['vote_components'] = field([], state='NOT_APPLICABLE', origin='DERIVED')
        prediction = deepcopy(reference)
        prediction['fields']['raw_description']['value'] = 'Approve the three-year frequency'
        prediction['fields']['series_identifiers']['value'][0]['value']['value'] = 'S123'
        prediction['fields']['participation']['value'] = 'VOTED'
        prediction['fields']['vote_components'] = label()['fields']['vote_components']
        result = score_system(reference, answer(prediction), meta('Source text'))
        self.assertEqual(set(result['primary_errors']),
                         {'raw_description', 'series_identifiers', 'participation', 'vote_components'})
        self.assertFalse(result['source_value_correct'])

    def test_literal_quote_cannot_cross_source_cells(self):
        reference = label()
        prediction = deepcopy(reference)
        prediction['fields']['issuer_name']['raw_text'] = 'foo bar'
        result = score_system(reference, answer(prediction), meta('foo', 'bar'))
        self.assertIn('issuer_name', result['unsupported_quotes'])
        self.assertTrue(result['source_value_correct'])

    def test_other_equivalence_changes_only_reviewed_raw_wording(self):
        reference = label()
        reference['fields']['vote_components']['value'][0]['direction'] = field('OTHER')
        reference['fields']['vote_components']['value'][0]['direction']['raw_text'] = 'Three-year frequency'
        prediction = deepcopy(reference)
        prediction['fields']['vote_components']['value'][0]['direction']['raw_text'] = 'Every three years'
        source = meta('Three-year frequency; Every three years')
        raw = answer(prediction)
        path = 'fields.vote_components.value[0].direction'
        decision = dict(reference_field_path=path, prediction_field_path=path, block_index=0, cell_index=0,
                        column='Voting frequency', passage='Three-year frequency',
                        reference_sha256=binding(reference), answer_sha256=binding(raw),
                        reason='Both spans identify the same frequency')
        self.assertFalse(score_system(reference, raw, source)['source_value_correct'])
        result = score_system(reference, raw, source, other_equivalences=[decision])
        self.assertTrue(result['source_value_correct'])
        self.assertFalse(result['exact'])
        self.assertEqual(result['other_equivalence_paths'], [path])
        wrong = deepcopy(decision)
        wrong['passage'] = 'Absent passage'
        with self.assertRaises(ValueError):
            score_system(reference, raw, source, other_equivalences=[wrong])
        prediction['fields']['vote_components']['value'][0]['direction']['value'] = 'AGAINST'
        with self.assertRaises(ValueError):
            score_system(reference, answer(prediction), source,
                         other_equivalences=[dict(decision, answer_sha256=binding(answer(prediction)))])

    def test_system_report_keeps_all_attempts_and_coverage_groups(self):
        reference = label()
        good = score_system(reference, answer(reference), meta('source'))
        bad = score_system(reference, dict(status='LENGTH_STOP', text=''), meta('source'))
        report = paired_report_system([dict(id='one', family='family', era='2000s', layout='text',
                                            behavior_tags=['nonvoting'], original=bad, trained=good)])
        self.assertEqual(report['scorer'], 'historical-system-v1')
        self.assertEqual(report['aggregate']['wins'], 1)
        self.assertEqual(report['eras']['2000s']['targets'], 1)
        self.assertEqual(report['behavior_tags']['nonvoting']['targets'], 1)

    def test_review_binds_other_decision_to_source_and_exact_answer(self):
        reference = label()
        reference['fields']['vote_components']['value'][0]['direction'] = field('OTHER')
        reference['fields']['vote_components']['value'][0]['direction']['raw_text'] = 'Three-year frequency'
        prediction = deepcopy(reference)
        prediction['fields']['vote_components']['value'][0]['direction']['raw_text'] = 'Every three years'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, trained = root/'original.json', root/'trained.json'
            original.write_text(json.dumps(answer(reference)))
            trained.write_text(json.dumps(answer(prediction)))
            cells = [dict(block_index=0, cell_index=0, text='Three-year frequency; Every three years')]
            review = create_review(root/'review', [dict(id='one', source=cells[0]['text'], source_cells=cells,
                                   reference=reference, answers=dict(original=str(original), trained=str(trained)))],
                                   identity='trained-run', kind='evaluation', scorer='historical-system-v1')
            packet = json.loads((review/'packet.json').read_text())
            mapping = json.loads((review/'mapping.json').read_text())['one']
            letter = next(key for key, model in mapping.items() if model == 'trained')
            decisions = json.loads((review/'decision-template.json').read_text())
            decisions.update(reviewer='reviewer', rationale='Read the declared source cell')
            for row in decisions['cases']['one'].values():
                row['reason'] = 'Reviewed against the cell'
            path = 'fields.vote_components.value[0].direction'
            decisions['cases']['one'][letter]['other_equivalences'] = [
                dict(reference_field_path=path, prediction_field_path=path, block_index=0, cell_index=0,
                     column='Voting frequency', passage='Three-year frequency',
                     reference_sha256=packet['cases'][0]['reference_sha256'],
                     answer_sha256=packet['cases'][0]['answers'][letter]['answer_sha256'],
                     reason='Both spans identify the same frequency')]
            accept_decisions(review, decisions, 'trained-run')
            self.assertTrue((review/'accepted.json').exists())

    def test_report_reads_three_role_reference_and_versioned_scorer(self):
        reference = label()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, trained = root/'original', root/'trained'
            original.mkdir()
            trained.mkdir()
            for directory in (original, trained):
                (directory/'answer-0.json').write_text(json.dumps(answer(reference)))
            manifest = dict(packet_id='one', family='family', era='2000s', layout='text',
                            behavior_tags=['identifier'], blocks=[dict(cells=[dict(text='Source cell')])])
            metadata = dict(split='development', packet=dict(manifest=manifest))
            prepared = dict(examples=[metadata], rows=dict(development=[dict(messages=[
                dict(role='system', content='rules'), dict(role='user', content='source'),
                dict(role='assistant', content=json.dumps(reference))])]))
            state = dict(identity='trained-run', completed_phases={
                'baseline': dict(output=str(original)), 'final': dict(output=str(trained))})
            (root/'evaluation-admission.json').write_text('{}')
            review = create_review(root/'review-final', [dict(id='one', source='Source cell',
                                   source_cells=[dict(block_index=0, cell_index=0, text='Source cell')],
                                   reference=reference, answers=dict(original=str(original/'answer-0.json'),
                                                                     trained=str(trained/'answer-0.json')))],
                                   identity='trained-run', kind='evaluation', scorer='historical-system-v1')
            decisions = json.loads((review/'decision-template.json').read_text())
            decisions.update(reviewer='reviewer', rationale='Read the source cell')
            for row in decisions['cases']['one'].values():
                row['reason'] = 'Reviewed against the source'
            accept_decisions(review, decisions, 'trained-run')
            self.assertEqual(finish_report(root, state, prepared), 'COMPLETE')
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['scorer'], 'historical-system-v1')
            self.assertEqual(report['aggregate']['targets'], 1)

    def test_ledger_admission_preserves_prefix_when_evaluation_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'gpu-ledger.jsonl'
            path.write_text(json.dumps(dict(status='EXITED', phase='training', elapsed_seconds=100))+'\n')
            prefixes = ledger_prefixes(root)
            path.write_text(path.read_text()+json.dumps(dict(status='EXITED', phase='baseline',
                                                              elapsed_seconds=30))+'\n')
            check_prefixes(root, prefixes)
            path.write_text(json.dumps(dict(status='EXITED', phase='training', elapsed_seconds=99))+'\n')
            with self.assertRaisesRegex(ValueError, 'prior resource ledger'):
                check_prefixes(root, prefixes)

    def test_development_pause_keeps_complete_answers_and_remaining_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = dict(status='TRAINED', operation='separate', identity='trained-run', attempts={},
                         completed_phases={name: {} for name in ('training', 'original-export',
                                                                 'original-conversion', 'trained-export',
                                                                 'trained-conversion')},
                         reservations=dict(training=6000, **{'original-export': 300,
                             'original-conversion': 600, 'trained-export': 300,
                             'trained-conversion': 600, 'baseline': 1650, 'final': 1650}),
                         resource_limits=dict(gpu_total=14400, cpu_total=3600, gpu_phase=1800,
                                              training_phase=6000, cpu_phase=1800,
                                              stop_reserve=1200, test_reserve=3300))
            (root/'prepared.json').write_text('{}')
            (root/'gpu-ledger.jsonl').write_text(json.dumps(dict(status='EXITED', phase='training',
                                                                 elapsed_seconds=100))+'\n')
            config = dict(profile='historical-system-v1', development_examples=90, updates=660,
                          minimum_disk_bytes=0, limits=dict(phase_seconds=6000), cpu_limits={})
            def stop(command, output, limits, **kwargs):
                output.mkdir()
                (output/'stop.json').write_text('{}')
                (output.parent/'answer-0.json').write_text(json.dumps(answer(label())))
                return 'USER_STOP'
            with patch('proxybench.training.historical_run.validate_evaluation',
                       return_value=({}, state, {})), \
                 patch('proxybench.training.historical_run.environment', return_value={}), \
                 patch('proxybench.training.historical_run.supervise', side_effect=stop):
                self.assertEqual(launch(root, config, Path('config.json'), 'evaluate'), 130)
            saved = json.loads((root/'state.json').read_text())
            self.assertEqual(saved['status'], 'EVALUATION_PAUSED')
            self.assertIn(str(root/'phases/baseline-01/answer-0.json'),
                          saved['partial_phases']['baseline'][0]['files'])
            admitted = admit_schedule(saved, dict(cpu=[], gpu=[dict(phase='training', elapsed_seconds=100),
                                                               dict(phase='baseline', elapsed_seconds=40)]),
                                      phase='baseline', operation=['baseline', 'final'])
            self.assertEqual(admitted['remaining']['baseline'], 1610)

    def test_recovered_development_keeps_spent_time_without_a_ceiling(self):
        state = dict(completed_phases={'training': {}, 'original-export': {},
                                       'original-conversion': {}, 'trained-export': {},
                                       'trained-conversion': {}},
                     reservations={'baseline': None, 'final': None},
                     resource_limits=dict(evaluation_unbounded=True, gpu_total=None, gpu_phase=None))
        spent = dict(cpu=[], gpu=[dict(phase='training', elapsed_seconds=1463),
                                  dict(phase='baseline', elapsed_seconds=1800.1)])
        admitted = admit_schedule(state, spent, phase='baseline', operation=['baseline', 'final'])
        self.assertAlmostEqual(admitted['used_seconds'], 3263.1)
        self.assertAlmostEqual(admitted['phase_used_seconds'], 1800.1)
        self.assertEqual(admitted['remaining'], {'baseline': None, 'final': None})
        self.assertIsNone(admitted['time_ceiling_seconds'])

    def test_timeout_recovery_requires_matching_saved_prompt_prefix(self):
        prepared = dict(items=dict(development=[dict(input_ids=[1, 9], input_tokens=1),
                                                dict(input_ids=[2, 9], input_tokens=1),
                                                dict(input_ids=[3, 9], input_tokens=1)]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = dict(index=0, status='COMPLETE', max_new_tokens=1792, prompt_token_ids=[1])
            second = dict(index=1, status='STARTED', max_new_tokens=1792, prompt_token_ids=[2])
            (root/'answer-0.json').write_text(json.dumps(first))
            (root/'answer-1.json').write_text(json.dumps(second))
            self.assertEqual(saved_baseline_prefix(root, prepared, 3), 1)
            (root/'answer-0.json').write_text(json.dumps(dict(first, prompt_token_ids=[4])))
            with self.assertRaisesRegex(ValueError, 'request identity'):
                saved_baseline_prefix(root, prepared, 3)

    def test_timeout_recovery_retries_after_admission_write_without_resetting_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/'phases/baseline-01'
            (output/'supervisor').mkdir(parents=True)
            state = dict(status='FAILED', operation='separate', identity='run',
                         failed_phase='baseline', supervisor_status='PHASE_TIMEOUT',
                         active_phase='baseline', attempts={'baseline': 1},
                         completed_phases={'training': dict(output='training', files={})},
                         reservations={'baseline': 1650, 'final': 1650},
                         resource_limits={'gpu_total': 14400, 'gpu_phase': 1800,
                                          'stop_reserve': 1200, 'test_reserve': 3300})
            durable_json(root/'state.json', state)
            prepared = dict(items=dict(development=[dict(input_ids=[1, 9], input_tokens=1),
                                                    dict(input_ids=[2, 9], input_tokens=1),
                                                    dict(input_ids=[3, 9], input_tokens=1)]))
            durable_json(root/'prepared.json', prepared)
            for index, status in ((0, 'COMPLETE'), (1, 'STARTED')):
                durable_json(output/f'answer-{index}.json',
                             dict(index=index, status=status, max_new_tokens=1792,
                                  prompt_token_ids=[index+1]))
            result = dict(status='PHASE_TIMEOUT', execution_id='failed-attempt',
                          elapsed_seconds=1800.1, surviving_owned_pids=[])
            durable_json(output/'supervisor/result.json', result)
            gpu_ledger = root/'gpu-ledger.jsonl'
            before = json.dumps(dict(status='PHASE_TIMEOUT', phase='baseline',
                                     execution_id='failed-attempt', elapsed_seconds=1800.1))+'\n'
            gpu_ledger.write_text(before)
            original = dict(schema='historical-system-development-v1', scorer='historical-system-v1',
                            training_identity='run', training_commit='old',
                            training_phase=state['completed_phases']['training'],
                            evaluation_commit='old-eval', code={}, ledger_prefixes=ledger_prefixes(root))
            durable_json(root/'evaluation-admission.json', original)
            identity = dict(git_commit='old')
            code = inventory(Path('src/proxybench').rglob('*.py'))
            def read_training(_root, _config):
                return identity, json.loads((root/'state.json').read_text())
            def interrupted_write(path, value):
                if Path(path).name == 'state.json':
                    raise OSError('interrupted before state update')
                durable_json(path, value)
            with patch('proxybench.evaluation.system_admission.preserved_training', side_effect=read_training), \
                 patch('proxybench.evaluation.system_admission.current_evaluation_code',
                       return_value=(code, 'new-eval')):
                with patch('proxybench.evaluation.system_admission.durable_json', side_effect=interrupted_write):
                    with self.assertRaisesRegex(OSError, 'interrupted before state update'):
                        admit_timeout_recovery(root, dict(development_examples=3))
                self.assertEqual(json.loads((root/'state.json').read_text())['status'], 'FAILED')
                recovered = admit_timeout_recovery(root, dict(development_examples=3))
                self.assertEqual(recovered['complete_answers'], 1)
                self.assertEqual(admit_timeout_recovery(root, dict(development_examples=3)), recovered)
                validate_evaluation(root, dict(development_examples=3))
            self.assertEqual(gpu_ledger.read_text(), before)
            self.assertEqual(json.loads((root/'state.json').read_text())['status'], 'EVALUATION_PAUSED')

    def test_recovered_engine_generates_only_missing_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new, converted = root/'old', root/'new', root/'converted'
            for directory in (old, new, converted):
                directory.mkdir()
            model = converted/'model.gguf'
            model.write_bytes(b'converted')
            durable_json(converted/'complete.json', dict(status='COMPLETE', path=str(model), sha256=digest(model)))
            items = [dict(input_ids=[index+1, 9], input_tokens=1) for index in range(3)]
            rows = [dict(messages=[dict(role='user', content='source'),
                                   dict(role='assistant', content='answer')]) for _ in items]
            durable_json(root/'prepared.json', dict(items=dict(development=items), rows=dict(development=rows)))
            saved = dict(index=0, status='COMPLETE', prompt_token_ids=[1], max_new_tokens=1792,
                         token_ids=[9], format_valid=False, timing=dict(request_seconds=2))
            durable_json(old/'answer-0.json', saved)
            (old/'answer-0.tokens.jsonl').write_text('{"token_id":9}\n')
            durable_json(old/'answer-1.json', dict(index=1, status='STARTED', prompt_token_ids=[2],
                                                   max_new_tokens=1792))
            old_files = inventory(old.iterdir())
            durable_json(root/'state.json', dict(resource_limits=dict(evaluation_unbounded=True),
                                                completed_phases={'original-conversion': dict(output=str(converted))},
                                                partial_phases={'baseline': [dict(output=str(old), files=old_files)]}))
            pin = root/'pin.json'
            runtime = root/'engine.json'
            durable_json(pin, dict(files={}, library_path=''))
            durable_json(runtime, dict(server_arguments=[], startup_seconds=120))
            class Tokenizer:
                def apply_chat_template(self, *args, **kwargs):
                    return 'rendered'
            class Process:
                returncode = None
                pid = 12345
                def __init__(self, command, **kwargs):
                    kwargs['stdout'].write(b'offloaded 1/1 layers to GPU\n')
                    kwargs['stdout'].flush()
                def poll(self):
                    return None
                def terminate(self):
                    self.returncode = 0
                def wait(self, timeout=None):
                    return 0
                def kill(self):
                    self.returncode = -9
            class Socket:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
                def bind(self, address):
                    pass
                def getsockname(self):
                    return ('127.0.0.1', 12345)
            generated = []
            def generate_answer(port, tokenizer, prompt, path, *, index, **kwargs):
                generated.append(index)
                self.assertIsNone(kwargs['deadline'])
                result = dict(index=index, status='COMPLETE', prompt_token_ids=prompt,
                              max_new_tokens=1792, token_ids=[9], format_valid=True,
                              timing=dict(request_seconds=2))
                durable_json(path, result)
                return result
            fake_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()))
            config = dict(runtime_pin=str(pin), engine_configuration=str(runtime),
                          engine_root=str(root), development_examples=3)
            with patch.dict(sys.modules, {'transformers': fake_transformers}), \
                 patch('proxybench.training.historical_engine.socket.socket', return_value=Socket()), \
                 patch('proxybench.training.historical_engine.subprocess.Popen', Process), \
                 patch('proxybench.training.historical_engine.json_request', return_value={'status': 'ok'}), \
                 patch('proxybench.training.historical_engine.require_prompt_tokens') as prompt_check, \
                 patch('proxybench.training.historical_engine.generate', side_effect=generate_answer):
                engine(root, 'baseline', new, config)
            self.assertEqual(generated, [1, 2])
            self.assertEqual(prompt_check.call_count, 3)
            self.assertTrue(all(call.kwargs['timeout'] is None for call in prompt_check.call_args_list))
            self.assertEqual((new/'answer-0.json').read_bytes(), (old/'answer-0.json').read_bytes())
            self.assertEqual((new/'answer-0.tokens.jsonl').read_bytes(),
                             (old/'answer-0.tokens.jsonl').read_bytes())
            self.assertEqual(json.loads((old/'answer-1.json').read_text())['status'], 'STARTED')
            self.assertEqual(json.loads((new/'complete.json').read_text())['status'], 'COMPLETE')


if __name__ == '__main__':
    unittest.main()
