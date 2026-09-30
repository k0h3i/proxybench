"""Selected-model evaluation, review bindings, and crash recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.evaluation.workflow import create_run, load_answers, report, answer_binding
from proxybench.runstate import Run, atomic_json, binding
from proxybench.training.labels import TYPES

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





class EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'run'
        self.reference = label()
        self.inputs = dict(identity=dict(dataset='d', model='m', prompt='p', runtime='r'),
                           cases=[dict(id='one', reference=self.reference, source_cells=[dict(block_index=0, cell_index=0, text='<script>bad()</script>')], messages=[])])
        self.config = dict(limits=dict(phase_seconds=5, total_seconds=15))
        create_run(self.path, self.inputs, self.config)

    def answer(self):
        return dict(status='COMPLETE', text=json.dumps(self.reference), token_ids=[1])

    def save_answer(self, run, answer):
        case = self.inputs['cases'][0]
        atomic_json(self.path / 'evaluation/results' / f'{binding(case["id"])}.json',
                    dict(binding=answer_binding(self.inputs, case, answer), answer=answer,
                         elapsed_seconds=0, session='session-' + '0' * 32))
        return load_answers(run, self.inputs)

    def test_failures_count_and_missing_is_incomplete(self):
        with Run(self.path) as run:
            self.assertEqual(report(run, self.inputs, {})['status'], 'INCOMPLETE')
            answers = self.save_answer(run, dict(status='TIMEOUT', text='', error='timeout'))
            result = report(run, self.inputs, answers)
            self.assertTrue(result['valid_accuracy'])
            self.assertEqual(result['targets'], 1)
            self.assertEqual(result['aggregate']['source_value_correct'], 0)

    def test_invalid_reference_prevents_accuracy(self):
        broken = deepcopy(self.inputs)
        broken['cases'][0]['reference'] = {}
        with Run(self.path) as run:
            result = report(run, broken, {})
            self.assertEqual(result['status'], 'INVALID_REFERENCES')
            self.assertFalse(result['valid_accuracy'])

    def test_one_writer_overwrite_and_stale_answer(self):
        with self.assertRaises(FileExistsError):
            create_run(self.path, self.inputs, self.config)
        with Run(self.path) as run:
            with self.assertRaises(BlockingIOError):
                with Run(self.path):
                    pass
            self.save_answer(run, self.answer())
            changed = deepcopy(self.inputs)
            changed['identity']['model'] = 'changed'
            with self.assertRaises(ValueError):
                load_answers(run, changed)

    def test_source_values_and_cell_boundaries(self):
        from proxybench.evaluation.answers import score_system
        prediction = deepcopy(self.reference)
        prediction['fields']['issuer_name']['raw_text'] = 'foo bar'
        prediction['fields']['ticker']['value'] = 'DIFFERENT'
        answer = dict(status='COMPLETE', text=json.dumps(prediction))
        result = score_system(self.reference, answer, dict(source_cells=[dict(text='foo'), dict(text='bar')]))
        self.assertIn('issuer_name', result['unsupported_quotes'])
        self.assertFalse(result['field_correct']['ticker'])
        self.assertFalse(result['source_value_correct'])

    def training_export_state(self, *, used):
        from proxybench.execution.resources import append_entry
        with Run(self.path) as run:
            run.state.update(operation='train', configuration=dict(export_seconds=150, limits=dict(total_seconds=600)),
                             dataset='synthetic-dataset', resource_limit_seconds=600, consumed_seconds=550,
                             pending_charge=dict(target='model-export', seconds=300))
            run.save()
        (self.path/'exports').mkdir()
        append_entry(self.path/'exports/resources.jsonl', dict(execution_id='old-export', status='EXITED', elapsed_seconds=used))

    def test_training_transition_recovers_completed_export_without_reservation(self):
        from argparse import Namespace
        from proxybench.__main__ import evaluation_command
        self.training_export_state(used=300)
        config = self.path/'inference.json'
        atomic_json(config, dict(limits=dict(phase_seconds=50,total_seconds=600)))
        model = self.path/'exports/model.gguf'
        model.write_bytes(b'synthetic')
        args = Namespace(command='evaluate', run_dir=str(self.path), config=str(config),
                         project_root=None, report_only=False, model=None)
        with patch('proxybench.extraction.runtime.runtime_identity', return_value='synthetic'), \
             patch('proxybench.training.runtime.export_complete', return_value=model, create=True), \
             patch('proxybench.training.runtime.export_model', side_effect=AssertionError('Repeated completed export')), \
             patch('proxybench.evaluation.workflow.prepare_inputs', return_value=self.inputs), \
             patch('proxybench.evaluation.workflow.load_answers', return_value={'one':dict(status='TIMEOUT',text='')}):
            result = evaluation_command(args)
        self.assertEqual(result['status'], 'COMPLETE')
        with Run(self.path) as run:
            self.assertEqual(run.state['operation'], 'evaluate')
            self.assertEqual(run.state['consumed_seconds'], 550)
            self.assertEqual(run.state['evaluation_resource_base_seconds'], 550)
            self.assertIsNone(run.state['pending_charge'])
            self.assertEqual(run.state['resource_limit_seconds'], 600)

    def test_partial_export_retry_receives_only_remaining_budget(self):
        from proxybench.__main__ import evaluation_export
        from proxybench.execution.resources import append_entry
        self.training_export_state(used=150)
        def finish(adapter, root, config, *, remaining_seconds):
            self.assertEqual(remaining_seconds, 200)
            self.assertEqual(config['limits']['total_seconds'], 600)
            append_entry(root/'resources.jsonl', dict(execution_id='retry', status='EXITED', elapsed_seconds=60))
            return root/'model.gguf'
        with Run(self.path) as run, \
             patch('proxybench.training.runtime.export_complete', return_value=None, create=True), \
             patch('proxybench.training.runtime.export_model', side_effect=finish):
            evaluation_export(run)
            self.assertEqual(run.state['consumed_seconds'], 460)
            self.assertIsNone(run.state['pending_charge'])
        with Run(self.path) as run, \
             patch('proxybench.training.runtime.export_complete', return_value=self.path/'exports/model.gguf', create=True), \
             patch('proxybench.training.runtime.export_model', side_effect=AssertionError('Repeated export')):
            evaluation_export(run)
            self.assertEqual(run.state['consumed_seconds'], 460)

    def test_active_export_preserves_reservation_and_blocks_recovery(self):
        import os
        import time
        from proxybench.__main__ import evaluation_export
        self.training_export_state(used=150)
        root = self.path/'exports'
        atomic_json(root/'resources.active.json', dict(execution_id='still-running', supervisor_pid=os.getpid(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    started_monotonic=time.monotonic(), started_wall=time.time(), run=str(root)))
        with Run(self.path) as run, \
             patch('proxybench.training.runtime.export_complete', side_effect=AssertionError('Recovered active export')):
            with self.assertRaisesRegex(ValueError, 'supervisor still exists'):
                evaluation_export(run)
            self.assertEqual(run.state['consumed_seconds'], 550)
            self.assertEqual(run.state['pending_charge'], dict(target='model-export',seconds=300))
