"""Shared evaluation sessions retain answers and account for one model lifetime."""
from argparse import Namespace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

from proxybench.evaluation.session import generate_session, session_request
from proxybench.evaluation.workflow import create_run, load_answers, report
from proxybench.execution.resources import append_entry
from proxybench.runstate import Run, atomic_json, binding
from test_evaluation_workflow import label


class EvaluationSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.config = dict(model='unavailable.gguf', limits=dict(phase_seconds=7, total_seconds=100))
        self.inputs = dict(identity=dict(dataset='d', model='m', prompt='p', runtime='r'),
                           cases=[dict(id=str(i), reference=label(), source_cells=[],
                                       messages=[dict(role='user', content=f'case {i}')]) for i in range(3)])
        create_run(self.root, self.inputs, self.config)

    def answer(self, index, status='COMPLETE'):
        return dict(index=index, status=status, text=json.dumps(label()), token_ids=[1, 9])

    def register(self, run, *, status='EXITED', elapsed=8, cases=None, reserve=True):
        cases = self.inputs['cases'] if cases is None else cases
        name = 'session-' + uuid.uuid4().hex
        folder = self.root / 'evaluation' / 'capture' / name
        request = session_request(self.inputs, self.config, cases)
        atomic_json(folder / 'request.json', request)
        run.state.setdefault('evaluation_sessions', []).append(dict(name=name, request_sha256=binding(request)))
        run.save()
        if reserve:
            run.reserve(name, run.state['resource_limit_seconds'] - run.state['consumed_seconds'])
        if status is not None:
            append_entry(folder / 'resources.jsonl', dict(execution_id=name, status=status, elapsed_seconds=elapsed))
        return folder

    def complete(self, folder, count, *, elapsed=8):
        for index in range(count):
            atomic_json(folder / 'answers' / f'answer-{index}.json', self.answer(index))
        atomic_json(folder / 'progress.json', dict(index=count-1, state='COMPLETE'))
        append_entry(folder / 'resources.jsonl', dict(execution_id=folder.name, status='EXITED', elapsed_seconds=elapsed))

    def test_all_cases_share_one_supervised_attempt_and_one_charge(self):
        def supervise(command, output, limits, **kwargs):
            self.assertEqual(command[-2], 'proxybench.evaluation.worker')
            folder = Path(command[-1])
            request = json.loads((folder / 'request.json').read_text())
            self.assertEqual(request, session_request(self.inputs, self.config, self.inputs['cases']))
            self.assertEqual(limits['phase_seconds'], 100)
            self.assertEqual(limits['total_seconds'], 100)
            self.complete(folder, 3)
            return 'EXITED'
        with Run(self.root) as run, patch('proxybench.execution.live.supervise', side_effect=supervise) as worker:
            answers = generate_session(run, self.inputs)
            self.assertEqual(set(answers), {'0', '1', '2'})
            self.assertEqual(run.state['consumed_seconds'], 8)
            self.assertIsNone(run.state['pending_charge'])
            self.assertEqual(generate_session(run, self.inputs), answers)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(run.state['consumed_seconds'], 8)

    def test_user_stop_preserves_first_answer_and_resumes_only_missing_cases(self):
        with Run(self.root) as run:
            folder = self.register(run, status='USER_STOP', elapsed=12)
            atomic_json(folder / 'answers/answer-0.json', self.answer(0))
            atomic_json(folder / 'answers/answer-1.json', dict(index=1, status='STARTED'))
            atomic_json(folder / 'progress.json', dict(index=1, state='STARTED'))
        def supervise(command, output, limits, **kwargs):
            folder = Path(command[-1])
            request = json.loads((folder / 'request.json').read_text())
            self.assertEqual([case['id'] for case in request['cases']], ['1', '2'])
            self.assertEqual(limits['total_seconds'], 88)
            self.complete(folder, 2)
            return 'EXITED'
        with Run(self.root) as run, patch('proxybench.execution.live.supervise', side_effect=supervise):
            self.assertEqual(set(generate_session(run, self.inputs)), {'0', '1', '2'})
            self.assertEqual(run.state['consumed_seconds'], 20)
        with Run(self.root) as run:
            load_answers(run, self.inputs)
            self.assertEqual(run.state['consumed_seconds'], 20)

    def test_terminal_answer_wins_over_stale_active_marker_and_needs_no_model(self):
        from proxybench.__main__ import evaluation_command
        with Run(self.root) as run:
            folder = self.register(run, status='USER_STOP')
            for index in range(3):
                atomic_json(folder / 'answers' / f'answer-{index}.json', self.answer(index))
            atomic_json(folder / 'progress.json', dict(index=2, state='STARTED'))
        args = Namespace(command='evaluate', run_dir=str(self.root), report_only=False, model=None)
        with patch('proxybench.__main__.file_hash', side_effect=AssertionError('Model unavailable')), \
             patch('proxybench.evaluation.workflow.runtime_binding', side_effect=AssertionError('Runtime unavailable')):
            result = evaluation_command(args)
        self.assertEqual(result['status'], 'PENDING_REVIEW')
        with Run(self.root) as run:
            self.assertEqual(run.state['consumed_seconds'], 8)

    def test_request_timeout_counts_only_active_case_and_keeps_partial_tokens(self):
        with Run(self.root) as run:
            folder = self.register(run, status='REQUEST_TIMEOUT', elapsed=18)
            atomic_json(folder / 'answers/answer-0.json', self.answer(0))
            atomic_json(folder / 'answers/answer-1.json', dict(index=1, status='STARTED'))
            (folder / 'answers/answer-1.tokens.jsonl').write_text('{"token_id": 4}\n{"token_id": 5}\n{"tok')
            atomic_json(folder / 'progress.json', dict(index=1, state='STARTED'))
            answers = load_answers(run, self.inputs)
            self.assertEqual(set(answers), {'0', '1'})
            self.assertEqual(answers['1']['status'], 'TIMEOUT')
            self.assertEqual(answers['1']['token_ids'], [4, 5])
            result = report(run, self.inputs, answers)
            self.assertEqual(result['missing_answers'], ['2'])
            self.assertEqual(run.state['consumed_seconds'], 18)

    def test_user_stop_does_not_import_an_unfinished_failure_envelope(self):
        with Run(self.root) as run:
            folder = self.register(run, status='USER_STOP')
            atomic_json(folder / 'answers/answer-0.json', self.answer(0, 'FAILED'))
            atomic_json(folder / 'progress.json', dict(index=0, state='STARTED'))
            self.assertEqual(load_answers(run, self.inputs), {})
            atomic_json(folder / 'progress.json', dict(index=0, state='COMPLETE'))
            self.assertEqual(load_answers(run, self.inputs)['0']['status'], 'FAILED')

    def test_startup_failure_does_not_fail_unattempted_cases(self):
        with Run(self.root) as run:
            self.register(run, status='PROCESS_FAILED', elapsed=11)
            self.assertEqual(load_answers(run, self.inputs), {})
            self.assertEqual(run.state['consumed_seconds'], 11)
            self.assertIsNone(run.state['pending_charge'])

    def test_raw_length_stop_is_normalized_after_interrupted_publication(self):
        with Run(self.root) as run:
            self.config['response_tokens'] = 2
            run.state['configuration'] = self.config
            folder = self.register(run, status='USER_STOP')
            raw = self.answer(0, 'TERMINATION_MISMATCH')
            raw['terminal'] = dict(stop_type='limit', tokens_predicted=2)
            atomic_json(folder / 'answers/answer-0.json', raw)
            atomic_json(folder / 'progress.json', dict(index=0, state='STARTED'))
            self.assertEqual(load_answers(run, self.inputs)['0']['status'], 'LENGTH_STOP')
            self.assertEqual(json.loads((folder / 'answers/answer-0.json').read_text()), raw)
            self.assertEqual(load_answers(run, self.inputs)['0']['status'], 'LENGTH_STOP')

    def test_capture_without_execution_ledger_is_rejected(self):
        with Run(self.root) as run:
            folder = self.register(run, status=None)
            atomic_json(folder / 'answers/answer-0.json', self.answer(0))
            with self.assertRaisesRegex(ValueError, 'no reconciled execution ledger'):
                load_answers(run, self.inputs)
            self.assertEqual(run.state['consumed_seconds'], 100)
            self.assertEqual(run.state['pending_charge']['target'], folder.name)

    def test_preflight_error_settles_unused_reservation_without_failed_cases(self):
        with Run(self.root) as run, patch('proxybench.execution.live.supervise', side_effect=ValueError('Memory unavailable')):
            with self.assertRaisesRegex(ValueError, 'Memory unavailable'):
                generate_session(run, self.inputs)
            self.assertEqual(load_answers(run, self.inputs), {})
            self.assertEqual(run.state['consumed_seconds'], 0)
            self.assertIsNone(run.state['pending_charge'])

    def test_active_owner_blocks_recovery_and_preserves_reservation(self):
        with Run(self.root) as run:
            folder = self.register(run, status=None)
            atomic_json(folder / 'resources.active.json', dict(execution_id=folder.name, supervisor_pid=os.getpid(),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        started_monotonic=time.monotonic(), started_wall=time.time(), run=str(folder)))
            atomic_json(folder / 'answers/answer-0.json', self.answer(0))
            with self.assertRaisesRegex(ValueError, 'supervisor still exists'):
                load_answers(run, self.inputs)
            self.assertEqual(run.state['consumed_seconds'], 100)
            self.assertEqual(run.state['pending_charge']['target'], folder.name)

    def test_reconciled_elapsed_exceeding_reservation_cannot_be_reset(self):
        with Run(self.root) as run:
            self.register(run, status='RECONCILED', elapsed=105)
            self.assertEqual(load_answers(run, self.inputs), {})
            self.assertEqual(run.state['consumed_seconds'], 105)
            with patch('proxybench.execution.live.supervise', side_effect=AssertionError('Launched exhausted run')):
                with self.assertRaisesRegex(ValueError, 'cumulative run time limit'):
                    generate_session(run, self.inputs)
        with Run(self.root) as run:
            load_answers(run, self.inputs)
            self.assertEqual(run.state['consumed_seconds'], 105)

    def test_prior_training_and_export_charge_is_preserved(self):
        with Run(self.root) as run:
            run.state.update(consumed_seconds=35, evaluation_resource_base_seconds=35)
            self.register(run, elapsed=12)
            load_answers(run, self.inputs)
            self.assertEqual(run.state['consumed_seconds'], 47)
            load_answers(run, self.inputs)
            self.assertEqual(run.state['consumed_seconds'], 47)

    def test_changed_manifest_and_stale_messages_are_rejected(self):
        with Run(self.root) as run:
            folder = self.register(run)
            request = json.loads((folder / 'request.json').read_text())
            request['cases'][0]['messages'][0]['content'] = 'changed'
            atomic_json(folder / 'request.json', request)
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                load_answers(run, self.inputs)
            run.state['evaluation_sessions'][0]['request_sha256'] = binding(request)
            with self.assertRaisesRegex(ValueError, 'stale inputs'):
                load_answers(run, self.inputs)

    def test_mismatched_index_and_conflicting_captures_are_rejected(self):
        with Run(self.root) as run:
            folder = self.register(run)
            atomic_json(folder / 'answers/answer-0.json', self.answer(1))
            with self.assertRaisesRegex(ValueError, 'mismatched case index'):
                load_answers(run, self.inputs)
            atomic_json(folder / 'answers/answer-0.json', self.answer(0))
            load_answers(run, self.inputs)
            other = self.register(run, reserve=False)
            atomic_json(other / 'answers/answer-0.json', self.answer(0, 'TIMEOUT'))
            with self.assertRaisesRegex(ValueError, 'Conflicting evaluation captures'):
                load_answers(run, self.inputs)


if __name__ == '__main__':
    unittest.main()
