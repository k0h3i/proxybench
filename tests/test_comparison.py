"""Failure behavior for the controlled operator schedule, using synthetic data."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from proxybench.annotation.bindings import sha256
from proxybench.execution import comparison as c
from proxybench.execution.runner import ScheduledInput


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def transcript(prompt=b'synthetic', effort='medium', tool=False):
    payloads = [('session_meta', {'id': 'session', 'base_instructions': {'text': 'synthetic base'}}),
                ('turn_context', {'model': c.MODEL, 'effort': effort, 'cwd': '/work', 'turn_id': 'turn'}),
                ('response_item', {'type': 'message', 'role': 'developer', 'content': [{'text': 'synthetic rules'}]}),
                ('response_item', {'type': 'message', 'role': 'user', 'content': [{'text': '<environment_context>test'}]}),
                ('response_item', {'type': 'message', 'role': 'user', 'content': [{'text': prompt.decode()}]}),
                ('response_item', {'type': 'message', 'role': 'assistant', 'phase': 'final_answer', 'content': [{'text': '{}'}]})]
    if tool:
        payloads.append(('response_item', {'type': 'function_call', 'name': 'read_file'}))
    return [{'type': a, 'payload': b} for a, b in payloads]


def put_transcript(directory, rows):
    save(directory / 'rollout-0.jsonl', b'\n'.join(json.dumps(x).encode() for x in rows) + b'\n')
    events = [{'type': 'thread.started', 'thread_id': 'session'}, {'type': 'turn.completed', 'usage': {}}]
    save(directory / 'stdout.bin', b'\n'.join(json.dumps(x).encode() for x in events) + b'\n')


class CaptureTests(unittest.TestCase):
    def test_expired_global_deadline_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c.subprocess, 'Popen') as launch:
            d = Path(tmp)
            save(d / 'input', b'')
            result = c.capture(['unused'], d/'input', d, seconds=1,
                               global_deadline=time.monotonic()-1, limit=100)
            self.assertEqual(result['outcome'], 'TIMEOUT')
            launch.assert_not_called()

    def run_capture(self, code, *, seconds=.3, limit=1024, state=False):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save(d / 'input', b'')
            state_path = d / 'state' if state else None
            if state:
                state_path.mkdir()
                code = code.replace('STATE', repr(str(state_path)))
            result = c.capture([sys.executable, '-c', code], d / 'input', d, seconds=seconds,
                               global_deadline=time.monotonic() + 10, limit=limit, state=state_path)
            return result, {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}

    def test_timeout_keeps_partial_and_cancels(self):
        r, files = self.run_capture("import os,time; os.write(1,b'partial'); time.sleep(5)")
        self.assertEqual(r['outcome'], 'TIMEOUT')
        self.assertTrue(r['termination_established'])
        self.assertEqual(files['raw.bin'], b'partial')

    def test_stream_limit(self):
        r, files = self.run_capture("import os; os.write(1,b'x'*10000)", limit=50)
        self.assertEqual(r['outcome'], 'TRUNCATED')
        self.assertEqual(len(files['raw.bin']), 50)

    def test_late_native_final_never_becomes_primary(self):
        r, files = self.run_capture("from pathlib import Path; import time; Path(STATE+'/final.txt').write_bytes(b'{}'); time.sleep(5)", state=True)
        self.assertEqual(r['outcome'], 'TIMEOUT')
        self.assertNotIn('raw.bin', files)
        self.assertEqual(files['late-final.bin'], b'{}')

    def test_fast_final_cannot_bypass_limit(self):
        with patch.dict(c.LIMITS, output_bytes=20):
            result, files = self.run_capture("from pathlib import Path; Path(STATE+'/final.txt').write_bytes(b'x'*100)", state=True)
            self.assertEqual(result['outcome'], 'TRUNCATED')
            self.assertNotIn('raw.bin', files)
            self.assertEqual(len(files['late-final.bin']), 20)

    def test_phase_interrupts_preparation(self):
        started = time.monotonic()
        with self.assertRaises(c.DeadlineError):
            with c.phase(started + .05):
                time.sleep(5)
        self.assertLess(time.monotonic() - started, .5)

    def test_incomplete_durable_write_is_rejected(self):
        stream = unittest.mock.MagicMock()
        stream.__enter__.return_value = stream
        stream.write.return_value = 1
        with patch.object(Path, 'open', return_value=stream):
            with self.assertRaisesRegex(c.IntegrityError, 'Incomplete'):
                c.durable(Path('unused'), b'longer')


class ProvenanceTests(unittest.TestCase):
    def test_session_model_context_and_export_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = transcript()
            put_transcript(d, rows)
            context = c.application_context(rows)
            self.assertEqual(c.sol_provenance(d, b'synthetic', b'{}', True, context)['effort'], 'medium')
            for prompt, raw in ((b'swapped-input', b'{}'), (b'synthetic', b'changed')):
                with self.assertRaises(c.IntegrityError):
                    c.sol_provenance(d, prompt, raw, True, context)
            rows[2]['payload']['content'] = [{'text': 'unexpected reference instructions'}]
            put_transcript(d, rows)
            with self.assertRaisesRegex(c.IntegrityError, 'context differs'):
                c.sol_provenance(d, b'synthetic', b'{}', True, context)

    def test_tool_effort_and_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            with self.assertRaises(c.IntegrityError):
                c.sol_provenance(d, b'synthetic', None, False)
            for rows in (transcript(effort='low'), transcript(tool=True)):
                put_transcript(d, rows)
                with self.assertRaises(c.IntegrityError):
                    c.sol_provenance(d, b'synthetic', b'{}', True)

    def test_stale_added_and_symlinked_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save(d / 'module.py', b'original')
            expected = c.file_hashes(d)
            c.assert_package(d, expected)
            save(d / 'module.py', b'changed')
            with self.assertRaises(c.IntegrityError):
                c.assert_package(d, expected)
            save(d / 'module.py', b'original')
            save(d / 'sitecustomize.py', b'added')
            with self.assertRaises(c.IntegrityError):
                c.assert_package(d, expected)


class ScheduleTests(unittest.TestCase):
    def exercise(self, *, incident=None, reserve=False, prep_timeout=False, final_timeout=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / 'snapshot'
            save(snapshot / '__init__.py', b'')
            save(root / 'baseline.json', {'software': {'package_hashes': c.file_hashes(snapshot)}})
            items = [ScheduledInput(f'synthetic-{i}', 'input', sha256(b'source'), 'reference', 'a'*64, 'b'*64) for i in range(6)]
            save(root / 'schedule.json', [asdict(x) for x in items])
            save(root / 'contract.md', b'Synthetic schema only')
            save(root / 'models.json', {})
            save(root / 'auth.json', {'auth_mode': 'chatgpt'})
            put_transcript(root / 'probe', transcript())
            d = root / 'run'
            preflight_result = (b'source', None, None)
            with patch.object(c, 'preflight', return_value=preflight_result):
                c.freeze(root, d, schedule_path=root/'schedule.json', snapshot=snapshot,
                         baseline=root/'baseline.json', cli=sys.executable, contract=root/'contract.md',
                         authorization='SYNTHETIC ONLY', model_cache=root/'models.json', probe=root/'probe/rollout-0.jsonl')
            calls = []
            def fake_capture(command, input_path, directory, **kw):
                calls.append(directory.name)
                save(directory / 'raw.bin', b'{}')
                save(directory / 'stdout.bin', b'')
                if incident:
                    raise c.IntegrityError(incident)
                return {'outcome': 'EXITED', 'returncode': 0, 'termination_established': True}
            def admission(*a):
                if prep_timeout and len(calls) == 0 and admission.calls >= 6:
                    raise c.DeadlineError()
                admission.calls += 1
                return preflight_result
            admission.calls = 0
            with patch.object(c, 'preflight', side_effect=admission), patch.object(c, 'capture', side_effect=fake_capture), \
                 patch.object(c, 'sol_provenance', side_effect=c.DeadlineError() if final_timeout else None,
                              return_value={'session_id': 'synthetic'}), \
                 patch.object(c, 'score_fragment', return_value={'passed': False}), \
                 patch.object(c, 'parsed_response', return_value=({}, 0, {}, [], False)), redirect_stdout(io.StringIO()):
                if reserve:
                    # Freeze uses the same limits, then reduce elapsed budget through the clock.
                    clock = time.monotonic()
                    with patch.object(c.time, 'monotonic', side_effect=[clock, clock + 2200, clock + 2200] + [clock + 2200]*50):
                        report = c.execute(d, auth=root/'auth.json')
                else:
                    report = c.execute(d, auth=root/'auth.json')
            return report, calls

    def test_all_twelve_ordered_slots(self):
        report, calls = self.exercise()
        self.assertEqual(len(calls), 12)
        self.assertEqual(report['terminal_results'], 12)
        self.assertEqual([r['association']['system_id'] for r in report['results'][:4]], ['parser','sol','sol','parser'])

    def test_integrity_stop_keeps_all_denominators(self):
        report, calls = self.exercise(incident='binding changed')
        self.assertEqual(len(calls), 1)
        self.assertEqual(report['terminal_results'], 12)
        self.assertEqual(sum(r['status']=='NOT_ATTEMPTED_STOP' for r in report['results']), 11)
        self.assertIsNone(report['summaries']['sol']['controlled_success'])

    def test_reservation_stops_before_either_member(self):
        report, calls = self.exercise(reserve=True)
        self.assertEqual(calls, [])
        self.assertEqual(report['terminal_results'], 12)

    def test_preparation_timeout_submits_no_task(self):
        report, calls = self.exercise(prep_timeout=True)
        self.assertEqual(calls, [])
        self.assertTrue(all(r['status']=='PREPARATION_TIMEOUT' for r in report['results']))

    def test_missing_finalization_provenance_stops_remaining_slots(self):
        report, calls = self.exercise(final_timeout=True)
        self.assertEqual(len(calls), 2)
        self.assertIn('provenance missing', report['stop'])
        self.assertEqual(report['terminal_results'], 12)


if __name__ == '__main__':
    unittest.main()
