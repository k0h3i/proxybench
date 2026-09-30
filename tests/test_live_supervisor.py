"""CPU-only process tests for live logs, interrupts, limits, and ownership."""

import io
import json
import os
import signal
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from proxybench.execution.live import Console, supervise
from proxybench.execution.resources import group_members, ledger_entries


class LiveSupervisorTests(unittest.TestCase):
    def run_worker(self, script, *, limits=None, stream=None, phase='test', expected_stderr=None,
                   expected_stdout=None, progress=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = dict(cpu_only=True, start_host_bytes=0, stop_host_bytes=0, sample_seconds=.1,
                            phase_seconds=5, total_seconds=10, grace_seconds=.3, heartbeat_seconds=.1)
            settings.update(limits or {})
            began = time.monotonic()
            result = supervise([sys.executable, '-u', '-c', script], root/'worker', settings,
                               ledger=root/'ledger.jsonl', phase=phase, console=stream or io.StringIO(),
                               progress=progress)
            elapsed = time.monotonic()-began
            saved = (root/'worker/stdout.log').read_text()
            if expected_stdout is not None:
                self.assertEqual((root/'worker/stdout.log').read_bytes(), expected_stdout)
            if expected_stderr is not None:
                self.assertEqual((root/'worker/stderr.log').read_bytes(), expected_stderr)
            record = json.loads((root/'worker/result.json').read_text())
            self.assertFalse(record['surviving_owned_pids'])
            self.assertFalse((root/'ledger.active.json').exists())
            self.assertEqual(len(ledger_entries(root/'ledger.jsonl')), 1)
            self.assertGreaterEqual(record['elapsed_seconds'], 0)
            self.assertEqual(record['worker_attempt_seconds'], record['elapsed_seconds'])
            self.assertEqual(record['failed_attempt_seconds'],
                             0 if result == 'EXITED' else record['elapsed_seconds'])
            return result, saved, elapsed, record

    def test_live_steps_saved_and_forwarded(self):
        stream = io.StringIO()
        result, saved, _, _ = self.run_worker("import time\nprint('step 001/192', flush=True)\ntime.sleep(.1)\nprint('step 002/192', flush=True)", stream=stream)
        self.assertEqual(result, 'EXITED')
        self.assertIn('step 001/192', saved)
        self.assertIn('step 002/192', stream.getvalue())

    def test_operation_output_stays_in_logs_with_compact_completion(self):
        stdout = b'Synthetic library banner\n'
        stderr = b'Synthetic library warning: \xff\n'
        for operation in ('export', 'convert', 'evaluation'):
            with self.subTest(operation=operation):
                stream = io.StringIO()
                script = f"""import os, sys
from proxybench.execution.resources import durable_json
sys.stdout.buffer.write({stdout!r})
sys.stdout.flush()
sys.stderr.buffer.write({stderr!r})
sys.stderr.flush()
durable_json(os.environ['PROXYBENCH_PHASE_FILE'],
             dict(phase='cleanup', completed=3, total=3, failed=0))
"""
                result, _, _, _ = self.run_worker(
                    script, phase=operation, stream=stream,
                    expected_stdout=stdout, expected_stderr=stderr)
                self.assertEqual(result, 'EXITED')
                text = stream.getvalue()
                self.assertNotIn('Synthetic library', text)
                self.assertNotIn(f'[{operation}:', text)
                self.assertIn('Cleanup 3/3 (100%) | generation failures 0', text)
                if operation == 'evaluation':
                    self.assertIn('Generation finished. Scoring and review are pending.', text)
                else:
                    self.assertIn(f'{operation.capitalize()} worker finished.', text)
                    self.assertNotIn('published', text)

    def test_operation_failure_keeps_status_counts_and_diagnostics(self):
        for operation in ('export', 'convert', 'evaluation'):
            with self.subTest(operation=operation):
                stream = io.StringIO()
                script = """import os, sys
from proxybench.execution.resources import durable_json
print('Hidden library banner', flush=True)
sys.stderr.write('Synthetic operation failure\\n')
sys.stderr.flush()
durable_json(os.environ['PROXYBENCH_PHASE_FILE'],
             dict(phase='cleanup', completed=1, total=3, failed=1))
raise SystemExit(7)
"""
                result, saved, _, record = self.run_worker(
                    script, phase=operation, stream=stream,
                    expected_stderr=b'Synthetic operation failure\n')
                self.assertEqual(result, 'PROCESS_FAILED')
                self.assertEqual(record['returncode'], 7)
                self.assertIn('Hidden library banner', saved)
                text = stream.getvalue()
                self.assertNotIn('Hidden library banner', text)
                self.assertIn('1/3 (33.3%) | generation failures 1', text)
                self.assertIn('Supervisor status: PROCESS_FAILED', text)
                self.assertIn('Synthetic operation failure', text)
                self.assertIn('stderr.log', text)
                self.assertNotIn('worker finished', text)
                self.assertNotIn('Generation finished', text)

    def test_adapter_loading_keeps_exact_output_in_logs_only(self):
        stdout = '🦥 Unsloth banner\nPhase: inference\n'.encode()
        stderr = b'Compiler warning: _POSIX_C_SOURCE redefined\nLibrary warning: \xff\n'
        script = f"""import os, sys
from proxybench.execution.resources import durable_json
sys.stdout.buffer.write({stdout!r})
sys.stdout.flush()
sys.stderr.buffer.write({stderr!r})
sys.stderr.flush()
durable_json(os.environ['PROXYBENCH_PHASE_FILE'], dict(phase='inference'))
"""
        stream = io.StringIO()
        result, _, _, _ = self.run_worker(script, phase='validate-adapter', stream=stream,
                                          expected_stdout=stdout, expected_stderr=stderr)
        self.assertEqual(result, 'EXITED')
        text = stream.getvalue()
        self.assertIn('Inference | elapsed', text)
        self.assertIn('Adapter loading test worker finished.', text)
        for hidden in ('Unsloth', 'Compiler warning', 'Library warning', 'Phase: inference', '[validate-adapter:'):
            self.assertNotIn(hidden, text)

    def test_adapter_loading_failure_still_shows_error_and_log_path(self):
        stream = io.StringIO()
        script = """import sys
print('Hidden Unsloth banner', flush=True)
sys.stderr.write('Final adapter error\\n')
sys.stderr.flush()
raise SystemExit(7)
"""
        result, saved, _, record = self.run_worker(script, phase='validate-adapter', stream=stream,
                                                   expected_stderr=b'Final adapter error\n')
        self.assertEqual(result, 'PROCESS_FAILED')
        self.assertEqual(record['returncode'], 7)
        self.assertIn('Hidden Unsloth banner', saved)
        text = stream.getvalue()
        self.assertNotIn('Hidden Unsloth banner', text)
        self.assertIn('Adapter loading test did not complete. Supervisor status: PROCESS_FAILED.', text)
        self.assertIn('Final adapter error', text)
        self.assertIn('Worker diagnostics:', text)
        self.assertIn('stderr.log', text)
        self.assertNotIn('worker finished', text)

    def test_evaluation_zero_exit_does_not_claim_missing_results_are_finished(self):
        stream = io.StringIO()
        script = """import os
from proxybench.execution.resources import durable_json
durable_json(os.environ['PROXYBENCH_PHASE_FILE'],
             dict(phase='cleanup', completed=2, total=3, failed=1))
"""
        result, _, _, _ = self.run_worker(script, phase='evaluation', stream=stream)
        self.assertEqual(result, 'EXITED')
        text = stream.getvalue()
        self.assertIn('2/3 (66.7%) | generation failures 1', text)
        self.assertIn('Generation incomplete', text)
        self.assertNotIn('Generation finished', text)
        self.assertIn('Worker diagnostics:', text)
        self.assertIn('answers/server.log', text)

    def test_evaluation_progress_adds_prior_results_and_retains_failure_count(self):
        stream = io.StringIO()
        script = """import os
from proxybench.execution.resources import durable_json
durable_json(os.environ['PROXYBENCH_PHASE_FILE'],
             dict(phase='cleanup', completed=3, total=3, failed=1))
"""
        result, _, _, _ = self.run_worker(
            script, phase='evaluation', stream=stream,
            progress=dict(completed=2, total=5, failed=1))
        self.assertEqual(result, 'EXITED')
        text = stream.getvalue()
        self.assertIn('Loading 2/5 (40%) | generation failures 1', text)
        self.assertIn('Cleanup 5/5 (100%) | generation failures 2', text)
        self.assertIn('Generation finished. Scoring and review are pending.', text)
        self.assertNotIn('Evaluation complete', text)

    def test_unbounded_time_keeps_ledger_and_memory_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root/'ledger.jsonl'
            ledger.write_text(json.dumps(dict(status='PHASE_TIMEOUT', phase='inference',
                                              elapsed_seconds=1800.1))+'\n')
            limits = dict(cpu_only=True, start_host_bytes=0, stop_host_bytes=0,
                          phase_seconds=None, total_seconds=None, sample_seconds=.05,
                          heartbeat_seconds=.05, grace_seconds=.3)
            script = "import os, time; assert 'PROXYBENCH_PHASE_DEADLINE' not in os.environ; assert 'PROXYBENCH_TOTAL_DEADLINE' not in os.environ; time.sleep(.15)"
            result = supervise([sys.executable, '-u', '-c', script], root/'worker', limits,
                               ledger=ledger, phase='inference', phase_used=1800.1, console=io.StringIO())
            self.assertEqual(result, 'EXITED')
            self.assertEqual([entry['status'] for entry in ledger_entries(ledger)], ['PHASE_TIMEOUT', 'EXITED'])
            self.assertGreater((root/'worker/memory.jsonl').stat().st_size, 0)
            record = json.loads((root/'worker/result.json').read_text())
            self.assertEqual(record['failed_attempt_seconds'], 0)
            self.assertEqual(record['cumulative_failed_attempt_seconds'], 1800.1)

    def test_interrupt_during_terminal_queue_write_does_not_reenter_it(self):
        original, in_write, reentered = Console.put, [], []
        def put(console, value):
            if in_write:
                reentered.append(True)
            in_write.append(True)
            if value and value.startswith('[test:'):
                os.kill(os.getpid(), signal.SIGINT)
            original(console, value)
            in_write.pop()
        with patch.object(Console, 'put', put):
            result, _, _, _ = self.run_worker('import time; time.sleep(10)')
        self.assertEqual(result, 'USER_STOP')
        self.assertEqual(reentered, [])

    def test_slow_terminal_does_not_block_time_limit(self):
        release = threading.Event()
        class Slow:
            def write(self, value):
                release.wait(3)
            def flush(self):
                pass
        try:
            result, saved, elapsed, _ = self.run_worker("import time\nprint('visible later', flush=True)\ntime.sleep(10)",
                                      limits={'phase_seconds': .4}, stream=Slow())
            self.assertEqual(result, 'PHASE_TIMEOUT')
            self.assertLess(elapsed, 2)
            self.assertIn('visible later', saved)
        finally:
            release.set()

    def test_first_interrupt_during_update_or_save_can_finish_safely(self):
        for phase in ('training', 'compilation', 'saving', 'saving checkpoint'):
            with self.subTest(phase=phase):
                script = f"""import os, signal, time
from pathlib import Path
from proxybench.execution.resources import durable_json
durable_json(os.environ['PROXYBENCH_PHASE_FILE'], {{'phase': '{phase}'}})
time.sleep(.1)
os.kill(os.getppid(), signal.SIGINT)
while not Path(os.environ['PROXYBENCH_STOP_FILE']).exists():
    time.sleep(.01)
print('safe boundary saved', flush=True)
"""
                result, saved, _, _ = self.run_worker(script)
                self.assertEqual(result, 'EXITED')
                self.assertIn('safe boundary saved', saved)

    def test_interrupt_during_loading_or_evaluation_stops_without_checkpoint(self):
        for phase in ('loading', 'evaluation', 'development-loss'):
            with self.subTest(phase=phase):
                script = f"""import os, signal, time
from proxybench.execution.resources import durable_json
durable_json(os.environ['PROXYBENCH_PHASE_FILE'], {{'phase': '{phase}'}})
time.sleep(.1)
os.kill(os.getppid(), signal.SIGINT)
time.sleep(5)
print('must not complete', flush=True)
"""
                result, saved, _, _ = self.run_worker(script)
                self.assertEqual(result, 'USER_STOP')
                self.assertNotIn('must not complete', saved)

    def test_second_interrupt_and_exhausted_grace_stop_owned_group(self):
        for second in (False, True):
            with self.subTest(second=second):
                script = """import os, signal, subprocess, sys, time
from proxybench.execution.resources import durable_json
durable_json(os.environ['PROXYBENCH_PHASE_FILE'], {'phase': 'training'})
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
print('child', child.pid, flush=True)
time.sleep(.1)
os.kill(os.getppid(), signal.SIGINT)
""" + ("time.sleep(.1)\nos.kill(os.getppid(), signal.SIGINT)\n" if second else '') + 'time.sleep(5)\n'
                result, saved, elapsed, _ = self.run_worker(script)
                self.assertEqual(result, 'FORCED_STOP' if second else 'STOP_GRACE_EXHAUSTED')
                self.assertLess(elapsed, 2)

    def test_phase_label_never_resets_clock_and_nonzero_exit_is_preserved(self):
        script = """import os, time
from proxybench.execution.resources import durable_json
for index in range(100):
    durable_json(os.environ['PROXYBENCH_PHASE_FILE'], {'phase': str(index)})
    time.sleep(.05)
"""
        result, _, elapsed, _ = self.run_worker(script, limits={'phase_seconds': .4})
        self.assertEqual(result, 'PHASE_TIMEOUT')
        self.assertLess(elapsed, 2)
        result, _, _, record = self.run_worker('raise SystemExit(7)')
        self.assertEqual(result, 'PROCESS_FAILED')
        self.assertEqual(record['returncode'], 7)

    def test_hard_limit_does_not_wait_for_stop_grace(self):
        script = """import os, signal, time
from proxybench.execution.resources import durable_json
durable_json(os.environ['PROXYBENCH_PHASE_FILE'], {'phase': 'saving'})
time.sleep(.1)
os.kill(os.getppid(), signal.SIGINT)
time.sleep(5)
"""
        result, _, elapsed, _ = self.run_worker(script, limits={'phase_seconds': .4, 'grace_seconds': 30})
        self.assertEqual(result, 'PHASE_TIMEOUT')
        self.assertLess(elapsed, 2)

    def test_training_requests_clean_stop_before_hard_deadline(self):
        script = """import os, time
from pathlib import Path
from proxybench.execution.resources import durable_json
durable_json(os.environ['PROXYBENCH_PHASE_FILE'], {'phase': 'training'})
while not Path(os.environ['PROXYBENCH_STOP_FILE']).exists():
    time.sleep(.01)
print('clean stop saved', flush=True)
"""
        result, saved, _, record = self.run_worker(
            script, limits={'phase_seconds': 1.2, 'automatic_stop_margin_seconds': .6})
        self.assertEqual(result, 'EXITED')
        self.assertIn('clean stop saved', saved)
        self.assertTrue(record['automatic_stop_requested'])

    def test_slow_memory_query_cannot_postpone_hard_deadline(self):
        release = threading.Event()
        calls = []
        def memory():
            if calls:
                release.wait(3)
            calls.append(True)
            return {'available_bytes': 100}
        try:
            with patch('proxybench.execution.live.host_memory', memory):
                result, _, elapsed, _ = self.run_worker('import time; time.sleep(10)', limits={'phase_seconds': .4})
            self.assertEqual(result, 'PHASE_TIMEOUT')
            self.assertLess(elapsed, 2)
        finally:
            release.set()

    def test_training_display_consolidates_output_and_retains_saved_bytes(self):
        stream = io.StringIO()
        script = """import sys
from proxybench.training.measurements import MeasurementRecorder, event_line
measurements = MeasurementRecorder(emit=lambda event: print(event_line(event), flush=True))
print('Library banner: model loading', flush=True)
measurements.phase('training', global_step=660, sample_position=660, planned_updates=660,
                   epoch_boundaries=[330, 660], batch_size=1,
                   gradient_accumulation_steps=1, effective_batch_size=1)
print('Phase: training', flush=True)
print('step 660/660 | loss 1', flush=True)
sys.stderr.buffer.write(b'warning bytes: \\xff\\n')
sys.stderr.flush()
measurements.phase('validating final adapter')
measurements.phase('publishing final adapter')
measurements.event('attempt', status='COMPLETE', result_published=True)
"""
        result, saved, _, _ = self.run_worker(script, phase='train', stream=stream,
                                             expected_stderr=b'warning bytes: \xff\n')
        self.assertEqual(result, 'EXITED')
        self.assertIn('PROXYBENCH_EVENT', saved)
        self.assertIn('Library banner: model loading', saved)
        self.assertIn('step 660/660 | loss 1', saved)
        text = stream.getvalue()
        self.assertNotIn('PROXYBENCH_EVENT', text)
        self.assertNotIn('step 660/660 | loss 1', text)
        self.assertNotIn('[train:', text)
        self.assertNotIn('warning bytes:', text)
        self.assertNotIn('Library banner:', text)
        self.assertIn('Publishing final adapter', text)
        self.assertIn('Training complete.', text)

    def test_training_failed_publication_does_not_report_success(self):
        stream = io.StringIO()
        script = """from proxybench.training.measurements import MeasurementRecorder, event_line
measurements = MeasurementRecorder(emit=lambda event: print(event_line(event), flush=True))
measurements.phase('publishing final adapter', global_step=660, sample_position=660,
                   planned_updates=660, epoch_boundaries=[330, 660])
raise RuntimeError('publication failed')
"""
        result, _, _, _ = self.run_worker(script, phase='train', stream=stream)
        self.assertEqual(result, 'PROCESS_FAILED')
        self.assertIn('660/660 updates (100%)', stream.getvalue())
        self.assertNotIn('Training complete.', stream.getvalue())
        self.assertIn('Supervisor status: PROCESS_FAILED', stream.getvalue())
        self.assertIn('RuntimeError: publication failed', stream.getvalue())
        self.assertIn('stderr.log', stream.getvalue())

    def test_training_failure_shows_bounded_stderr_tail_and_capture_path(self):
        stream = io.StringIO()
        noise = b'library banner\n' * 350000
        tail = b'RuntimeError: synthetic failure\n'
        stderr = noise + tail
        script = ("import sys\nsys.stderr.buffer.write(b'library banner\\n' * 350000)\n"
                  f'sys.stderr.buffer.write({tail!r})\nsys.stderr.flush()\nraise SystemExit(7)\n')
        result, _, _, _ = self.run_worker(script, phase='train', stream=stream,
                                          expected_stderr=stderr)
        self.assertEqual(result, 'PROCESS_FAILED')
        text = stream.getvalue()
        self.assertIn('RuntimeError: synthetic failure', text)
        self.assertIn('stderr.log', text)
        self.assertLessEqual(text.count('library banner'), 12)
        self.assertLess(len(text), 10000)

    def test_worker_reported_failure_shows_diagnostics_even_with_zero_exit(self):
        stream = io.StringIO()
        script = """import sys
from proxybench.training.measurements import MeasurementRecorder, event_line
measurements = MeasurementRecorder(emit=lambda event: print(event_line(event), flush=True))
sys.stderr.write('Worker validation failed\\n')
measurements.event('attempt', status='FAILED')
"""
        result, _, _, _ = self.run_worker(script, phase='train', stream=stream,
                                          expected_stderr=b'Worker validation failed\n')
        self.assertEqual(result, 'EXITED')
        self.assertIn('Worker validation failed', stream.getvalue())
        self.assertIn('stderr.log', stream.getvalue())
        self.assertNotIn('Training complete.', stream.getvalue())

    def test_training_tty_refreshes_before_phase_change_with_long_heartbeat(self):
        class Terminal(io.StringIO):
            def __init__(self):
                super().__init__()
                self.frames = []

            def isatty(self):
                return True

            def write(self, value):
                self.frames.append((time.monotonic(), value))
                return super().write(value)

        stream = Terminal()
        script = """import time
from proxybench.training.measurements import MeasurementRecorder, event_line
measurements = MeasurementRecorder(emit=lambda event: print(event_line(event), flush=True))
measurements.phase('training', global_step=0, sample_position=0, planned_updates=660,
                   epoch_boundaries=[330, 660])
time.sleep(.15)
measurements.event('update', global_step=1, sample_position=1, loss=.75)
time.sleep(2)
measurements.phase('saving checkpoint')
"""
        result, _, _, _ = self.run_worker(script, phase='train', stream=stream,
                                          limits={'heartbeat_seconds': 180})
        self.assertEqual(result, 'EXITED')
        initial_at = next(at for at, frame in stream.frames if '0/660' in frame)
        save_at = next(at for at, frame in stream.frames if 'Saving checkpoint' in frame)
        updates = [(at, frame) for at, frame in stream.frames
                   if '1/660' in frame and at < save_at]
        self.assertTrue(updates, 'An ordinary update must appear while training is still active')
        self.assertLess(updates[0][0] - initial_at, 1.6)
        self.assertIn('loss 0.75', updates[0][1])
        self.assertIn('\x1b[2K', updates[0][1])

    def test_malformed_training_measurements_preserve_worker_and_original_output(self):
        from proxybench.training.measurements import EVENT_PREFIX, EVENT_SCHEMA, event_line
        stream = io.StringIO()
        common = dict(schema_version=EVENT_SCHEMA, attempt_elapsed_seconds=0)
        events = [dict(common, kind='phase', phase='training', global_step=0, sample_position=0,
                       planned_updates=10, epoch_boundaries=[10]),
                  dict(common, kind='phase', phase=[]),
                  dict(common, kind='update', global_step=10, sample_position=10, loss=10**500),
                  dict(common, kind='phase', phase='publishing final adapter'),
                  dict(common, kind='attempt', status='COMPLETE', result_published=True)]
        nested = EVENT_PREFIX + '[' * 10000 + '0' + ']' * 10000 + '\n'
        wire = (nested + ''.join(event_line(item) + '\n' for item in events)).encode()
        script = f'import sys\nsys.stdout.buffer.write({wire!r})\nsys.stdout.flush()\n'
        result, _, _, record = self.run_worker(script, phase='train', stream=stream, expected_stdout=wire)
        self.assertEqual(result, 'EXITED')
        self.assertEqual(record['returncode'], 0)
        self.assertNotIn('loss unavailable', stream.getvalue())
        self.assertIn('Training complete.', stream.getvalue())

    def test_training_saving_phase_keeps_budget_stop_and_slow_terminal_is_bounded(self):
        release = threading.Event()
        class Slow:
            def isatty(self):
                return True
            def write(self, value):
                release.wait(3)
            def flush(self):
                pass
        script = """import os, time
from pathlib import Path
from proxybench.execution.resources import durable_json
from proxybench.training.measurements import MeasurementRecorder, event_line
measurements = MeasurementRecorder(emit=lambda event: print(event_line(event), flush=True))
measurements.phase('saving checkpoint', global_step=330, sample_position=330,
                   planned_updates=660, epoch_boundaries=[330, 660])
durable_json(os.environ['PROXYBENCH_PHASE_FILE'], {'phase': 'saving checkpoint'})
print('saved output survives a slow terminal', flush=True)
while not Path(os.environ['PROXYBENCH_STOP_FILE']).exists():
    time.sleep(.01)
print('clean stop saved', flush=True)
measurements.event('attempt', status='CLEAN_STOP', resume_eligible=True)
"""
        try:
            result, saved, elapsed, record = self.run_worker(
                script, phase='train', stream=Slow(),
                limits={'phase_seconds': 1.2, 'automatic_stop_margin_seconds': .6})
            self.assertEqual(result, 'EXITED')
            self.assertTrue(record['automatic_stop_requested'])
            self.assertLess(elapsed, 2)
            self.assertIn('saved output survives a slow terminal\n', saved)
            self.assertIn('clean stop saved\n', saved)
        finally:
            release.set()
