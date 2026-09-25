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
    def run_worker(self, script, *, limits=None, stream=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = dict(cpu_only=True, start_host_bytes=0, stop_host_bytes=0, sample_seconds=.1,
                            phase_seconds=5, total_seconds=10, grace_seconds=.3, heartbeat_seconds=.1)
            settings.update(limits or {})
            began = time.monotonic()
            result = supervise([sys.executable, '-u', '-c', script], root/'worker', settings,
                               ledger=root/'ledger.jsonl', phase='test', console=stream or io.StringIO())
            elapsed = time.monotonic()-began
            saved = (root/'worker/stdout.log').read_text()
            record = json.loads((root/'worker/result.json').read_text())
            self.assertFalse(record['surviving_owned_pids'])
            self.assertFalse((root/'ledger.active.json').exists())
            self.assertEqual(len(ledger_entries(root/'ledger.jsonl')), 1)
            self.assertGreaterEqual(record['elapsed_seconds'], 0)
            return result, saved, elapsed, record

    def test_live_steps_saved_and_forwarded(self):
        stream = io.StringIO()
        result, saved, _, _ = self.run_worker("import time\nprint('step 001/192', flush=True)\ntime.sleep(.1)\nprint('step 002/192', flush=True)", stream=stream)
        self.assertEqual(result, 'EXITED')
        self.assertIn('step 001/192', saved)
        self.assertIn('step 002/192', stream.getvalue())

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
        for phase in ('training', 'compilation', 'saving'):
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
