"""Exercise compact operation displays with saved logs and a fake clock."""

import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from proxybench.execution.operation_display import OperationDisplay


class OperationDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.capture = Path(self.temp.name) / 'capture'
        self.capture.mkdir()
        self.now = 100.
        self.output = []
        self.display = self.make_display()

    def make_display(self, **kwargs):
        return OperationDisplay(self.output.append, operation=kwargs.pop('operation', 'evaluation'),
                                capture_path=self.capture, clock=lambda: self.now, **kwargs)

    def test_initial_loading_has_no_invented_metrics_or_success(self):
        self.display.render(immediate=True)
        self.assertEqual(self.output, ['Loading | elapsed 0m00s\n'])
        self.display.feed_stdout(b'Loading library banner\n', final=True)
        self.display.feed_stderr(b'Library warning\n')
        self.assertEqual(len(self.output), 1)

    def test_plain_output_reports_phase_changes_tens_and_final_total(self):
        self.display.observe(dict(phase='evaluation', completed=0, total=23, failed=0))
        self.output.clear()
        for completed in range(1, 24):
            self.now += 1
            self.display.observe(dict(phase='evaluation', completed=completed, total=23))
            self.display.tick()
        self.assertEqual(len(self.output), 3)
        for text, expected in zip(self.output, ('10/23', '20/23', '23/23')):
            self.assertIn(expected, text)
            self.assertEqual(len(text.splitlines()), 1)
        self.display.observe(dict(phase='cleanup'))
        self.assertIn('Cleanup', self.output[-1])
        self.display.observe(dict(phase='cleanup'))
        self.assertEqual(len(self.output), 4)
        self.assertNotIn('\x1b', ''.join(self.output))

    def test_plain_output_reports_crossed_threshold_when_polling_skips_counts(self):
        self.display.observe(dict(phase='evaluation', completed=9, total=30))
        self.output.clear()
        self.display.observe(dict(phase='evaluation', completed=12))
        self.assertEqual(len(self.output), 1)
        self.assertIn('12/30', self.output[0])

    def test_tty_refresh_limits_and_two_physical_rows(self):
        self.display.tty = True
        with patch('proxybench.execution.operation_display.shutil.get_terminal_size',
                   return_value=os.terminal_size((45, 24))):
            self.display.observe(dict(phase='evaluation', completed=0, total=25))
            for now in (100.1, 100.5, 100.99):
                self.now = now
                self.display.observe(dict(phase='evaluation', completed=1))
                self.display.tick()
            self.assertEqual(len(self.output), 1)
            self.now = 101.
            self.display.tick()
            self.assertEqual(len(self.output), 2)
            self.display.observe(dict(phase='cleanup'))
            self.assertEqual(len(self.output), 3)
        for frame in self.output:
            text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', frame).replace('\r', '')
            self.assertEqual(len(text.splitlines()), 2)
            self.assertTrue(all(len(line) < 45 for line in text.splitlines()))
        self.assertIn('[', self.output[0])
        self.assertIn('\x1b[1A', self.output[1])

    def test_restored_counts_offset_session_counts_and_budget(self):
        self.display = self.make_display(progress=dict(completed=17, total=30, failed=2),
                                         started=100, prior_seconds=25, phase_used=20,
                                         total_seconds=1000, phase_seconds=500)
        self.display.render(immediate=True)
        self.assertIn('Loading 17/30', self.output[-1])
        self.display.observe(dict(phase='evaluation', completed=0, total=13, failed=0))
        self.assertIn('17/30', self.output[-1])
        self.display.observe(dict(phase='evaluation', completed=3, total=13, failed=1))
        self.assertIn('20/30', self.output[-1])
        self.assertIn('generation failures 3', self.output[-1])
        self.now = 111
        rows = self.display.compact_rows()
        self.assertIn('elapsed 0m11s | budget 7m49s | charged 0m36s', rows[1])

    def test_known_initial_total_supports_partial_optional_counters(self):
        display = self.make_display(progress=dict(completed=17, total=30, failed=2))
        display.observe(dict(phase='evaluation', completed=3, failed=1))
        self.assertEqual(display.counts, (20, 30, 3))
        display.observe(dict(phase='evaluation', completed=4, total=30))
        self.assertEqual(display.counts, (20, 30, 3))
        display.observe(dict(phase='cleanup', completed=13, total=13))
        self.assertEqual(display.counts, (30, 30, 3))

    def test_bad_optional_metadata_cannot_replace_counters(self):
        self.display.observe(dict(phase='evaluation', completed=10, total=30, failed=2))
        expected = self.display.counts
        for fields in (dict(completed=True), dict(completed=-1), dict(completed=9),
                       dict(completed=31), dict(completed=10**500), dict(failed=11),
                       dict(failed=1), dict(total=31), dict(total=None), dict(total=[]),
                       dict(completed='20'), dict(failed={}), dict(completed=float('nan'))):
            with self.subTest(fields=fields):
                self.display.observe(dict(phase='evaluation', **fields))
                self.assertEqual(self.display.counts, expected)
        for state in (None, [], {}, dict(phase=[]), dict(phase=None), dict(phase='untrusted\nphase')):
            self.display.observe(state)
            self.assertEqual(self.display.phase, 'evaluation')
        self.display.observe(dict(phase='cleanup', completed='bad'))
        self.assertEqual(self.display.phase, 'cleanup')
        self.assertEqual(self.display.counts, expected)

    def test_invalid_initial_progress_is_ignored(self):
        for progress in (None, [], {}, dict(completed=1, total=0), dict(total=True)):
            display = self.make_display(progress=progress)
            display.render(immediate=True)
            self.assertEqual(display.counts, None)
            self.assertIn('Loading | elapsed', self.output[-1])

    def test_generation_success_never_claims_scoring_or_review_finished(self):
        self.display.observe(dict(phase='evaluation', completed=3, total=3, failed=1))
        self.assertNotIn('Generation finished', ''.join(self.output))
        self.display.finish('EXITED')
        self.assertEqual(self.output[-1], 'Generation finished. Scoring and review are pending.\n')
        self.assertIn('generation failures 1', ''.join(self.output))
        before = list(self.output)
        self.display.finish('EXITED')
        self.now += 10
        self.display.tick()
        self.display.observe(dict(phase='loading'))
        self.assertEqual(self.output, before)

    def test_partial_exit_and_failure_statuses_never_claim_success(self):
        for status in ('EXITED', 'PROCESS_FAILED', 'PHASE_TIMEOUT', 'USER_STOP', 'FORCED_STOP'):
            with self.subTest(status=status):
                self.output.clear()
                display = self.make_display()
                display.observe(dict(phase='evaluation', completed=2, total=3, failed=1))
                display.finish(status)
                text = ''.join(self.output)
                self.assertNotIn('Generation finished', text)
                self.assertIn('2/3', text)
                self.assertIn('Worker diagnostics:', text)
                self.assertIn(str(self.capture.parent / 'answers/server.log'), text)
                self.assertIn('incomplete' if status == 'EXITED' else status, text)

    def test_success_requires_known_counts(self):
        self.display.finish('EXITED')
        self.assertIn('Generation incomplete', ''.join(self.output))

    def test_export_and_conversion_phases_do_not_claim_publication(self):
        for operation in ('export', 'convert'):
            self.output.clear()
            display = self.make_display(operation=operation)
            display.render(immediate=True)
            for phase in ('serialization', 'conversion', 'payload-inspection', 'cleanup'):
                display.observe(dict(phase=phase))
            display.finish('EXITED')
            text = ''.join(self.output)
            self.assertIn(f'{operation.capitalize()} worker finished.', text)
            self.assertNotIn('published', text)
            self.assertNotIn('complete', text)
            self.assertNotIn('diagnostics', text)

    def test_adapter_loading_display_keeps_logs_quiet_and_reports_inference(self):
        display = self.make_display(operation='validate-adapter', phase_seconds=900)
        display.render(immediate=True)
        display.feed_stdout(b'Unsloth banner\nPhase: inference\n', final=True)
        display.feed_stderr(b'Compiler warning\n')
        display.observe(dict(phase='inference'))
        display.finish('EXITED')
        text = ''.join(self.output)
        self.assertIn('Inference | elapsed 0m00s | budget 15m00s', text)
        self.assertIn('Adapter loading test worker finished.', text)
        for hidden in ('Unsloth', 'Compiler warning', 'Phase: inference', 'Generation finished', 'diagnostics'):
            self.assertNotIn(hidden, text)

    def test_adapter_loading_failure_keeps_short_diagnostics(self):
        (self.capture / 'stderr.log').write_bytes(b'Old compiler warning\n' * 1000 + b'Final adapter error\n')
        display = self.make_display(operation='validate-adapter')
        display.finish('PROCESS_FAILED')
        text = ''.join(self.output)
        self.assertIn('Adapter loading test did not complete. Supervisor status: PROCESS_FAILED.', text)
        self.assertIn('Final adapter error', text)
        self.assertIn(f'Worker diagnostics: {self.capture / "stderr.log"}', text)
        self.assertLess(len(text), 9000)
        self.assertNotIn('worker finished', text)

    def test_message_clears_frame_before_supervisor_stop(self):
        self.display.tty = True
        self.display.render(immediate=True)
        self.display.message('Stop requested.\n')
        self.assertEqual(self.output[-1], '\r\x1b[2K\x1b[1A\r\x1b[2KStop requested.\n')
        self.display.message('Stopping owned processes.')
        self.assertEqual(self.output[-1], 'Stopping owned processes.\n')

    def test_large_failure_log_is_read_directly_and_bounded(self):
        path = self.capture / 'stderr.log'
        path.write_bytes(b'Not visible old diagnostic\n' * 10000
                         + b'\n'.join(f'failure line {number}'.encode() for number in range(20))
                         + b'\n\x1b[31mFinal failure\x1b[0m\n')
        self.display.feed_stderr(b'Unsaved chatter')
        self.display.finish('PROCESS_FAILED')
        text = ''.join(self.output)
        self.assertLess(len(text), 9000)
        self.assertNotIn('Not visible old diagnostic', text)
        self.assertNotIn('failure line 8\n', text)
        self.assertIn('failure line 9\n', text)
        self.assertIn('Final failure', text)
        self.assertNotIn('Unsaved chatter', text)
        self.assertNotIn('\x1b', text)
        self.assertIn(str(path), text)

    def test_single_huge_diagnostic_is_bounded(self):
        (self.capture / 'stderr.log').write_bytes(b'x' * 100000 + b'final error')
        self.display.finish('PROCESS_FAILED')
        text = ''.join(self.output)
        self.assertLess(len(text), 9000)
        self.assertIn('final error', text)


if __name__ == '__main__':
    unittest.main()
