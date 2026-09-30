"""Exercise compact operation displays with saved logs and a fake clock."""

from pathlib import Path
import tempfile
import unittest

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

    def test_success_requires_known_counts(self):
        self.display.finish('EXITED')
        self.assertIn('Generation incomplete', ''.join(self.output))

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
