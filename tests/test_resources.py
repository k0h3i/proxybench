"""Device limits refuse launch and stop the owned worker during execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from proxybench.execution.resources import device_memory, supervise


class DeviceLimitTests(unittest.TestCase):
    def test_activity_fields_preserve_memory_and_unknown_readings(self):
        from types import SimpleNamespace
        result = SimpleNamespace(returncode=0, stdout='24576, 1024, 23552, 35, 24, 61, 105.17, [N/A]', stderr='')
        with patch('proxybench.execution.resources.subprocess.run', return_value=result):
            sample = device_memory()
        self.assertEqual(sample['free_bytes'], 23552 * 1024**2)
        self.assertEqual(sample['gpu_utilization_percent'], 35)
        self.assertIsNone(sample['sm_clock_mhz'])

    def test_memory_preflight_and_live_stop(self):
        limits = dict(start_host_bytes=4, stop_host_bytes=1, device_margin_bytes=2,
                      phase_seconds=10, total_seconds=20)
        for free, available, settings, expected in (
                ([1], [10], {}, 'PREFLIGHT_REFUSED'),
                ([3, 1], [10, 10], {}, 'DEVICE_MEMORY_LIMIT'),
                ([3, 3], [10, 9], dict(stop_host_bytes=10, stop_host_samples=1), 'HOST_MEMORY_LIMIT')):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                samples = [dict(total_bytes=10, used_bytes=10-n, free_bytes=n) for n in free]
                host = [dict(available_bytes=value) for value in available]
                with patch('proxybench.execution.resources.host_memory', side_effect=host), \
                     patch('proxybench.execution.resources.device_memory', side_effect=samples):
                    status = supervise([sys.executable, '-c', 'import time; time.sleep(30)'],
                                       root / 'run', {**limits, **settings}, ledger=root / 'ledger.jsonl')
                self.assertEqual(status, expected)
                result = json.loads((root / 'run/result.json').read_text())
                self.assertEqual(result['status'], expected)
                if expected == 'PREFLIGHT_REFUSED':
                    self.assertFalse((root / 'ledger.jsonl').exists())
                    self.assertFalse((root / 'run/stdout.log').exists())
                else:
                    self.assertIsNotNone(result['returncode'])
                    self.assertEqual(result['memory_margin_pass'], expected == 'HOST_MEMORY_LIMIT')
                    self.assertFalse((root / 'ledger.lock').exists())
                    self.assertFalse(result['surviving_owned_pids'])


if __name__ == '__main__':
    unittest.main()
