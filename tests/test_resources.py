"""Device limits refuse launch and stop the owned worker during execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from proxybench.execution.resources import limit_reason, supervise


class DeviceLimitTests(unittest.TestCase):
    def test_configured_host_limit_stops_on_first_sample(self):
        limits = dict(stop_host_bytes=10, stop_host_samples=1, phase_seconds=30, total_seconds=60)
        self.assertEqual(limit_reason(0, 9, 1, 1, limits), (1, 'HOST_MEMORY_LIMIT'))

    def test_device_preflight_and_live_stop(self):
        limits = dict(start_host_bytes=4, stop_host_bytes=1, device_margin_bytes=2,
                      phase_seconds=10, total_seconds=20)
        host = {'available_bytes': 10}
        for free, expected in [([1], 'PREFLIGHT_REFUSED'), ([3, 1], 'DEVICE_MEMORY_LIMIT')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                samples = [dict(total_bytes=10, used_bytes=10-n, free_bytes=n) for n in free]
                with patch('proxybench.execution.resources.host_memory', return_value=host), \
                     patch('proxybench.execution.resources.device_memory', side_effect=samples):
                    status = supervise([sys.executable, '-c', 'import time; time.sleep(30)'],
                                       root / 'run', limits, ledger=root / 'ledger.jsonl')
                self.assertEqual(status, expected)
                result = json.loads((root / 'run/result.json').read_text())
                self.assertEqual(result['status'], expected)
                if expected == 'PREFLIGHT_REFUSED':
                    self.assertFalse((root / 'ledger.jsonl').exists())
                    self.assertFalse((root / 'run/stdout.log').exists())
                else:
                    self.assertIsNotNone(result['returncode'])
                    self.assertFalse(result['memory_margin_pass'])
                    self.assertFalse((root / 'ledger.lock').exists())


if __name__ == '__main__':
    unittest.main()
