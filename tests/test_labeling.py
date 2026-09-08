"""Low-effort capture and bounded labeling without reference dependencies."""
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from proxybench.execution import labeling as l
from proxybench.annotation.xml_packets import build
from test_training_preparation import PRIMARY, votes
from test_comparison import transcript, put_transcript, save


class LabelingTests(unittest.TestCase):
    def test_low_effort_and_substitution_native_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = transcript(effort='low')
            put_transcript(d, rows)
            self.assertEqual(l.sol_provenance(d, b'synthetic', b'{}', True)['effort'], 'low')
            for changed in (transcript(effort='medium'), transcript(effort='low', tool=True)):
                put_transcript(d, changed)
                with self.assertRaises(l.IntegrityError):
                    l.sol_provenance(d, b'synthetic', b'{}', True)
            put_transcript(d, rows)
            with self.assertRaises(l.IntegrityError):
                l.sol_provenance(d, b'synthetic', b'changed', True)

    def test_reserve_includes_cleanup(self):
        with patch.object(l.time, 'monotonic', return_value=100):
            self.assertFalse(l.enough_time(444))
            self.assertTrue(l.enough_time(445))

    def frozen(self, root):
        bundle = build(PRIMARY, votes(), primary_id='p', votes_id='v', index=0, input_id='a')['bundle']
        save(root / 'models.json', {})
        run = root / 'run'
        l.freeze(run, [json.dumps(bundle).encode()], contract=b'contract', cli=sys.executable,
                 model_cache=root/'models.json', authorization='synthetic test')
        return run

    def test_no_references_required_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.frozen(root)
            self.assertEqual(json.loads((run/'manifest.json').read_bytes())['effort'], 'low')
            self.assertNotIn('reference', (run/'schedule.json').read_text())
            with self.assertRaises(l.IntegrityError):
                l.freeze(run, [b'{}'], contract=b'', cli=sys.executable,
                         model_cache=root/'models.json', authorization='synthetic')

    def test_failed_probe_leaves_target_unstarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.frozen(Path(tmp))
            with patch.object(l, 'isolation_probe'), patch.object(l, 'attempt', return_value={
                    'execution_eligible': False, 'status': 'INTEGRITY_STOP'}) as attempt, redirect_stdout(io.StringIO()):
                report = l.execute(run, auth=Path(tmp)/'absent-auth')
            self.assertEqual(attempt.call_count, 1)
            self.assertEqual(report['scheduled'], 1)
            self.assertEqual(report['results'][0]['status'], 'NOT_ATTEMPTED_STOP')
            self.assertEqual(report['training_admitted'], 0)

    def test_native_cancellation_retains_partial_and_fits_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            state = d/'state'
            state.mkdir()
            save(d/'input', b'')
            code = 'import time; from pathlib import Path; Path(' + repr(str(state/'final.txt')) + ').write_text("partial"); time.sleep(10)'
            result = l.capture([sys.executable, '-c', code], d/'input', d, seconds=.2,
                               global_deadline=time.monotonic()+3, limit=4096, state=state)
            self.assertTrue(result['termination_established'])
            self.assertEqual(result['outcome'], 'TIMEOUT')
            self.assertLess(result['cleanup_seconds'], l.LIMITS['cleanup_reserve'])
            self.assertFalse((d/'raw.bin').exists())
            self.assertEqual((d/'late-final.bin').read_bytes(), b'partial')


if __name__ == '__main__':
    unittest.main()
