"""CPU measurements with a simulated clock and no device imports."""

import builtins
from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.training.measurements import (
    MeasurementRecorder, capture_runtime_identity, compare_runtime_identity, epoch_progress,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.recorder = MeasurementRecorder(clock=self.clock)

    def update(self, step, *, seconds=5):
        return self.recorder.record_update(global_step=step, sample_position=step,
            nonpadding_tokens=100, supervised_tokens=25,
            full_loop_seconds=seconds, compute_seconds=seconds - 1, loss=2.0)

    def test_nested_stages_and_failed_attempt_keep_elapsed_cost(self):
        with self.recorder.stage('preparation'):
            self.clock.advance(3)
        with self.assertRaisesRegex(RuntimeError, 'publication failed'):
            with self.recorder.stage('final_publication'):
                with self.recorder.stage('checkpoint_hashing'):
                    self.clock.advance(4)
                self.clock.advance(2)
                raise RuntimeError('publication failed')
        result = self.recorder.finish('FAILED')
        self.assertEqual(result['attempt_elapsed_seconds'], 9)
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(result['stages']['final_publication'], dict(seconds=6, calls=1, failed_calls=1))
        self.assertEqual(result['stages']['checkpoint_hashing']['seconds'], 4)
        self.assertEqual(result['stage_accounting'], 'inclusive_nested')
        self.clock.advance(10)
        self.assertEqual(self.recorder.elapsed(), 9)

    def test_first_update_and_startup_remain_in_attempt_totals(self):
        with self.recorder.stage('model_loading'):
            self.clock.advance(40)
        self.assertIsNone(self.recorder.remaining_loop_seconds(10))
        self.clock.advance(20)
        self.assertEqual(self.update(1, seconds=20)['update_period'], 'first_update')
        self.clock.advance(5)
        self.assertEqual(self.update(2, seconds=5)['update_period'], 'steady_state')
        summary = self.recorder.snapshot()
        self.assertEqual(summary['attempt_elapsed_seconds'], 65)
        self.assertEqual(summary['full_loop']['seconds'], 25)
        self.assertEqual(summary['first_update']['seconds'], 20)
        self.assertEqual(summary['steady_state']['seconds'], 5)
        self.assertEqual(self.recorder.remaining_loop_seconds(10), 40)

    def test_resume_has_new_first_update_and_does_not_count_old_tokens(self):
        self.clock.advance(5)
        result = self.update(397)
        self.assertEqual(result['attempt_elapsed_seconds'], 5)
        self.assertEqual(result['global_step'], 397)
        self.assertEqual(result['throughput']['updates'], 1)
        self.assertEqual(result['update_period'], 'first_update')
        self.assertEqual(self.recorder.remaining_loop_seconds(660), 263 * 5)

    def test_zero_duration_does_not_fabricate_rate_or_eta(self):
        result = self.recorder.record_update(global_step=1, sample_position=1,
            nonpadding_tokens=100, supervised_tokens=20, full_loop_seconds=0)
        self.assertIsNone(result['throughput']['nonpadding_tokens_per_second'])
        self.assertIsNone(self.recorder.remaining_loop_seconds(10))

    def test_invalid_measurements_fail_before_entering_window(self):
        base = dict(global_step=1, sample_position=1, nonpadding_tokens=100,
                    supervised_tokens=20, full_loop_seconds=5)
        for change in (dict(global_step=True), dict(nonpadding_tokens=-1),
                       dict(supervised_tokens=101), dict(full_loop_seconds=float('nan')),
                       dict(compute_seconds=-1), dict(compute_timing='guessed')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.recorder.record_update(**(base | change))
        self.assertEqual(self.recorder.snapshot()['full_loop']['updates'], 0)
        self.update(2)
        with self.assertRaises(ValueError):
            self.update(2)

    def test_memory_peaks_require_observed_reset_and_keep_units(self):
        values = dict(allocated_bytes=12, reserved_bytes=20, peak_allocated_bytes=18,
                      peak_reserved_bytes=24, free_device_bytes=100)
        with self.assertRaises(ValueError):
            self.recorder.memory_sample(**values)
        resets = []
        self.recorder.begin_memory_window('steady_state', reset=lambda: resets.append(True), global_step=397)
        self.clock.advance(3)
        result = self.recorder.memory_sample(**values)
        self.assertEqual(resets, [True])
        self.assertEqual(result['peak_window'], dict(label='steady_state', peak_reset_step=397,
                                                    peak_reset_elapsed_seconds=0))
        self.assertEqual(result['peak_reserved_bytes'], 24)
        with self.assertRaises(ValueError):
            self.recorder.memory_sample(**(values | dict(peak_allocated_bytes=1)))

    def test_failed_peak_reset_does_not_create_new_window(self):
        def fail():
            raise RuntimeError('reset failed')
        with self.assertRaises(RuntimeError):
            self.recorder.begin_memory_window('attempt', reset=fail, global_step=0)
        self.assertIsNone(self.recorder.memory_window)

    def test_epoch_boundaries_use_sample_positions(self):
        for position, epoch, count, fraction in (
                (0, 1, 0, 0), (329, 1, 329, 329/330), (330, 1, 330, 1),
                (331, 2, 1, 1+1/330), (659, 2, 329, 1+329/330), (660, 2, 330, 2)):
            with self.subTest(position=position):
                row = epoch_progress(position, [330, 660])
                self.assertEqual((row['epoch'], row['examples_completed']), (epoch, count))
                self.assertEqual(row['fractional_epoch'], fraction)
        self.assertEqual(epoch_progress(12, [10, 14])['fractional_epoch'], 1.5)
        for position, boundaries in ((True, [10]), (11, [10]), (0, []), (0, [0]), (2, [10, 10])):
            with self.assertRaises(ValueError):
                epoch_progress(position, boundaries)


class RuntimeIdentityTests(unittest.TestCase):
    def capture(self, root, **changes):
        arguments = dict(code_root=root, model=dict(id='synthetic', revision='one', files={'weights': 'aaa'}),
                         tokenizer={'files': {'tokenizer.json': 'bbb'}},
                         optimizer={'class': 'AdamW', 'fused': None, 'foreach': None, 'implementation': 'foreach'},
                         hardware={'device': 'CPU', 'driver': 'not_applicable'},
                         kernels={'loss': 'reference', 'deterministic': False},
                         package_names=['synthetic'], package_version=lambda name: '1.0')
        return capture_runtime_identity(**(arguments | changes))

    def test_capture_never_imports_device_packages_or_exposes_source_root(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'unsloth', 'triton', 'transformers'}:
                raise AssertionError('GPU package imported')
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'worker.py').write_text('value = 1\n')
            with patch('builtins.__import__', guarded):
                result = self.capture(tmp)
            self.assertNotIn(tmp, json.dumps(result))
            self.assertEqual(list(result['code']), ['worker.py'])
            comparison = compare_runtime_identity(result, result)
            self.assertEqual(comparison['status'], 'MATCH')
            self.assertFalse(comparison['exact_continuation_claim'])

    def test_each_runtime_domain_change_requires_new_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'worker.py'
            path.write_text('value = 1\n')
            previous = self.capture(tmp)
            for field in ('packages', 'model', 'tokenizer', 'optimizer', 'hardware', 'kernels'):
                current = previous | {field: {'changed': 'new'}}
                result = compare_runtime_identity(previous, current)
                self.assertEqual(result['changed'], [field])
                self.assertEqual(result['status'], 'CHANGED')
                self.assertTrue(result['requires_acceptance'])
            path.write_text('value = 2\n')
            self.assertEqual(compare_runtime_identity(previous, self.capture(tmp))['changed'], ['code'])

    def test_missing_old_or_unknown_identity_never_becomes_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'worker.py').write_text('value = 1\n')
            current = self.capture(tmp)
            for previous in (None, {}, {'dataset': 'old'}, current | {'hardware': None},
                             current | {'hardware': {'device': 'CPU', 'driver': None}},
                             current | {'optimizer': {'implementation': None}},
                             current | {'kernels': {'loss': 'unknown'}}):
                result = compare_runtime_identity(previous, current)
                self.assertEqual(result['status'], 'UNVERIFIED')
                self.assertTrue(result['requires_acceptance'])
                self.assertFalse(result['exact_continuation_claim'])

    def test_missing_packages_are_explicit_and_cannot_match_as_complete(self):
        def missing(name):
            raise metadata.PackageNotFoundError(name)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'worker.py').write_text('value = 1\n')
            result = self.capture(tmp, package_version=missing)
            self.assertIsNone(result['packages']['synthetic'])
            self.assertEqual(compare_runtime_identity(result, result)['status'], 'UNVERIFIED')
