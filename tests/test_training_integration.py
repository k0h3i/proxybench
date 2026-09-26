"""Loop measurements include durable writes, monitoring, reports, and saves."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from proxybench.training.measurements import MeasurementRecorder
from proxybench.training.trajectory import train_updates


class TrainingMeasurementIntegrationTests(unittest.TestCase):
    def test_authoritative_loop_rate_includes_event_emission_and_boundary_stop(self):
        now = [0.0]
        def emit(event):
            now[0] += 1
        recorder = MeasurementRecorder(clock=lambda: now[0], emit=emit)
        with recorder.stage('full_loop'):
            now[0] += 5
            recorder.record_update(global_step=1, sample_position=1, nonpadding_tokens=10,
                                   supervised_tokens=2, full_loop_seconds=5)
            # A stop before the next update still incurs checkpoint and journal work.
            with recorder.stage('checkpoint'):
                now[0] += 7
        full = recorder.snapshot()['full_loop']
        self.assertEqual(full['seconds'], 14)
        self.assertEqual(full['timing'], 'inclusive_loop_host_wall')
        self.assertAlmostEqual(full['nonpadding_tokens_per_second'], 10 / 14)

    def test_resume_rejects_optimizer_settings_before_mutating_model(self):
        from proxybench.training.trajectory import publish_state, load_state, restore_loaded_state, assert_same
        with tempfile.TemporaryDirectory() as folder:
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
            checkpoint = publish_state(model, optimizer, None, Path(folder) / 'checkpoint',
                                       identity={}, order=[0], completed=0, history=[])
            saved = load_state(checkpoint, identity={}, order=[0], completed=0)
            saved['optimizer']['param_groups'][0]['lr'] = 0.1
            before = {name: value.clone() for name, value in model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, 'optimizer settings differ'):
                restore_loaded_state(model, optimizer, saved, order=[0], completed=0)
            assert_same(before, model.state_dict())
            self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)

    def test_matching_windows_include_checkpoint_and_reporting_cost(self):
        now, events = [0.0], []
        recorder = MeasurementRecorder(clock=lambda: now[0], emit=events.append)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

        def loss(index):
            now[0] += 3
            return model(torch.ones(1, 1)).square().mean(), 2

        def journal(path, value):
            now[0] += 1

        def memory():
            now[0] += 2
            return {}

        def report(line):
            now[0] += 4

        def save(step, history, clean):
            now[0] += 8
            return Path('/synthetic/checkpoint')

        with patch('proxybench.training.trajectory.durable_json', side_effect=journal):
            result = train_updates(model, optimizer, [0, 1], loss, Path('/synthetic/journal'),
                memory=memory, report=report, save=save, checkpoint_interval=2,
                measurements=recorder, sequence_tokens=lambda index: 7, epoch_boundaries=[1, 2])
        updates = [row for row in events if row['kind'] == 'update']
        self.assertEqual([row['full_loop_seconds'] for row in updates], [11, 19])
        self.assertEqual(updates[-1]['throughput']['seconds'], 30)
        self.assertEqual(updates[-1]['throughput']['nonpadding_tokens'], 14)
        self.assertEqual(updates[-1]['throughput']['supervised_tokens'], 4)
        self.assertAlmostEqual(updates[-1]['throughput']['nonpadding_tokens_per_second'], 14 / 30)
        self.assertEqual(updates[-1]['last_checkpoint']['global_step'], 2)
        self.assertEqual(updates[-1]['fractional_epoch'], 2)
        self.assertIsNone(updates[-1]['remaining_loop_seconds'])
        self.assertEqual(recorder.snapshot()['stages']['journal']['calls'], 6)
        self.assertEqual(recorder.snapshot()['stages']['monitor']['seconds'], 4)
        self.assertEqual(result['completed'], 2)

    def test_instrumentation_keeps_model_updates_and_checkpoint_cadence(self):
        def run(root, name, instrumented):
            torch.manual_seed(42)
            model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Dropout(0.2), torch.nn.Linear(3, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
            saved = []
            result = train_updates(model, optimizer, [0, 1, 2, 3],
                lambda index: (model(torch.tensor([[float(index)]])).square().mean(), 2),
                root / name, report=None, save=lambda step, *args: saved.append(step), checkpoint_interval=3,
                measurements=MeasurementRecorder() if instrumented else None, sequence_tokens=lambda _: 6)
            return model, optimizer, result, saved

        with tempfile.TemporaryDirectory() as folder:
            plain = run(Path(folder), 'plain', False)
            measured = run(Path(folder), 'measured', True)
        from proxybench.training.trajectory import assert_same
        assert_same(plain[0].state_dict(), measured[0].state_dict())
        assert_same(plain[1].state_dict(), measured[1].state_dict())
        self.assertEqual([row['loss'] for row in plain[2]['history']], [row['loss'] for row in measured[2]['history']])
        self.assertEqual(plain[3], measured[3])

    def test_worker_failure_records_attempt_without_claiming_publication(self):
        from proxybench.training.runtime import train
        with tempfile.TemporaryDirectory() as folder:
            capture = Path(folder) / 'phase.json'
            with patch.dict('os.environ', PROXYBENCH_PHASE_FILE=str(capture)), \
                    patch('proxybench.training.runtime._train', side_effect=RuntimeError('publication failed')), \
                    patch('builtins.print') as output:
                with self.assertRaisesRegex(RuntimeError, 'publication failed'):
                    train('synthetic', folder, {})
            result = json.loads((capture.parent / 'training-measurements.json').read_text())
            self.assertEqual(result['status'], 'FAILED')
            self.assertNotIn('result_published', output.call_args.args[0])
            self.assertFalse((capture.parent / 'training-result.json').exists())

    def test_device_event_read_follows_existing_safety_monitor(self):
        calls, events = [], []
        class Timer:
            def start(self):
                calls.append('start')
            def stop(self):
                calls.append('stop')
            def seconds(self):
                calls.append('read')
                return 0.125
        def monitor():
            calls.append('monitor')
            return {}
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as folder:
            train_updates(model, optimizer, [0], lambda _: (model(torch.ones(1, 1)).sum(), 1),
                          Path(folder) / 'journal', report=None, memory=monitor,
                          measurements=MeasurementRecorder(emit=events.append), compute_timer=Timer())
        self.assertEqual(calls, ['start', 'stop', 'monitor', 'read'])
        update = next(event for event in events if event['kind'] == 'update')
        self.assertEqual(update['compute_timing'], 'device_events')
        self.assertEqual(update['compute_seconds'], 0.125)
