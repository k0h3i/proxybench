"""CPU behavior contract for the isolated Trainer candidate."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from proxybench.training import trainer_experiment as candidate
from proxybench.training.trajectory import (assert_same, load_state, publish_state, random_state,
    require_clean_stop, restore_loaded_state, sample_order, train_updates)


class TrainerExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.torch, cls.np = torch, np

    def build(self):
        torch, np = self.torch, self.np
        torch.manual_seed(19)
        random.seed(23)
        np.random.seed(29)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=0)

        def loss(index):
            x = torch.tensor([[index, random.random(), float(np.random.random())]], dtype=torch.float32)
            return (model(x) - .5).square().mean(), index + 3

        return model, optimizer, loss

    def test_exact_custom_loop_values_order_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, order = Path(tmp), sample_order(8, 2)
            custom, custom_opt, loss = self.build()
            expected = train_updates(custom, custom_opt, order, loss, root / 'custom.json', report=None)
            expected_random = random_state()
            model, optimizer, loss = self.build()
            actual = candidate.train_updates(model, optimizer, order, loss, root / 'trainer.json', report=None)
            assert_same(custom.state_dict(), model.state_dict())
            assert_same(custom_opt.state_dict(), optimizer.state_dict())
            assert_same(expected_random, random_state())
            self.assertEqual([r['loss'] for r in expected['history']], [r['loss'] for r in actual['history']])
            self.assertEqual([r['index'] for r in actual['history']], order)
            self.assertEqual(actual['status'], 'TRAINED')
            self.assertTrue((root / 'trainer-experiment.json').exists())
            self.assertFalse(any((root / 'trainer-internal').glob('checkpoint-*')))

    def test_exact_clean_stop_resume_after_trainer_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, order = Path(tmp), sample_order(8, 2)
            full, full_opt, loss = self.build()
            expected = candidate.train_updates(full, full_opt, order, loss, root / 'full.json', report=None)
            model, optimizer, loss = self.build()
            checkpoint = root / 'checkpoint'

            def save(step, history, clean):
                self.assertTrue(clean)

                def consume_random(path):
                    random.random()
                    self.np.random.random()
                    self.torch.rand(3)

                return publish_state(model, optimizer, None, checkpoint, identity={'run': 1}, order=order,
                    completed=step, history=history, save_adapter=consume_random,
                    diagnostics={'trainer_experiment': {'scheduler': candidate.SCHEDULER}})

            result = candidate.train_updates(model, optimizer, order, loss, root / 'part.json',
                report=None, save=save, max_updates=7)
            self.assertEqual(result['completed'], 7)
            journal = json.loads((root / 'part.json').read_text())
            self.assertEqual(require_clean_stop(journal, checkpoint, {'run': 1}), 7)
            saved = load_state(checkpoint, identity={'run': 1}, order=order, completed=7)
            resumed, resumed_opt, loss = self.build()
            random.random()
            self.np.random.random()
            self.torch.rand(4)
            restore_loaded_state(resumed, resumed_opt, saved, order=order, completed=7)
            result = candidate.train_updates(resumed, resumed_opt, order, loss, root / 'part.json',
                report=None, start=7, history=saved['history'])
            assert_same(full.state_dict(), resumed.state_dict())
            assert_same(full_opt.state_dict(), resumed_opt.state_dict())
            self.assertEqual([r['loss'] for r in expected['history']], [r['loss'] for r in result['history']])

    def test_nonfinite_loss_and_gradient_fail_before_optimizer(self):
        torch = self.torch
        for gradient in (False, True):
            with self.subTest(gradient=gradient), tempfile.TemporaryDirectory() as tmp:
                model, optimizer, loss = self.build()
                initial = copy.deepcopy(model.state_dict())
                original = copy.deepcopy(optimizer.state_dict())
                if gradient:
                    next(model.parameters()).register_hook(lambda grad: grad * float('nan'))
                    bad_loss = loss
                else:
                    bad_loss = lambda index: (torch.tensor(float('nan'), requires_grad=True), 3)
                journal = Path(tmp) / 'journal.json'
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    candidate.train_updates(model, optimizer, [0], bad_loss, journal, report=None)
                assert_same(initial, model.state_dict())
                assert_same(original, optimizer.state_dict())
                self.assertEqual(json.loads(journal.read_text()),
                                 dict(status='UPDATING', completed=0, pending_step=1))

    def test_save_cadence_matches_custom_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            for function in (train_updates, candidate.train_updates):
                model, optimizer, loss = self.build()
                saves = []
                function(model, optimizer, list(range(8)), loss, Path(tmp) / 'journal.json', report=None,
                    checkpoint_interval=3, checkpoint_steps={4, 8},
                    save=lambda step, history, clean: saves.append((step, clean)))
                self.assertEqual(saves, [(3, False), (4, False), (6, False), (8, False)])

    def test_stop_before_first_update_keeps_rng_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, optimizer, loss = self.build()
            initial = copy.deepcopy(model.state_dict())
            rng = random_state()
            root = Path(tmp)

            def save(step, history, clean):
                return publish_state(model, optimizer, None, root / 'checkpoint', identity={},
                                     order=[0], completed=step, history=history)

            result = candidate.train_updates(model, optimizer, [0], loss, root / 'journal.json',
                                            stop=lambda: True, save=save, report=None)
            self.assertEqual(result['completed'], 0)
            assert_same(initial, model.state_dict())
            assert_same(rng, random_state())

    def test_failed_save_cannot_publish_clean_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, optimizer, loss = self.build()
            journal = Path(tmp) / 'journal.json'

            def fail(*args):
                raise RuntimeError('failed save')

            with self.assertRaisesRegex(RuntimeError, 'failed save'):
                candidate.train_updates(model, optimizer, [0, 1], loss, journal, max_updates=1,
                                        save=fail, report=None)
            self.assertEqual(json.loads(journal.read_text())['status'], 'BOUNDARY')

    def test_changed_effective_trainer_refuses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, optimizer, loss = self.build()
            root = Path(tmp)
            (root / 'trainer-experiment.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Effective Trainer'):
                candidate.train_updates(model, optimizer, [0, 1], loss, root / 'journal.json',
                                        start=1, history=[{}], report=None)
            self.assertFalse((root / 'journal.json').exists())

    def test_scheduler_never_changes_optimizer_state(self):
        model, optimizer, loss = self.build()
        before = copy.deepcopy(optimizer.state_dict())
        schedule = candidate.ConstantSchedule(optimizer)
        for _ in range(7):
            schedule.step()
        assert_same(before, optimizer.state_dict())
        schedule.load_state_dict(schedule.state_dict())
        self.assertEqual(schedule.get_last_lr(), [.0001])
        with self.assertRaises(ValueError):
            schedule.load_state_dict({'kind': 'linear'})

    def test_clean_stop_measurement_contains_last_update_and_save(self):
        from proxybench.training.measurements import MeasurementRecorder
        now, events = [0.0], []
        recorder = MeasurementRecorder(clock=lambda: now[0], emit=events.append)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, optimizer, loss = self.build()

            def save(step, history, clean):
                now[0] += 7
                return publish_state(model, optimizer, None, root / 'checkpoint', identity={},
                                     order=[0, 1], completed=step, history=history)

            result = candidate.train_updates(model, optimizer, [0, 1], loss, root / 'journal.json',
                max_updates=1, save=save, report=None, measurements=recorder,
                sequence_tokens=lambda index: 12, epoch_boundaries=[2])
            self.assertEqual(result['completed'], 1)
            self.assertEqual(recorder.snapshot()['full_loop']['nonpadding_tokens'], 12)
            update = [event for event in events if event['kind'] == 'update'][0]
            self.assertEqual(update['full_loop_seconds'], 7)
            self.assertEqual(update['last_checkpoint']['global_step'], 1)
            self.assertEqual(update['learning_rate'], .0001)
            self.assertEqual(update['weighted_loss_12'], result['history'][0]['loss'])
            self.assertFalse(update['resume_eligible'])
            self.assertIsNone(update['remaining_loop_seconds'])

    def test_device_profiling_rejects_before_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, optimizer, loss = self.build()
            journal = Path(tmp) / 'journal.json'
            with self.assertRaisesRegex(ValueError, 'profiling is not supported'):
                candidate.train_updates(model, optimizer, [0], loss, journal, compute_timer=object())
            self.assertFalse(journal.exists())

    def test_epoch_source_drift_rejects_before_journal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(candidate, 'EPOCH_SOURCE_SHA256', 'changed'):
            model, optimizer, loss = self.build()
            journal = Path(tmp) / 'journal.json'
            with self.assertRaisesRegex(ValueError, 'epoch implementation'):
                candidate.train_updates(model, optimizer, [0], loss, journal)
            self.assertFalse(journal.exists())

    def test_launcher_preserves_supervisor_limits_and_worker_module(self):
        from proxybench.execution import live
        config = {'training_loop': candidate.EXPERIMENT,
                  'limits': dict(total_seconds=14400, phase_seconds=6000, grace_seconds=30)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(live, 'supervise', return_value='EXITED') as supervisor:
            result = candidate.launch('train', tmp, config, max_updates=3, seconds=900,
                resource_ceiling=700, dataset='dataset', resume=False)
            self.assertEqual(result, 'EXITED')
            command, output, limits = supervisor.call_args.args
            self.assertEqual(command[2:4], ['proxybench.training.trainer_experiment', '_work'])
            self.assertEqual(limits['total_seconds'], 700)
            self.assertEqual(limits['phase_seconds'], 900)
            self.assertEqual(limits['grace_seconds'], 30)
            self.assertEqual(supervisor.call_args.kwargs['phase'], 'train')
            request = json.loads(Path(command[-1]).read_text())
            self.assertEqual(request['max_updates'], 3)
            self.assertFalse(request['resume'])

    def test_preflight_rejects_cross_engine_before_dataset_or_writes(self):
        from proxybench.training.resume import training_preflight
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.training.dataset.read_release') as reader:
            for marker, requested in ((candidate.EXPERIMENT, 'custom'), ('custom', candidate.EXPERIMENT)):
                with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, 'Training loop differs'):
                    training_preflight('missing', Path(tmp) / 'run', {'training_loop': marker},
                                       training_loop=requested, resume=True)
            reader.assert_not_called()
            self.assertFalse((Path(tmp) / 'run').exists())

    def test_custom_worker_rejects_experiment_before_initialization(self):
        from proxybench.training.runtime import train
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.training.dataset.read_release') as reader:
            with self.assertRaisesRegex(ValueError, 'Training loop differs'):
                train('missing', Path(tmp) / 'run', {'training_loop': candidate.EXPERIMENT})
            reader.assert_not_called()
            self.assertFalse((Path(tmp) / 'run').exists())

    def test_worker_retains_checkpoint_diagnostics_and_restores_patch(self):
        from proxybench.training import runtime, trajectory
        original_updates, original_publish = trajectory.train_updates, trajectory.publish_state

        def run(*args, **kwargs):
            self.assertEqual(kwargs['training_loop'], candidate.EXPERIMENT)
            self.assertEqual(trajectory.train_updates._proxybench_training_loop, candidate.EXPERIMENT)
            trajectory.publish_state('model', diagnostics={'runtime_identity': 123}, measurements='measurement')
            raise RuntimeError('worker failed')

        with patch.object(runtime, 'train', side_effect=run), patch.object(trajectory, 'publish_state') as publish:
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                candidate.worker(dict(dataset='data', run_dir='run',
                    config={'training_loop': candidate.EXPERIMENT}, resume=False, max_updates=2))
            self.assertEqual(publish.call_args.kwargs['diagnostics']['runtime_identity'], 123)
            self.assertEqual(publish.call_args.kwargs['diagnostics']['trainer_experiment']['scheduler'], candidate.SCHEDULER)
            self.assertEqual(publish.call_args.kwargs['measurements'], 'measurement')
        self.assertIs(trajectory.train_updates, original_updates)
        self.assertIs(trajectory.publish_state, original_publish)


if __name__ == '__main__':
    unittest.main()
