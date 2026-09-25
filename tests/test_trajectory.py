"""Exact CPU training resume, state corruption, and safe update boundaries."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from proxybench.training.trajectory import (assert_same, publish_state, require_clean_stop,
                                           restore_state, sample_order, train_updates, weighted_loss)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest('Requires the separate training environment, with CUDA disabled')
        self.torch, self.np = torch, np

    def build(self):
        torch, np = self.torch, self.np
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=0)
        def loss(index):
            x = torch.tensor([[index, random.random(), float(np.random.random())]], dtype=torch.float32)
            return (model(x)-.5).square().mean(), index+3
        return model, optimizer, loss

    def test_uninterrupted_and_resumed_training_are_exact(self):
        order = sample_order(8, 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full, full_opt, loss = self.build()
            expected = train_updates(full, full_opt, order, loss, root/'full.json', report=lambda _: None)
            part, opt, loss = self.build()
            checkpoint = root/'checkpoint'
            def save(step, history, clean):
                def consume_random(path):
                    random.random()
                    self.np.random.random()
                    self.torch.rand(3)
                return publish_state(part, opt, None, checkpoint, identity={'run': 'one'}, order=order,
                                     completed=step, history=history, save_adapter=consume_random)
            result = train_updates(part, opt, order, loss, root/'part.json',
                       stop=lambda: json.loads((root/'part.json').read_text())['completed'] == 7,
                       save=save, report=lambda _: None)
            journal = json.loads((root/'part.json').read_text())
            self.assertEqual(require_clean_stop(journal, checkpoint, {'run': 'one'}), 7)
            resumed, resumed_opt, loss = self.build()
            # Deliberately consume each RNG after initialization.
            random.random(); self.np.random.random(); self.torch.rand(2)
            restored = restore_state(resumed, resumed_opt, checkpoint, identity={'run': 'one'}, order=order, completed=7)
            result = train_updates(resumed, resumed_opt, order, loss, root/'part.json', start=7,
                                   history=restored['history'], report=lambda _: None)
            assert_same(full.state_dict(), resumed.state_dict())
            assert_same(full_opt.state_dict(), resumed_opt.state_dict())
            self.assertEqual([r['index'] for r in result['history']], order)
            self.assertEqual([r['loss'] for r in result['history']], [r['loss'] for r in expected['history']])
            self.assertEqual(result['completed'], len(order))

    def test_partial_saves_old_steps_and_identity_changes_refuse_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, optimizer, _ = self.build()
            path = publish_state(model, optimizer, None, root/'checkpoint', identity={'run': 1},
                                 order=[0, 1], completed=0, history=[])
            for identity, completed, order in [({'run': 2}, 0, [0, 1]), ({'run': 1}, 1, [0, 1]), ({'run': 1}, 0, [1, 0])]:
                with self.assertRaises(ValueError):
                    restore_state(model, optimizer, path, identity=identity, order=order, completed=completed)
            with self.assertRaises(ValueError):
                require_clean_stop(dict(status='UPDATING', completed=0, pending_step=1), path, {'run': 1})
            with self.assertRaises(FileExistsError):
                publish_state(model, optimizer, None, path, identity={'run': 1}, order=[0, 1], completed=0, history=[])
            (path/'state.pt').write_bytes(b'partial')
            with self.assertRaises(ValueError):
                restore_state(model, optimizer, path, identity={'run': 1}, order=[0, 1], completed=0)
            def interrupted(path):
                raise RuntimeError('interrupted save')
            with self.assertRaises(RuntimeError):
                publish_state(model, optimizer, None, root/'partial', identity={}, order=[], completed=0,
                              history=[], save_adapter=interrupted)
            self.assertFalse((root/'partial').exists())
            self.assertTrue(list(root.glob('partial.incomplete-*')))

    def test_failed_update_retains_pending_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, optimizer, loss = self.build()
            journal = Path(tmp)/'journal.json'
            def failed(index):
                raise RuntimeError('update failed')
            with self.assertRaises(RuntimeError):
                train_updates(model, optimizer, [4], failed, journal)
            self.assertEqual(json.loads(journal.read_text()), dict(status='UPDATING', completed=0, pending_step=1))

    def test_loss_uses_answer_token_weights(self):
        self.assertEqual(weighted_loss([dict(loss=2, response_tokens=3), dict(loss=4, response_tokens=1)]), 2.5)

    def test_configured_checkpoint_interval_covers_expanded_schedule(self):
        model, optimizer, loss = self.build()
        saved = []
        with tempfile.TemporaryDirectory() as tmp:
            result = train_updates(model, optimizer, list(range(8)), loss, Path(tmp)/'journal.json',
                                   checkpoint_interval=2,
                                   save=lambda step, history, clean: saved.append((step, clean)),
                                   report=lambda _: None)
        self.assertEqual(result['completed'], 8)
        self.assertEqual(saved, [(2, False), (4, False), (6, False), (8, False)])

    def test_epoch_and_final_checkpoints_join_periodic_saves(self):
        model, optimizer, loss = self.build()
        saved = []
        with tempfile.TemporaryDirectory() as tmp:
            train_updates(model, optimizer, list(range(8)), loss, Path(tmp)/'journal.json',
                          checkpoint_interval=3, checkpoint_steps={4, 8},
                          save=lambda step, history, clean: saved.append(step), report=lambda _: None)
        self.assertEqual(saved, [3, 4, 6, 8])
