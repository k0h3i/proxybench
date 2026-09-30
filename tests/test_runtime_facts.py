"""Optimizer recording preserves the configured numerical behavior."""

import unittest

from proxybench.training.runtime_facts import optimizer_identity


class RuntimeFactsTests(unittest.TestCase):
    def test_reading_optimizer_identity_does_not_change_settings(self):
        import torch
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        before = dict(optimizer.param_groups[0])
        optimizer_identity(model, optimizer)
        self.assertEqual(optimizer.param_groups[0], before)

    def test_explicit_foreach_is_recorded(self):
        import torch
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), foreach=True)
        self.assertEqual(optimizer_identity(model, optimizer)['groups'][0]['resolved_backend'], 'foreach')
