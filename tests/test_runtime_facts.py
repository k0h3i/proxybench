"""Optimizer recording preserves the configured numerical behavior."""

import unittest

from proxybench.training.runtime_facts import optimizer_identity


class RuntimeFactsTests(unittest.TestCase):
    def test_optimizer_groups_record_defaults_without_changing_them(self):
        import torch
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        before = dict(optimizer.param_groups[0])
        value = optimizer_identity(model, optimizer)
        self.assertEqual(optimizer.param_groups[0], before)
        group = value['groups'][0]
        self.assertEqual(group['parameters'], ['weight', 'bias'])
        self.assertEqual(group['settings']['betas'], (0.9, 0.999))
        self.assertEqual(group['settings']['eps'], 1e-8)
        self.assertIsNone(group['settings']['fused'])
        self.assertEqual(group['resolved_backend'], 'single_tensor')

    def test_explicit_foreach_is_recorded(self):
        import torch
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), foreach=True)
        self.assertEqual(optimizer_identity(model, optimizer)['groups'][0]['resolved_backend'], 'foreach')
