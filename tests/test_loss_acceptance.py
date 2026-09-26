"""CPU proofs of causal loss semantics, and bounded acceptance preparation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.training.loss_acceptance import (
    compare_loss_case, installed_helper_source, launch_acceptance, loss_cases, reference_loss,
)
from proxybench.training.adapters import ResponseCollator
from proxybench.training.sequences import sequence


class LossAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def values(self, rows):
        torch = self.torch
        labels = torch.tensor(rows)
        generator = torch.Generator(device='cpu').manual_seed(42)
        hidden = torch.randn((*labels.shape, 7), generator=generator, requires_grad=True)
        weight = torch.randn((11, 7), generator=generator, requires_grad=True)
        return hidden, weight, labels

    def test_loss_and_gradients_match_explicit_next_token_pairs(self):
        torch = self.torch
        for name, rows in loss_cases().items():
            with self.subTest(case=name):
                hidden, weight, labels = self.values(rows)
                loss = reference_loss(hidden, weight, labels)
                gradient_hidden, gradient_weight = torch.autograd.grad(loss, (hidden, weight))
                expected_hidden = torch.zeros_like(hidden)
                expected_weight = torch.zeros_like(weight)
                expected_losses = []
                count = int((labels[:, 1:] != -100).sum())
                # Independent enumerated causal targets and analytic softmax gradient.
                for batch in range(labels.shape[0]):
                    for target_position in range(1, labels.shape[1]):
                        target = int(labels[batch, target_position])
                        if target == -100:
                            continue
                        predictor = hidden.detach()[batch, target_position - 1]
                        logits = weight.detach() @ predictor
                        expected_losses.append(torch.logsumexp(logits, 0) - logits[target])
                        derivative = logits.softmax(0)
                        derivative[target] -= 1
                        derivative /= count
                        expected_hidden[batch, target_position - 1] = derivative @ weight.detach()
                        expected_weight += derivative[:, None] * predictor[None, :]
                torch.testing.assert_close(loss, torch.stack(expected_losses).mean(), rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(gradient_hidden, expected_hidden, rtol=2e-5, atol=2e-7)
                torch.testing.assert_close(gradient_weight, expected_weight, rtol=2e-5, atol=2e-7)

    def test_first_response_uses_last_prompt_hidden_and_termination_is_supervised(self):
        torch = self.torch
        hidden, weight, labels = self.values([[-100, -100, 3, 9]])
        gradient, = torch.autograd.grad(reference_loss(hidden, weight, labels), (hidden,))
        self.assertGreater(float(gradient[0, 1].abs().sum()), 0)
        self.assertGreater(float(gradient[0, 2].abs().sum()), 0)
        self.assertTrue(torch.equal(gradient[0, 0], torch.zeros(7)))
        self.assertTrue(torch.equal(gradient[0, 3], torch.zeros(7)))

    def test_ignored_predictors_do_not_change_loss_or_weight_gradient(self):
        torch = self.torch
        hidden, weight, labels = self.values(loss_cases()['right_padding'])
        original = reference_loss(hidden, weight, labels)
        original_weight, = torch.autograd.grad(original, (weight,))
        changed = hidden.detach().clone()
        changed[0, [0, 3, 4, 5]] = 1000
        result = reference_loss(changed, weight, labels)
        changed_weight, = torch.autograd.grad(result, (weight,))
        torch.testing.assert_close(original, result, rtol=0, atol=0)
        torch.testing.assert_close(original_weight, changed_weight, rtol=0, atol=0)

    def test_unequal_lengths_use_token_mean_not_sequence_mean(self):
        torch = self.torch
        hidden, weight, labels = self.values(loss_cases()['unequal_lengths'])
        batched = reference_loss(hidden, weight, labels)
        separate = [reference_loss(hidden[i:i+1], weight, labels[i:i+1]) for i in range(2)]
        torch.testing.assert_close(batched, (separate[0] * 3 + separate[1] * 2) / 5)
        self.assertGreater(abs(float((batched - (separate[0] + separate[1]) / 2).detach())), 1e-4)

    def test_adapter_gradients_follow_the_causal_hidden_gradient(self):
        torch = self.torch
        hidden, weight, labels = self.values(loss_cases()['unequal_lengths'])
        generator = torch.Generator(device='cpu').manual_seed(19)
        inputs = torch.randn((*labels.shape, 5), generator=generator)
        adapter_a = torch.randn((5, 2), generator=generator, requires_grad=True)
        adapter_b = torch.randn((2, 7), generator=generator, requires_grad=True)
        adapted = hidden.detach() + inputs @ adapter_a @ adapter_b
        loss = reference_loss(adapted, weight, labels)
        gradient_hidden, gradient_a, gradient_b = torch.autograd.grad(loss, (adapted, adapter_a, adapter_b))
        flat_inputs, flat_gradient = inputs.flatten(0, 1), gradient_hidden.flatten(0, 1)
        expected_a = flat_inputs.T @ flat_gradient @ adapter_b.detach().T
        expected_b = (flat_inputs @ adapter_a.detach()).T @ flat_gradient
        torch.testing.assert_close(gradient_a, expected_a)
        torch.testing.assert_close(gradient_b, expected_b)

    def test_sequence_and_collator_keep_first_response_termination_and_padding(self):
        class Tokenizer:
            eos_token_id = 9
            def apply_chat_template(self, messages, **kwargs):
                prefix = [1, 2]
                return prefix if kwargs.get('add_generation_prompt') else prefix + [3] * len(messages[-1]['content']) + [9]
        items = [sequence(Tokenizer(), 'prompt', answer, context_cap=12) for answer in ('a', 'aaa')]
        batch = ResponseCollator(0)(items)
        self.assertEqual(batch['labels'].tolist(), [[-100, -100, 3, 9, -100, -100], [-100, -100, 3, 3, 3, 9]])
        hidden, weight, _ = self.values(batch['labels'].tolist())
        gradient, = self.torch.autograd.grad(reference_loss(hidden, weight, batch['labels']), (hidden,))
        self.assertGreater(float(gradient[0, 1].abs().sum()), 0)
        self.assertGreater(float(gradient[0, 2].abs().sum()), 0)
        self.assertTrue((gradient[0, 3:] == 0).all())

    def test_empty_supervision_and_wrong_shapes_fail(self):
        hidden, weight, labels = self.values([[-100, -100]])
        with self.assertRaisesRegex(ValueError, 'supervised next token'):
            reference_loss(hidden, weight, labels)
        with self.assertRaisesRegex(ValueError, 'shapes differ'):
            reference_loss(hidden, weight, labels[:, :-1])

    def test_forward_only_case_never_claims_or_requests_unsupported_gradients(self):
        torch = self.torch
        tolerance = dict(loss_absolute=2e-5, gradient_relative_l2=2e-4)
        with patch('torch.autograd.grad', side_effect=AssertionError('unsupported backward')):
            report = compare_loss_case(torch.tensor(1.0), torch.tensor(1.0), (), (),
                                       torch.tensor([[-100, 1]]), tolerance, gradients=False)
        self.assertTrue(report['passed'])
        self.assertEqual(report['gradient_status'], 'UNSUPPORTED_DTYPE')
        self.assertNotIn('hidden_gradient', report)
        report = compare_loss_case(torch.tensor(1.1), torch.tensor(1.0), (), (),
                                   torch.tensor([[-100, 1]]), tolerance, gradients=False)
        self.assertFalse(report['passed'])

    def test_installed_helper_applies_one_shift_without_gpu_import(self):
        identity = installed_helper_source()
        self.assertIs(identity['internal_shift'], True)
        self.assertEqual(identity['filter_argument'], 'accuracy_threshold')
        self.assertEqual(len(identity['sha256']), 64)

    def test_continuation_launch_keeps_recipe_and_uses_bounded_fresh_workers(self):
        configuration = json.loads(Path('configs/training.json').read_text())
        requests, limits_seen = [], []
        def supervise(command, output, limits, **kwargs):
            self.assertEqual(command[1:4], ['-m', 'proxybench.training.loss_acceptance', '_worker'])
            requests.append(json.loads(Path(command[-1]).read_text()))
            limits_seen.append(limits)
            return 'EXITED'
        with tempfile.TemporaryDirectory() as tmp, \
                patch('proxybench.execution.live.supervise', side_effect=supervise), \
                patch('proxybench.training.loss_acceptance.compare_continuation', return_value={'status': 'PASSED'}):
            root = Path(tmp) / 'fresh'
            launch_acceptance(root, 'configs/training.json', continuation=True)
            with self.assertRaises(FileExistsError):
                launch_acceptance(root, 'configs/training.json', continuation=True)
        self.assertEqual([(r['updates'], r['resume']) for r in requests], [(4, False), (2, False), (4, True)])
        for request in requests:
            # load_config expands paths, but numerical recipe and full schedule stay fixed.
            for key in ('epochs', 'seed', 'checkpoint_interval', 'batch_size', 'accumulation', 'learning_rate'):
                self.assertEqual(request['configuration'][key], configuration[key])
        self.assertTrue(all(limit['phase_seconds'] == 1800 and limit['total_seconds'] == 5400 for limit in limits_seen))


if __name__ == '__main__':
    unittest.main()
