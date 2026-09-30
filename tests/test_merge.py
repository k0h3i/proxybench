import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class MergeTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional model environment is unavailable')
        self.torch = torch

    def test_finite_checks_use_bounded_views_and_preserve_values(self):
        from proxybench.training.merge import tensor_isfinite
        torch = self.torch
        for dtype in (torch.bfloat16, torch.float32):
            original = torch.arange(120, dtype=dtype).view(10, 12)
            for tensor in (original, original.t(), original[:, ::2], original[:1].expand(10, 12)):
                with self.subTest(dtype=dtype, stride=tensor.stride()):
                    saved = tensor.clone()
                    observed = []
                    real_isfinite = torch.isfinite

                    def inspect(part):
                        self.assertLessEqual(part.numel(), 7)
                        self.assertEqual(part.dtype, dtype)
                        self.assertEqual(part.untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())
                        observed.append(part.numel())
                        return real_isfinite(part)

                    with patch('torch.isfinite', side_effect=inspect):
                        self.assertTrue(tensor_isfinite(tensor, chunk_elements=7))
                    self.assertEqual(sum(observed), tensor.numel())
                    self.assertTrue(torch.equal(tensor, saved))

    def test_nonfinite_values_at_chunk_boundaries_are_rejected(self):
        from proxybench.training.merge import tensor_isfinite
        torch = self.torch
        for dtype in (torch.bfloat16, torch.float32):
            for value in (float('nan'), float('inf'), -float('inf')):
                for index in (0, 6, 7, 14, 19):
                    for transpose in (False, True):
                        with self.subTest(dtype=dtype, value=value, index=index, transpose=transpose):
                            tensor = torch.zeros(4, 5, dtype=dtype)
                            tensor.view(-1)[index] = value
                            if transpose:
                                tensor = tensor.t()
                            self.assertFalse(tensor_isfinite(tensor, chunk_elements=7))

    def test_scalar_empty_and_invalid_chunks(self):
        from proxybench.training.merge import tensor_isfinite
        torch = self.torch
        self.assertTrue(tensor_isfinite(torch.tensor(1.0), chunk_elements=1))
        self.assertFalse(tensor_isfinite(torch.tensor(float('nan')), chunk_elements=1))
        self.assertTrue(tensor_isfinite(torch.empty(0, 3), chunk_elements=1))
        for size in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                tensor_isfinite(torch.tensor(1.0), chunk_elements=size)

    def test_hash_matches_original_bytes_for_contiguous_and_strided_weights(self):
        from proxybench.training.merge import tensor_hash
        torch = self.torch
        for dtype in (torch.bfloat16, torch.float32):
            tensor = torch.tensor([[0.0, -0.0, 1.25], [-2.0, 3.0, 4.0]], dtype=dtype)
            for value in (tensor, tensor.t(), tensor[:, ::2]):
                expected = hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                self.assertEqual(tensor_hash(value), expected)

    def model(self, *, nonfinite=False, change_unadapted=False):
        torch = self.torch

        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = torch.nn.Linear(3, 3, bias=False)
                self.base_layer.weight.data.fill_(1)
                self.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(3, 1, bias=False)})
                self.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(1, 3, bias=False)})
                self.lora_A['default'].weight.data.fill_(0.25)
                self.lora_B['default'].weight.data.fill_(0.5)
                self.active_adapters = ['default']
                self.scaling = {'default': 2.0}

            def get_base_layer(self):
                return self.base_layer

            def forward(self, value):
                return self.base_layer(value) + self.lora_B['default'](self.lora_A['default'](value)) * 2

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = Projection()
                self.untouched = torch.nn.Parameter(torch.ones(2, 3))
                self.config = SimpleNamespace(tie_word_embeddings=False)

            def forward(self, input_ids, use_cache, logits_to_keep):
                assert use_cache is False and logits_to_keep == 1
                inputs = input_ids[:, -1:].float().unsqueeze(-1).expand(-1, -1, 3)
                return SimpleNamespace(logits=self.projection(inputs))

        class Adapter:
            def __init__(self):
                self.base = Base()
                self.safe_merge = None

            def parameters(self):
                return self.base.parameters()

            def get_base_model(self):
                return self.base

            def merge_and_unload(self, safe_merge=False):
                self.safe_merge = safe_merge
                projection = self.base.projection
                layer = projection.base_layer
                with torch.no_grad():
                    layer.weight.add_(projection.lora_B['default'].weight @ projection.lora_A['default'].weight * 2)
                    if nonfinite:
                        self.base.untouched[-1, -1] = float('nan')
                    if change_unadapted:
                        self.base.untouched[-1, -1] = 2
                self.base.projection = layer
                return self.base

        return Adapter()

    def test_merge_retains_safe_merge_hash_checks_and_stages(self):
        from proxybench.training.merge import merge_model, tensor_isfinite
        model = self.model()
        stages = []
        with (tempfile.TemporaryDirectory() as root,
              patch('proxybench.training.merge.host_memory', return_value={'available_bytes': 16 * 1024**3}),
              patch('proxybench.training.merge.tensor_isfinite', side_effect=lambda p: tensor_isfinite(p, chunk_elements=2)) as finite):
            merged = merge_model(model, [1, 2], Path(root), phase=stages.append)
            result = json.loads((Path(root) / 'merge.json').read_text())
            self.assertIs(merged, model.base)
            self.assertTrue(model.safe_merge)
            self.assertEqual(finite.call_count, len(list(merged.parameters())))
            self.assertEqual(result['changed'], ['projection.weight'])
            self.assertEqual(result['score_max_absolute_difference'], 0)
            self.assertEqual(stages, ['hashing base weights', 'reference inference', 'safe merge',
                                      'checking merged weights', 'merged inference'])

    def test_merge_rejects_nonfinite_or_unexpected_changes_without_publication(self):
        from proxybench.training.merge import merge_model
        for arguments, message in (({'nonfinite': True}, 'Nonfinite merged tensor'),
                                   ({'change_unadapted': True}, 'Unexpected changed')):
            with (self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as root,
                  patch('proxybench.training.merge.host_memory', return_value={'available_bytes': 16 * 1024**3})):
                with self.assertRaisesRegex(ValueError, message):
                    merge_model(self.model(**arguments), [1, 2], Path(root))
                self.assertFalse((Path(root) / 'merge.json').exists())
