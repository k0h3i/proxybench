import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class MergedExportTests(unittest.TestCase):
    def test_untied_weights_use_cpu_checks_and_reject_nonfinite_values(self):
        try:
            import torch
            from safetensors import safe_open
        except ImportError:
            self.skipTest('Optional model environment is unavailable')
        from proxybench.training.merge import tensor_hash, tensor_isfinite
        from proxybench.training.merged_export import publish_merged

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 3, dtype=torch.bfloat16)
        model.model.embed_tokens.weight.data.fill_(1)
        model.lm_head = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
        model.lm_head.weight = torch.nn.Parameter(torch.full((3, 4), 2, dtype=torch.bfloat16).t())
        model.config = SimpleNamespace(
            architectures=['Qwen3_5ForCausalLM'], tie_word_embeddings=False,
            save_pretrained=lambda p: (Path(p) / 'config.json').write_text(json.dumps(dict(
                architectures=['Qwen3_5ForCausalLM'], tie_word_embeddings=False))))
        model.generation_config = SimpleNamespace(save_pretrained=lambda p: (Path(p) / 'generation_config.json').write_text('{}'))
        tokenizer = SimpleNamespace(save_pretrained=lambda p: (Path(p) / 'tokenizer.json').write_text('{}'))
        expected = {n: tensor_hash(p) for n, p in model.named_parameters()}
        original_pointers = {p.data_ptr() for p in model.parameters()}

        def inspect_copy(tensor):
            self.assertEqual(tensor.device.type, 'cpu')
            self.assertNotIn(tensor.data_ptr(), original_pointers)
            return tensor_isfinite(tensor, chunk_elements=2)

        with (tempfile.TemporaryDirectory() as root,
              patch('proxybench.training.merged_export.shutil.disk_usage', return_value=SimpleNamespace(free=1024**4)),
              patch('proxybench.training.merged_export.host_memory', return_value={'available_bytes': 16 * 1024**3}),
              patch('proxybench.training.merged_export.tensor_isfinite', side_effect=inspect_copy) as finite):
            path = Path(root) / 'model'
            result = publish_merged(model, tokenizer, path, expected, shard_bytes=16)
            self.assertEqual(finite.call_count, 2)
            self.assertEqual(result['identity']['tied_aliases'], {})
            self.assertNotEqual(expected['model.embed_tokens.weight'], expected['lm_head.weight'])
            index = json.loads((path / 'model.safetensors.index.json').read_text())
            for name, parameter in model.named_parameters():
                with safe_open(path / index['weight_map'][name], framework='pt', device='cpu') as stream:
                    self.assertTrue(torch.equal(stream.get_tensor(name), parameter))
            for position, value in enumerate((float('nan'), float('inf'), -float('inf'))):
                model.lm_head.weight.data[-1, -1] = value
                failed = Path(root) / f'nonfinite-{position}'
                matching_hashes = {n: tensor_hash(p) for n, p in model.named_parameters()}
                with self.assertRaisesRegex(ValueError, 'Merged value differs: lm_head.weight'):
                    publish_merged(model, tokenizer, failed, matching_hashes, shard_bytes=16)
                self.assertFalse(failed.exists())
                incomplete = next(Path(root).glob(f'nonfinite-{position}.incomplete-*'))
                self.assertFalse((incomplete / 'complete.json').exists())

    def test_shards_preserve_values_dtypes_and_tied_alias(self):
        try:
            import torch
            import safetensors
        except ImportError:
            self.skipTest('Optional model environment is unavailable')
        from proxybench.training.checkpoints import validate_checkpoint
        from proxybench.training.merged_export import publish_merged
        from proxybench.training.merge import tensor_hash

        class Config:
            architectures = ['Qwen3_5ForCausalLM']
            tie_word_embeddings = True

            def save_pretrained(self, path):
                (Path(path) / 'config.json').write_text(json.dumps(dict(
                    architectures=self.architectures, tie_word_embeddings=True)))

        class Tokenizer:
            def save_pretrained(self, path):
                (Path(path) / 'tokenizer.json').write_text('{}')

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 3, dtype=torch.bfloat16)
        model.model.norm = torch.nn.LayerNorm(3)
        model.lm_head = torch.nn.Linear(3, 4, bias=False)
        model.lm_head.weight = model.model.embed_tokens.weight
        model.config = Config()
        model.generation_config = SimpleNamespace(save_pretrained=lambda p: (Path(p) / 'generation_config.json').write_text('{}'))
        expected = {n: tensor_hash(p) for n, p in model.named_parameters()}
        with (tempfile.TemporaryDirectory() as root,
              patch('proxybench.training.merged_export.shutil.disk_usage', return_value=SimpleNamespace(free=1024**4)),
              patch('proxybench.training.merged_export.host_memory', return_value={'available_bytes': 16 * 1024**3})):
            path = Path(root) / 'model'
            result = publish_merged(model, Tokenizer(), path, expected, shard_bytes=16)
            self.assertEqual(result['identity']['tied_aliases'], {'lm_head.weight': 'model.embed_tokens.weight'})
            self.assertEqual(result['tensors']['model.embed_tokens.weight']['dtype'], 'torch.bfloat16')
            self.assertEqual(result['tensors']['model.norm.weight']['dtype'], 'torch.float32')
            self.assertEqual(len(list(path.glob('*.safetensors'))), 3)
            with self.assertRaises(FileExistsError):
                publish_merged(model, Tokenizer(), path, expected)
            (path / 'model-00001.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash'):
                validate_checkpoint(path)
            wrong = {**expected, 'model.norm.weight': 'bad'}
            with self.assertRaisesRegex(ValueError, 'differs'):
                publish_merged(model, Tokenizer(), Path(root) / 'failed', wrong)
            self.assertFalse((Path(root) / 'failed').exists())
            incomplete = next(Path(root).glob('failed.incomplete-*'))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                validate_checkpoint(incomplete)
