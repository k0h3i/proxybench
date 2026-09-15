import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class MergedExportTests(unittest.TestCase):
    def test_shards_preserve_values_dtypes_and_tied_alias(self):
        try:
            import torch
            import safetensors
        except ImportError:
            self.skipTest('Optional model environment is unavailable')
        from proxybench.training.checkpoints import validate_checkpoint
        from proxybench.training.merged_export import publish_merged
        from proxybench.training.optimization import tensor_hash

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
