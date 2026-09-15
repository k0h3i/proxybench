"""Boundaries and adapter coverage for the bounded training test."""
import json
from pathlib import Path
import unittest
import tempfile

from proxybench.training.smoke import (adapter_targets, generation_input, identity, prepare,
                                       ResponseCollator, accepted_rows, GenerationRecorder,
                                       require_same_adapter, save_adapter)


class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, messages, **kwargs):
        prefix = [1, 2, 3]
        return prefix if len(messages) == 1 else prefix + [4, 5, 99]
    def decode(self, ids, **kwargs):
        return 'answer<end>' if ids == [4, 5, 99] else '<end>'


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.configuration = json.loads((Path(__file__).resolve().parents[1] / 'configs/qwen35-4b-smoke.json').read_text())

    def test_wrong_model_and_dataset_refused(self):
        identity(self.configuration)
        for key, wrong in [('model_id', 'Qwen/Qwen3.5-9B'), ('dataset_sha256', 'changed')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                identity({**self.configuration, key: wrong})

    def test_changed_dataset_bytes_refused_before_manifest_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.jsonl'
            path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Dataset bytes differ'):
                accepted_rows(path, self.configuration)

    def test_prefix_mask_termination_and_saved_order(self):
        rows = [{'messages': [{'content': 'prompt'}, {'content': 'answer'}]} for _ in range(15)]
        items, bounds = prepare(Tokenizer(), rows, self.configuration)
        self.assertEqual(items[0]['labels'], [-100, -100, -100, 4, 5, 99])
        self.assertEqual(generation_input(items[0]), [1, 2, 3])
        self.assertEqual(sorted(bounds['order'][:15]), list(range(15)))
        self.assertEqual(sorted(bounds['order'][15:]), list(range(15)))
        self.assertEqual(prepare(Tokenizer(), rows, self.configuration)[1], bounds)
        with self.assertRaises(ValueError):
            prepare(Tokenizer(), rows, {**self.configuration, 'context_ceiling': 5})

    def test_language_only_adapter_scope(self):
        class Linear:
            pass
        names = ['model.layers.0.self_attn.q_proj', 'model.layers.1.linear_attn.in_proj_qkv',
                 'model.layers.0.mlp.down_proj', 'visual.layers.0.mlp.up_proj', 'lm_head', 'embed_tokens']
        targets, _ = adapter_targets([(name, Linear()) for name in names], Linear)
        self.assertEqual(targets, names[:3])
        with self.assertRaises(ValueError):
            adapter_targets([(names[0], Linear())], Linear)

    def test_actual_collator_preserves_padding_and_termination(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Requires the separate training environment')
        items = [dict(input_ids=[1, 2, 99], attention_mask=[1, 1, 1], labels=[-100, 2, 99]),
                 dict(input_ids=[1, 99], attention_mask=[1, 1], labels=[-100, 99])]
        batch = ResponseCollator(99)(items)
        self.assertEqual(batch['input_ids'].tolist(), [[1, 2, 99], [1, 99, 99]])
        self.assertEqual(batch['labels'].tolist(), [[-100, 2, 99], [-100, 99, -100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch['labels'].dtype, torch.long)

    def test_adapter_survives_interrupted_generation_and_reloads_exactly(self):
        try:
            import torch
            from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
            from safetensors.torch import load_file
        except ImportError:
            self.skipTest('Requires the separate training environment')

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class TokenizerMetadata:
            def save_pretrained(self, path):
                (Path(path) / 'tokenizer_config.json').write_text('{}\n')

        configuration = LoraConfig(r=2, target_modules=['linear'])
        model = get_peft_model(Tiny(), configuration)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.fill_(0.25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_adapter(model, TokenizerMetadata(), root / 'adapter')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with GenerationRecorder(root / 'partial.jsonl', 0) as recorder:
                    recorder.put(torch.tensor([[1, 2, 3]]))
                    recorder.put(torch.tensor([7]))
                    recorder.put(torch.tensor([8]))
                    self.assertEqual((root / 'partial.jsonl').read_text(), '7\n8\n')
                    raise RuntimeError('interrupted')
            self.assertEqual((root / 'partial.jsonl').read_text(), '7\n8\n')
            state = load_file(str(root / 'adapter/adapter_model.safetensors'))
            restored = get_peft_model(Tiny(), LoraConfig(r=2, target_modules=['linear']))
            set_peft_model_state_dict(restored, state)
            require_same_adapter(state, get_peft_model_state_dict(restored, save_embedding_layers=False))
            with self.assertRaises(FileExistsError):
                save_adapter(model, TokenizerMetadata(), root / 'adapter')
            name = next(iter(state))
            wrong = dict(state)
            wrong[name] = state[name] + 1
            with self.assertRaisesRegex(ValueError, 'tensor differs'):
                require_same_adapter(state, wrong)
            wrong[name] = state[name].double()
            with self.assertRaisesRegex(ValueError, 'tensor differs'):
                require_same_adapter(state, wrong)
            with self.assertRaisesRegex(ValueError, 'names differ'):
                require_same_adapter(state, {})


if __name__ == '__main__':
    unittest.main()
