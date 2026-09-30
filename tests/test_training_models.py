"""Pinned model selection and portable metadata."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from proxybench.training.adapters import digest
from proxybench.training import runtime


class TrainingModelTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        stack.enter_context(patch('pathlib.Path.cwd', return_value=self.root))
        (self.root / 'configs').mkdir()
        self.configs = {}
        for model_id, spec in runtime.supported_bases().items():
            base = self.root / spec['base_path']
            base.mkdir(parents=True)
            (base / 'config.json').write_text(json.dumps({'synthetic_model': model_id}))
            (base / 'model.safetensors').write_bytes(b'synthetic weights')
            (base / 'tokenizer.json').write_text('{}')
            (base / 'tokenizer_config.json').write_text('{}')
            if model_id == runtime.NINE_B_MODEL:
                (base / 'model-info.json').write_text('{"gpu_loading_and_training":"NOT_RUN"}')
            manifest = self.root / spec['base_manifest']
            manifest.write_text(json.dumps(dict(model_id=model_id, model_revision=spec['model_revision'],
                                                 files={p.name: digest(p) for p in base.iterdir()})))
            pin = 'BASE_MANIFEST_SHA256' if model_id == runtime.BASE_MODEL else 'NINE_B_MANIFEST_SHA256'
            stack.enter_context(patch.object(runtime, pin, digest(manifest)))
            self.configs[model_id] = dict(model_id=model_id, model_revision=spec['model_revision'],
                                         base_path=str(base), base_manifest=str(manifest),
                                         context_tokens=5120, rank=8, alpha=16, dropout=0, seed=42)
        # Make portable manifest paths resolve inside the synthetic project too.
        stack.enter_context(patch('proxybench.training.runtime.supported_bases',
            side_effect=lambda: {
                runtime.BASE_MODEL: dict(model_revision=runtime.BASE_REVISION,
                    base_path='artifacts/models/Qwen3.5-4B', base_manifest=str(self.root / 'configs/base-model.json'),
                    manifest_sha256=runtime.BASE_MANIFEST_SHA256),
                runtime.NINE_B_MODEL: dict(model_revision=runtime.NINE_B_REVISION,
                    base_path='artifacts/models/Qwen3.5-9B', base_manifest=str(self.root / 'configs/base-model-9b.json'),
                    manifest_sha256=runtime.NINE_B_MANIFEST_SHA256),
            }))

    def test_each_model_selects_the_full_configuration_without_gpu_packages(self):
        for model_id, config in self.configs.items():
            with self.subTest(model=model_id), patch.dict(sys.modules, torch=None, unsloth=None, huggingface_hub=None):
                selected = runtime.select_base(self.configs[runtime.BASE_MODEL], config['base_path'])
                self.assertEqual(selected, config)
                self.assertEqual(runtime.base_snapshot(selected), config['base_path'])
        self.assertEqual(self.configs[runtime.BASE_MODEL]['model_id'], runtime.BASE_MODEL)

    def test_wrong_revision_manifest_shard_and_metadata_are_rejected(self):
        config = self.configs[runtime.NINE_B_MODEL]
        with self.assertRaisesRegex(ValueError, 'Unsupported base'):
            runtime.base_snapshot({**config, 'model_revision': 'main'})
        manifest = Path(config['base_manifest'])
        original = manifest.read_bytes()
        manifest.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'pinned identity'):
            runtime.base_snapshot(config)
        manifest.write_bytes(original)
        for name in ('model.safetensors', 'model-info.json'):
            path = Path(config['base_path']) / name
            original = path.read_bytes()
            path.write_bytes(b'tampered')
            with self.subTest(file=name), self.assertRaisesRegex(ValueError, 'pinned hash'):
                runtime.base_snapshot(config)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                runtime.base_snapshot(config)
            path.write_bytes(original)
        extra = Path(config['base_path']) / 'injected.py'
        extra.write_text('untrusted')
        with self.assertRaisesRegex(ValueError, 'unexpected file'):
            runtime.base_snapshot(config)

    def test_unknown_swapped_artifact_path_symlink_and_traversal_are_rejected(self):
        config = self.configs[runtime.NINE_B_MODEL]
        root = Path(config['base_path'])
        with self.assertRaisesRegex(ValueError, 'retained Qwen3.5-9B'):
            runtime.base_snapshot({**config, 'base_path': self.configs[runtime.BASE_MODEL]['base_path']})
        alias = self.root / 'alias'
        alias.symlink_to(root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            runtime.select_base(config, alias)
        with self.assertRaisesRegex(ValueError, 'parent traversal'):
            runtime.select_base(config, str(root / '..' / root.name))
        unknown = self.root / 'unknown'
        unknown.mkdir()
        (unknown / 'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'supported pinned checkpoint'):
            runtime.select_base(config, unknown)

    def test_revision_specific_download_metadata_is_allowed(self):
        config = self.configs[runtime.NINE_B_MODEL]
        trees = Path(config['base_path']) / '.cache/huggingface/trees'
        trees.mkdir(parents=True)
        (trees / (runtime.NINE_B_REVISION + '.json')).write_text('{}')
        runtime.base_snapshot(config)
        (trees / (runtime.BASE_REVISION + '.json')).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unexpected file'):
            runtime.base_snapshot(config)

    def test_adapter_selection_records_9b_without_changing_dtype_or_targets(self):
        class Linear:
            pass
        model = SimpleNamespace(named_modules=lambda: [(f'model.layers.0.{branch}.q_proj', Linear())
                                for branch in ('self_attn', 'linear_attn', 'mlp')],
                                named_parameters=lambda: [('lora_A', SimpleNamespace(requires_grad=True))],
                                peft_config={'default': SimpleNamespace()})
        unsloth = SimpleNamespace(FastLanguageModel=SimpleNamespace(get_peft_model=Mock(return_value=model)))
        with patch.dict(sys.modules, unsloth=unsloth, torch=SimpleNamespace(nn=SimpleNamespace(Linear=Linear))):
            runtime.attach_adapter(model, self.configs[runtime.NINE_B_MODEL], self.root)
        self.assertEqual(model.peft_config['default'].base_model_name_or_path, runtime.NINE_B_MODEL)
        self.assertEqual(model.peft_config['default'].revision, runtime.NINE_B_REVISION)
        self.assertEqual(unsloth.FastLanguageModel.get_peft_model.call_args.kwargs['use_gradient_checkpointing'], 'unsloth')

    def test_load_base_keeps_bf16_nonquantized_text_only_loading(self):
        model = type('Qwen3_5ForCausalLM', (), {})()
        model.config = SimpleNamespace()
        loader = Mock(return_value=(model, 'tokenizer'))
        torch = SimpleNamespace(bfloat16='BF16', cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True))
        with patch.dict(sys.modules, unsloth=SimpleNamespace(FastLanguageModel=SimpleNamespace(from_pretrained=loader)), torch=torch):
            runtime.load_base(self.configs[runtime.NINE_B_MODEL], 'authenticated-snapshot')
        arguments = loader.call_args.kwargs
        self.assertEqual(arguments['dtype'], 'BF16')
        self.assertFalse(arguments['load_in_4bit'])
        self.assertTrue(arguments['load_in_16bit'])
        self.assertTrue(arguments['text_only'])
        self.assertTrue(arguments['local_files_only'])

    def test_wrong_adapter_fails_before_export_or_validation_gpu_import(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path=runtime.NINE_B_MODEL,
                                                                   revision=runtime.NINE_B_REVISION)))
        from proxybench.training.validation import validate_adapter
        for operation in (runtime.export_adapter, validate_adapter):
            with self.subTest(operation=operation.__name__), patch.dict(sys.modules, unsloth=None):
                with self.assertRaisesRegex(ValueError, 'selected pinned base'):
                    operation(adapter, self.root / 'not-created', self.configs[runtime.BASE_MODEL])
        self.assertFalse((self.root / 'not-created').exists())
        runtime.require_adapter_base(adapter, self.configs[runtime.NINE_B_MODEL])

    def test_adapter_metadata_must_be_a_bounded_regular_json_object(self):
        import os
        adapter = self.root / 'adapter'
        adapter.mkdir()
        metadata = adapter / 'adapter_config.json'
        config = self.configs[runtime.NINE_B_MODEL]
        for value in ('[]', 'null', '{broken', '{"revision":"a","revision":"b"}'):
            metadata.write_text(value)
            with self.subTest(metadata=value), self.assertRaises(ValueError):
                runtime.require_adapter_base(adapter, config)
        metadata.unlink()
        os.mkfifo(metadata)
        with self.assertRaisesRegex(ValueError, 'missing or invalid'):
            runtime.require_adapter_base(adapter, config)

    def test_validation_rejects_default_4b_tokenizer_for_9b_before_launch(self):
        from proxybench.training.validation import validate_runtime
        adapter = self.root / 'adapter-9b'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path=runtime.NINE_B_MODEL,
                                                                   revision=runtime.NINE_B_REVISION)))
        tokenizer = self.root / 'adapter-4b'
        tokenizer.mkdir()
        (tokenizer / 'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path=runtime.BASE_MODEL,
                                                                     revision=runtime.BASE_REVISION)))
        args = SimpleNamespace(training_config='training', config='inference', adapter=adapter, run_dir=self.root / 'output')
        inference = dict(tokenizer=str(tokenizer), model='default-4b.gguf')
        with patch('proxybench.extraction.runtime.load_config', side_effect=[self.configs[runtime.NINE_B_MODEL], inference]), \
             patch('proxybench.training.validation.require_project_environment'), patch.object(runtime, 'launch') as launch:
            with self.assertRaisesRegex(ValueError, 'selected pinned base'):
                validate_runtime(args)
        launch.assert_not_called()
        self.assertFalse(args.run_dir.exists())

    def test_validation_rejects_4b_gguf_metadata_with_a_9b_adapter(self):
        from proxybench.training.validation import validate_runtime
        adapter = self.root / 'adapter-9b'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path=runtime.NINE_B_MODEL,
                                                                   revision=runtime.NINE_B_REVISION)))
        gguf = self.root / 'gguf-4b'
        gguf.mkdir()
        (gguf / 'model-info.json').write_text(json.dumps(dict(base_model=runtime.BASE_MODEL, base_revision=runtime.BASE_REVISION)))
        args = SimpleNamespace(training_config='training', config='inference', adapter=adapter, run_dir=self.root / 'output')
        inference = dict(tokenizer=str(adapter), model=str(gguf / 'model.gguf'))
        with patch('proxybench.extraction.runtime.load_config', side_effect=[self.configs[runtime.NINE_B_MODEL], inference]), \
             patch('proxybench.training.validation.require_project_environment'), patch.object(runtime, 'launch') as launch:
            with self.assertRaisesRegex(ValueError, 'GGUF model metadata differs'):
                validate_runtime(args)
        launch.assert_not_called()
        self.assertFalse(args.run_dir.exists())

    def test_cli_binds_selected_model_before_run_identity_and_launch(self):
        selected = self.configs[runtime.NINE_B_MODEL]
        config = dict(self.configs[runtime.BASE_MODEL], system_prompt='synthetic-prompt', limits={})
        run = SimpleNamespace(state={}, save=Mock(), path=self.root / 'run')
        with patch('proxybench.runstate.Run') as Run, patch.object(runtime, 'load_config', return_value=config), \
             patch.object(runtime, 'select_base', return_value={**config, **selected}), \
             patch.object(runtime, 'base_snapshot'), patch.object(runtime, 'digest', return_value='prompt'), \
             patch('proxybench.training.dataset.read_release', return_value=({}, {'synthetic': 'dataset'})), \
             patch('proxybench.training.resume.training_preflight'), patch.object(runtime, 'launch', return_value='EXITED') as launch, \
             patch.object(runtime, 'training_status', return_value='TRAINED'), patch.object(runtime, 'resource_floor'):
            Run.return_value.__enter__.return_value = run
            runtime.cli(SimpleNamespace(config='recipe', model=selected['base_path'], dataset='dataset', run_dir=self.root / 'run'))
            effective = {**config, **selected}
            self.assertEqual(Run.call_args.kwargs['identity']['recipe'], runtime.binding(effective))
            self.assertEqual(Run.call_args.kwargs['config'], effective)
            self.assertEqual(launch.call_args.args[2]['model_id'], runtime.NINE_B_MODEL)

    def test_unsupported_cli_model_rejects_before_creating_run_or_launching(self):
        config = dict(self.configs[runtime.BASE_MODEL])
        with patch.object(runtime, 'load_config', return_value=config), patch('proxybench.runstate.Run') as Run, \
             patch.object(runtime, 'launch') as launch:
            with self.assertRaisesRegex(ValueError, 'regular file'):
                runtime.cli(SimpleNamespace(config='recipe', model=self.root / 'missing', dataset='dataset', run_dir='run'))
        Run.assert_not_called()
        launch.assert_not_called()

    def test_cli_parses_new_flags_without_changing_default_4b_recipe(self):
        parser = argparse.ArgumentParser()
        runtime.add_cli(parser.add_subparsers(required=True))
        args = parser.parse_args(['train', '--model', 'artifacts/models/Qwen3.5-9B', '--run-dir', 'outside'])
        self.assertEqual(args.model, 'artifacts/models/Qwen3.5-9B')
        defaults = parser.parse_args(['train', '--run-dir', 'outside'])
        self.assertIsNone(defaults.model)
        self.assertEqual(defaults.config, 'configs/training.json')



if __name__ == '__main__':
    unittest.main()
