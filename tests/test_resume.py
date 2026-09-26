"""Reject unsafe resume before launch, deserialization, or GPU imports."""

import builtins
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from proxybench.execution.resources import durable_json
from proxybench.training.adapters import digest
from proxybench.training.resume import training_preflight
from proxybench.training.runtime import binding, resume_training, train
from proxybench.training.trajectory import load_state, publish_state, sample_order


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prompt = self.root / 'prompt.txt'
        self.prompt.write_text('Synthetic policy\n')
        self.output = self.root / 'run'
        self.output.mkdir()
        self.config = dict(batch_size=1, accumulation=1, max_grad_norm=1.0, epochs=2, seed=42,
                           system_prompt=str(self.prompt), limits=dict(total_seconds=30))
        self.rows = dict(training=[{}, {}], development=[])
        self.manifest = dict(system_prompt=dict(sha256=digest(self.prompt)))
        self.order = sample_order(2, 2, 42)
        self.identity = dict(dataset=binding(self.manifest), recipe=binding(self.config), order=binding(self.order))
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.checkpoint = publish_state(self.model, self.optimizer, None, self.output / 'checkpoints' / 'step-0',
                                        identity=self.identity, order=self.order, completed=0, history=[])
        self.journal = dict(status='CLEAN_STOP', completed=0, pending_step=None,
                            checkpoint=str(self.checkpoint), checkpoint_sha256=digest(self.checkpoint / 'manifest.json'))
        self.inputs = dict(identity=self.identity, order=self.order)
        self.run = dict(schema='proxybench-run-v1', operation='train', configuration=self.config,
                        identity=dict(operation='train', dataset=self.identity['dataset'],
                                      recipe=self.identity['recipe'], prompt=digest(self.prompt)),
                        dataset=str(self.root / 'dataset'), status='CLEAN_STOP', consumed_seconds=7,
                        resource_limit_seconds=30, pending_charge=None)
        self.write('training-inputs.json', self.inputs)
        self.write('training-journal.json', self.journal)
        self.write('run.json', self.run)
        self.write('training-result.json', dict(status='CLEAN_STOP', completed=0, history=[]))
        reader = patch('proxybench.training.dataset.read_release', side_effect=lambda _: (self.rows, self.manifest))
        reader.start()
        self.addCleanup(reader.stop)
        original_import = builtins.__import__
        def forbid_gpu(name, *args, **kwargs):
            if name.split('.')[0] in {'unsloth', 'transformers', 'peft', 'unsloth_zoo'}:
                raise AssertionError('GPU import reached')
            return original_import(name, *args, **kwargs)
        importer = patch('builtins.__import__', side_effect=forbid_gpu)
        importer.start()
        self.addCleanup(importer.stop)

    def write(self, name, value):
        # Noncanonical whitespace detects any accidental rewrite of provenance.
        (self.output / name).write_text(json.dumps(value, indent=3) + '\n\n')

    def snapshot(self):
        return {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob('*')
                if p.is_file() and p.name != '.writer.lock'}

    def preflight(self):
        return training_preflight(self.run['dataset'], self.output, self.config, resume=True)

    def rejected(self, pattern=None):
        before = self.snapshot()
        with patch('proxybench.training.runtime.launch', side_effect=AssertionError('Worker launched')):
            with self.assertRaisesRegex(ValueError, pattern or '.'):
                resume_training(self.output)
        with self.assertRaisesRegex(ValueError, pattern or '.'):
            train(self.run['dataset'], self.output, self.config, resume=True)
        self.assertEqual(self.snapshot(), before)

    def refresh_checkpoint(self, *, state=None, manifest=None):
        path = self.checkpoint / 'manifest.json'
        value = manifest if manifest is not None else json.loads(path.read_text())
        if state is not None:
            torch.save(state, self.checkpoint / 'state.pt')
            value['files']['state.pt'] = digest(self.checkpoint / 'state.pt')
        durable_json(path, value)
        durable_json(self.checkpoint / 'complete.json', dict(status='COMPLETE', manifest_sha256=digest(path)))
        self.journal['checkpoint_sha256'] = digest(path)
        self.write('training-journal.json', self.journal)

    def test_statuses_without_checkpoint_fail_intentionally(self):
        for status in ('READY', 'BOUNDARY', 'UPDATING', 'TRAINED', 'COMPLETE', None, []):
            with self.subTest(status=status):
                self.write('training-journal.json', dict(status=status, completed=0, pending_step=None))
                self.rejected('CLEAN_STOP')

    def test_malformed_journals_fail_without_incidental_errors(self):
        cases = [None, [], 0, 'CLEAN_STOP', {}, *[{k: v for k, v in self.journal.items() if k != missing}
                                                   for missing in self.journal]]
        for key, values in dict(completed=[-1, 5, True, False, 0.0, '0', None],
                                pending_step=[0, 1, False, 'none'],
                                checkpoint=['', '.', '../escape', '/a/../b', [], 1, None, '/bad\x00path'],
                                checkpoint_sha256=['sha', '', 'A' * 64, None, 0]).items():
            cases.extend(dict(self.journal, **{key: value}) for value in values)
        for journal in cases:
            with self.subTest(journal=journal):
                self.write('training-journal.json', journal)
                self.rejected()
        (self.output / 'training-journal.json').write_text('{broken')
        self.rejected()
        (self.output / 'training-journal.json').write_text('[' * 10000 + '0' + ']' * 10000)
        self.rejected()
        (self.output / 'training-journal.json').unlink()
        self.rejected()

    def test_outer_recipe_dataset_prompt_and_inputs_are_bound(self):
        for field in ('recipe', 'dataset', 'prompt'):
            value = deepcopy(self.run)
            value['identity'][field] = '0' * 64
            self.write('run.json', value)
            self.rejected('identity differs')
        self.write('run.json', self.run)
        for inputs in (None, [], {}, dict(self.inputs, order=list(reversed(self.order))),
                       dict(self.inputs, order=[False] + self.order[1:]),
                       dict(self.inputs, identity=dict(self.identity, recipe='0' * 64))):
            self.write('training-inputs.json', inputs)
            self.rejected()
        (self.output / 'training-inputs.json').unlink()
        self.rejected('Training inputs')

    def test_changed_current_dataset_recipe_prompt_and_order_fail(self):
        self.manifest['changed'] = True
        self.rejected('identity differs')
        del self.manifest['changed']
        self.config['seed'] = 43
        self.write('run.json', self.run)
        self.rejected('identity differs')
        self.config['seed'] = 42
        self.write('run.json', self.run)
        self.rows['training'].append({})
        self.rejected('sample order')
        self.rows['training'].pop()
        self.prompt.write_text('Changed policy')
        self.rejected('prompt differs')

    def test_metadata_inventory_and_publication_are_checked_before_deserialization(self):
        original = json.loads((self.checkpoint / 'manifest.json').read_text())
        cases = [None, [], {}, dict(original, identity=[]), dict(original, identity={'other': 'identity'}),
                 dict(original, files=None), dict(original, files={}), dict(original, files={'../state.pt': '0' * 64}),
                 dict(original, files={'state.pt': 'bad'}), dict(original, files={'missing': '0' * 64}),
                 *[dict(original, completed=value) for value in (-1, 5, True, 0.0, None, 1)]]
        for manifest in cases:
            with self.subTest(manifest=manifest), patch('torch.load', side_effect=AssertionError('Deserialization')):
                durable_json(self.checkpoint / 'manifest.json', manifest)
                manifest_hash = digest(self.checkpoint / 'manifest.json')
                durable_json(self.checkpoint / 'complete.json', dict(status='COMPLETE', manifest_sha256=manifest_hash))
                self.write('training-journal.json', dict(self.journal, checkpoint_sha256=manifest_hash))
                self.rejected()
        self.refresh_checkpoint(manifest=original)
        for complete in (None, [], {}, dict(status='PARTIAL'), dict(status='COMPLETE'),
                         dict(status='COMPLETE', manifest_sha256='0' * 64)):
            durable_json(self.checkpoint / 'complete.json', complete)
            self.rejected()

    def test_escaped_and_symbolic_checkpoint_paths_fail(self):
        alias = self.output / 'checkpoints' / 'alias'
        alias.symlink_to(self.checkpoint, target_is_directory=True)
        self.write('training-journal.json', dict(self.journal, checkpoint=str(alias)))
        self.rejected('symbolic links')
        alias.unlink()
        self.write('training-journal.json', self.journal)
        extra = self.checkpoint / 'link'
        extra.symlink_to(self.prompt)
        self.rejected('symbolic link')

    def test_file_hash_is_checked_in_worker_before_torch_load(self):
        (self.checkpoint / 'state.pt').write_bytes(b'tampered')
        before = self.snapshot()
        self.preflight()  # The command does not hash large checkpoint payloads.
        with patch('torch.load', side_effect=AssertionError('Deserialization')):
            with self.assertRaisesRegex(ValueError, 'file hash differs'):
                train(self.run['dataset'], self.output, self.config, resume=True)
        self.assertEqual(self.snapshot(), before)

    def test_cpu_trajectory_is_checked_before_gpu_imports(self):
        original = torch.load(self.checkpoint / 'state.pt', map_location='cpu', weights_only=False)
        cases = [None, [], {}, dict(original, order=list(reversed(self.order))),
                 dict(original, completed=True), dict(original, completed=1), dict(original, next_position=1),
                 dict(original, history=[{}]), dict(original, history={}), dict(original, random={}),
                 dict(original, optimizer=[]), dict(original, parameter_map=[]),
                 dict(original, diagnostics=[]), dict(original, diagnostics=None),
                 dict(original, optimizer=dict(original['optimizer'], param_groups=[None]))]
        for state in cases:
            with self.subTest(state_type=type(state).__name__):
                torch.save(state, self.checkpoint / 'state.pt')
                manifest = json.loads((self.checkpoint / 'manifest.json').read_text())
                manifest['files']['state.pt'] = digest(self.checkpoint / 'state.pt')
                self.refresh_checkpoint(manifest=manifest)
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    train(self.run['dataset'], self.output, self.config, resume=True)
                self.assertEqual(self.snapshot(), before)

    def test_metadata_fifos_are_rejected_without_reading_or_launch(self):
        import os
        for relative in ('training-journal.json', 'training-inputs.json',
                         'checkpoints/step-0/complete.json', 'checkpoints/step-0/manifest.json'):
            with self.subTest(relative=relative):
                path = self.output / relative
                content = path.read_bytes()
                path.unlink()
                os.mkfifo(path)
                try:
                    with patch('proxybench.training.runtime.launch', side_effect=AssertionError('launched')):
                        with self.assertRaises(ValueError):
                            resume_training(self.output)
                    with self.assertRaises(ValueError):
                        train(self.run['dataset'], self.output, self.config, resume=True)
                finally:
                    path.unlink()
                    path.write_bytes(content)

    def test_valid_resume_and_rejected_fresh_preserve_input_bytes(self):
        before = self.snapshot()
        accepted = self.preflight()
        state = load_state(accepted['checkpoint'], identity=accepted['identity'],
                           order=accepted['order'], completed=accepted['completed'])
        self.assertEqual(state['order'], self.order)
        with self.assertRaisesRegex(AssertionError, 'GPU import reached'):
            train(self.run['dataset'], self.output, self.config, resume=True)
        with self.assertRaisesRegex(ValueError, 'Training exists'):
            train(self.run['dataset'], self.output, self.config, resume=False)
        self.assertEqual(self.snapshot(), before)

    def test_cpu_history_and_random_state_validation_leave_rng_unchanged(self):
        from proxybench.training.trajectory import assert_same, random_state, validate_trajectory
        state = load_state(self.checkpoint, identity=self.identity, order=self.order, completed=0)
        row = dict(step=1, index=self.order[0], loss=1.0, response_tokens=2, update_seconds=.1)
        state.update(completed=1, next_position=1, history=[row])
        validate_trajectory(state, self.order, 1)
        cases = [dict(state, history=[{}]), dict(state, history=[dict(row, step=True)]),
                 dict(state, history=[dict(row, index=1-self.order[0])]),
                 dict(state, history=[dict(row, loss=float('nan'))]),
                 dict(state, history=[dict(row, response_tokens=False)]),
                 dict(state, history=[dict(row, update_seconds=-1)])]
        cases.extend(dict(state, random=dict(state['random'], **{key: None}))
                     for key in ('python', 'numpy', 'cpu', 'device'))
        before = random_state()
        for invalid in cases:
            with self.assertRaises(ValueError):
                validate_trajectory(invalid, self.order, 1)
            assert_same(before, random_state())

    def test_bounded_rejection_reconciles_charges_without_changing_provenance(self):
        (self.checkpoint / 'state.pt').write_bytes(b'tampered')
        ledger = self.output / 'resources.jsonl'
        ledger.write_text(json.dumps(dict(phase='train', status='CLEAN_STOP', elapsed_seconds=7)) + '\n')
        before = self.snapshot()
        def worker(*args, **kwargs):
            try:
                train(self.run['dataset'], self.output, self.config, resume=True)
            finally:
                with ledger.open('a') as stream:
                    stream.write(json.dumps(dict(phase='train', status='FAILED', elapsed_seconds=9)) + '\n')
        with patch('proxybench.training.runtime.launch', side_effect=worker):
            with self.assertRaisesRegex(ValueError, 'file hash differs'):
                resume_training(self.output)
        after = self.snapshot()
        self.assertEqual(json.loads(after['run.json'])['consumed_seconds'], 16)
        self.assertEqual({k: v for k, v in after.items() if k not in {'run.json', 'resources.jsonl'}},
                         {k: v for k, v in before.items() if k not in {'run.json', 'resources.jsonl'}})
