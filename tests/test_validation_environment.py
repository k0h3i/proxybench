"""Model load preparation rejects dependencies inside disposable artifacts."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from proxybench.training.validation import require_project_environment, validate_runtime


class ValidationEnvironmentTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        stack.enter_context(patch('pathlib.Path.cwd', return_value=self.root))
        stack.enter_context(patch.dict('os.environ', {}, clear=True))
        stack.enter_context(patch('sys.prefix', str(self.root / '.venv')))
        stack.enter_context(patch('sys.base_prefix', str(self.root / 'python')))
        stack.enter_context(patch('sys.executable', str(self.root / '.venv/bin/python')))
        self.training = {'base_path': str(self.root / 'artifacts/models/Qwen3.5-4B'),
                         'base_manifest': str(self.root / 'configs/base-model.json')}
        self.inference = {'model': str(self.root / 'artifacts/models/ProxyType-4B/model.gguf'),
                          'tokenizer': str(self.root / 'artifacts/models/ProxyType-4B/adapter'),
                          'library_path': str(self.root / 'runtime')}

    def check_environment(self):
        require_project_environment(self.training, self.inference)

    def test_project_environment_and_retained_models_are_supported(self):
        self.check_environment()

    def test_other_environment_and_artifact_interpreters_are_rejected(self):
        with patch('sys.prefix', str(self.root / 'other')):
            with self.assertRaisesRegex(ValueError, 'project .venv'):
                self.check_environment()
        for name in ('prefix', 'base_prefix', 'executable'):
            with self.subTest(name=name), patch('sys.' + name, str(self.root / 'artifacts/environments/old')):
                with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
                    self.check_environment()

    def test_old_dependencies_in_environment_are_rejected(self):
        for name in ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH',
                     'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'CC', 'CXX',
                     'PROXYBENCH_RUNTIME', 'PROXYBENCH_CUDA_LIB',
                     'PROXYBENCH_CONVERTER_SOURCE', 'CUDA_HOME', 'CUDA_PATH'):
            with self.subTest(name=name), patch.dict('os.environ', {name: str(self.root / 'artifacts/environments/old')}):
                with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
                    self.check_environment()

    def test_symlink_to_old_compiler_and_artifact_venv_are_rejected(self):
        old = self.root / 'artifacts/environments/old'
        old.mkdir(parents=True)
        link = self.root / 'compiler'
        link.symlink_to(old / 'cc')
        with patch.dict('os.environ', {'CC': str(link)}):
            with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
                self.check_environment()
        (self.root / '.venv').symlink_to(old, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
            self.check_environment()

    def test_import_search_and_resolved_compiler_paths_are_rejected(self):
        old = str(self.root / 'artifacts/environments/old')
        with patch('sys.path', [*sys.path, old]):
            with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
                self.check_environment()
        with patch.dict('os.environ', {'CXX': 'c++'}), patch('shutil.which', return_value=old):
            with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
                self.check_environment()

    def test_resolved_configuration_cannot_use_old_dependencies(self):
        for configuration, name in ((self.training, 'base_manifest'), (self.training, 'converter_source'),
                                    (self.inference, 'server'), (self.inference, 'runtime_manifest'),
                                    (self.inference, 'library_path')):
            with self.subTest(name=name), patch.dict(configuration, {name: str(self.root / 'artifacts/runs/old')}):
                with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
                    self.check_environment()
        with patch.dict(self.inference, {'external_libraries': {str(self.root / 'artifacts/old/lib.so'): 'hash'}}):
            with self.assertRaisesRegex(ValueError, 'artifact dependencies'):
                self.check_environment()

    def test_rejection_precedes_output_and_worker_launch(self):
        from types import SimpleNamespace
        args = SimpleNamespace(run_dir=self.root / 'result', training_config='training', config='inference')
        with patch('proxybench.extraction.runtime.load_config', side_effect=[self.training, self.inference]), \
             patch('proxybench.training.runtime.launch') as launch, \
             patch('sys.prefix', str(self.root / 'artifacts/environments/old')):
            with self.assertRaises(ValueError):
                validate_runtime(args)
            launch.assert_not_called()
            self.assertFalse(args.run_dir.exists())


    def test_base_directory_rejects_old_artifacts_and_symlinked_paths(self):
        from proxybench.training.runtime import base_model_path
        for path in ('artifacts/environments/old', 'artifacts/runs/old', 'artifacts/models/other'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'retained Qwen3.5-4B'):
                base_model_path({'base_path': str(self.root / path)})
        old = self.root / 'artifacts/runs/old'
        old.mkdir(parents=True)
        base = Path(self.training['base_path'])
        base.parent.mkdir()
        base.symlink_to(old, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            self.check_environment()
        base.unlink()
        source = self.root / 'converter'
        source.symlink_to(old, target_is_directory=True)
        with patch.dict(self.training, {'converter_source': str(source)}):
            with self.assertRaisesRegex(ValueError, 'symbolic links'):
                self.check_environment()
