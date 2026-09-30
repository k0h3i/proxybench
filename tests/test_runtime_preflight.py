"""CPU checks for missing or changed native runtime files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from proxybench.extraction.runtime import model_server, runtime_identity
from proxybench.training.adapters import digest


class RuntimePreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / 'runtime'
        self.runtime.mkdir()
        self.server = self.runtime / 'llama-server'
        self.server.write_bytes(b'synthetic server')
        self.runtime_library = self.runtime / 'libggml.so'
        self.runtime_library.write_bytes(b'synthetic runtime library')
        self.external_library = self.root / 'libcudart.so.12'
        self.external_library.write_bytes(b'synthetic external library')
        self.manifest = self.runtime / 'runtime-manifest.json'
        self.manifest.write_text(json.dumps(dict(
            source_commit='synthetic-revision',
            files={path.name: digest(path) for path in (self.server, self.runtime_library)},
        )))
        self.config = dict(
            runtime_manifest=str(self.manifest), server=str(self.server),
            source_commit='synthetic-revision',
            external_libraries={str(self.external_library): digest(self.external_library)},
        )

    def assert_missing_file(self, path):
        with self.assertRaises(ValueError) as raised:
            runtime_identity(self.config)
        self.assertIn('Missing', str(raised.exception))
        self.assertIn(str(path), str(raised.exception))
        self.assertIn('docs/preparation.md', str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_missing_manifest_names_file_and_preparation_guide(self):
        self.manifest.unlink()
        self.assert_missing_file(self.manifest)

    def test_missing_bound_server_names_file_and_preparation_guide(self):
        self.server.unlink()
        self.assert_missing_file(self.server)

    def test_missing_unbound_server_names_file_and_preparation_guide(self):
        missing = self.runtime / 'other-server'
        self.config['server'] = str(missing)
        self.assert_missing_file(missing)

    def test_missing_runtime_library_names_file_and_preparation_guide(self):
        self.runtime_library.unlink()
        self.assert_missing_file(self.runtime_library)

    def test_changed_external_library_is_rejected(self):
        self.external_library.write_bytes(b'changed library')
        with self.assertRaisesRegex(ValueError, 'External runtime library differs from its pinned hash'):
            runtime_identity(self.config)

    def test_changed_runtime_library_is_rejected(self):
        self.runtime_library.write_bytes(b'changed library')
        with self.assertRaisesRegex(ValueError, 'Runtime file differs from its manifest'):
            runtime_identity(self.config)

    def test_identity_does_not_import_model_packages_or_start_processes(self):
        program = '''
import json
import sys
from unittest.mock import patch

class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'cupy', 'nvidia', 'pynvml'}:
            raise AssertionError('Model package import: ' + fullname)

sys.meta_path.insert(0, NoModelImports())
with patch('subprocess.Popen', side_effect=AssertionError('Runtime process launch')):
    from proxybench.extraction.runtime import runtime_identity
    print(runtime_identity(json.loads(sys.argv[1])))
'''
        result = subprocess.run(
            [sys.executable, '-c', program, json.dumps(self.config)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), digest(self.manifest))

    def test_server_rejects_missing_library_before_model_import_or_launch(self):
        self.external_library.unlink()
        with patch.dict(sys.modules, {'transformers': None}), \
             patch('proxybench.extraction.runtime.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'Missing runtime file'):
                with model_server(self.root / 'output', self.config):
                    self.fail('Missing runtime library passed authentication')
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
