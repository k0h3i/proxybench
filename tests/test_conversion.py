"""Authenticate the whole converter tree before executing its source."""
import io
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proxybench.training import conversion
from proxybench.training.adapters import digest


class ConverterSourceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.source = Path(self.folder.name) / 'source'
        self.source.mkdir()
        archive = self.source / conversion.SOURCE_ARCHIVE_NAME
        self.files = {
            'convert_hf_to_gguf.py': b'raise AssertionError("must not execute")\n',
            'conversion/__init__.py': b'raise AssertionError("must not import")\n',
            'gguf-py/gguf/__init__.py': b'raise AssertionError("must not import")\n',
            'README.md': b'release source\n',
        }
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, content in self.files.items():
                member = tarfile.TarInfo(conversion.SOURCE_ARCHIVE_ROOT + '/' + name)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
                target = self.source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        self.accepted_hash = digest(archive)
        self.pin = patch.object(conversion, 'SOURCE_ARCHIVE_SHA256', self.accepted_hash)
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def test_accepts_complete_extraction_without_git(self):
        result = conversion.validate_converter_source(self.source)
        self.assertEqual(result['source_commit'], '329b6160f513915f1c607dbfae3d5ce864a64a4f')
        self.assertEqual(result['archive_sha256'], self.accepted_hash)
        self.assertEqual(set(result['files']), {*self.files, conversion.SOURCE_ARCHIVE_NAME})

    def test_rejects_altered_source_and_non_source_files(self):
        for name, original in self.files.items():
            with self.subTest(name=name):
                path = self.source / name
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'file differs'):
                    conversion.validate_converter_source(self.source)
                path.write_bytes(original)

    def test_rejects_missing_file(self):
        (self.source / 'conversion/__init__.py').unlink()
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            conversion.validate_converter_source(self.source)

    def test_rejects_added_imports_bytecode_and_executables(self):
        for name in ('sitecustomize.py', 'conversion/extra.py', 'gguf-py/gguf/extra.py',
                     'conversion/extra.pyc', 'conversion/extra.so', 'untracked-tool'):
            with self.subTest(name=name):
                path = self.source / name
                path.write_bytes(b'added executable source')
                with self.assertRaisesRegex(ValueError, 'file differs'):
                    conversion.validate_converter_source(self.source)
                path.unlink()

    def test_rejects_symbolic_link_even_to_identical_bytes(self):
        path = self.source / 'README.md'
        path.unlink()
        original = Path(self.folder.name) / 'original'
        original.write_bytes(self.files['README.md'])
        path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            conversion.validate_converter_source(self.source)

    def test_rejects_wrong_archive_checksum(self):
        with (self.source / conversion.SOURCE_ARCHIVE_NAME).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'archive checksum'):
            conversion.validate_converter_source(self.source)

    def test_rejects_missing_archive(self):
        (self.source / conversion.SOURCE_ARCHIVE_NAME).unlink()
        with self.assertRaisesRegex(ValueError, 'archive checksum'):
            conversion.validate_converter_source(self.source)

    def test_rejects_cached_unverified_import(self):
        for name in ('gguf', 'gguf.reader', 'conversion', 'conversion.other'):
            with self.subTest(name=name), patch.dict(sys.modules, {name: SimpleNamespace(__file__='/unverified/module.py')}):
                with self.assertRaisesRegex(ValueError, 'import is outside'):
                    conversion.validate_converter_source(self.source)

    def test_rejects_cached_package_with_unexpected_search_path(self):
        module = SimpleNamespace(__file__=str(self.source / 'conversion/__init__.py'),
                                 __path__=[str(self.source / 'conversion'), '/unverified'])
        with patch.dict(sys.modules, conversion=module):
            with self.assertRaisesRegex(ValueError, 'search path differs'):
                conversion.validate_converter_source(self.source)

    def test_rejects_source_before_subprocess_or_converter_imports(self):
        (self.source / 'sitecustomize.py').write_text('raise AssertionError("must not execute")')
        manifest = dict(tensors={'tensor': dict(shape=[1], dtype='torch.bfloat16')},
                        identity=dict(total_tensor_bytes=2))
        with patch.object(conversion, 'validate_checkpoint', return_value=manifest), \
                patch.object(conversion, 'phase'), \
                patch.object(conversion, 'host_memory', return_value=dict(available_bytes=1 << 60)), \
                patch.object(conversion.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 60)), \
                patch.object(conversion.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'file differs'):
                conversion.convert(Path(self.folder.name) / 'model', Path(self.folder.name) / 'out',
                                   dict(converter_source=str(self.source), minimum_disk_bytes=0))
            run.assert_not_called()

    def test_rechecks_source_before_payload_inspection_imports(self):
        model = Path(self.folder.name) / 'model'
        model.mkdir()
        (model / 'manifest.json').write_text('{}')
        manifest = dict(tensors={'tensor': dict(shape=[1], dtype='torch.bfloat16')},
                        identity=dict(total_tensor_bytes=2))

        def changed_during_conversion(command, *, env, check):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
            (self.source / 'conversion/extra.py').write_text('raise AssertionError("must not import")')

        with patch.object(conversion, 'validate_checkpoint', return_value=manifest), \
                patch.object(conversion, 'phase'), \
                patch.object(conversion, 'host_memory', return_value=dict(available_bytes=1 << 60)), \
                patch.object(conversion.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 60)), \
                patch.object(conversion.subprocess, 'run', side_effect=changed_during_conversion) as run:
            with self.assertRaisesRegex(ValueError, 'file differs'):
                conversion.convert(model, Path(self.folder.name) / 'out',
                                   dict(converter_source=str(self.source), minimum_disk_bytes=0))
            run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
