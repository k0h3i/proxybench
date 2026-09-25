import json
from pathlib import Path
import tempfile
import unittest

from proxybench.runstate import Run, atomic_json, file_hash
from proxybench.training.promotion import promote
from proxybench.training.runtime import BASE_MODEL, BASE_REVISION


class PromotionTests(unittest.TestCase):
    def test_only_selected_formats_are_published_and_overwrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'run'
            with Run(path, create=True, identity={'model': 'one'}, config={}) as run:
                adapter = run.path / 'adapter'
                adapter.mkdir()
                for name in ('adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
                    (adapter / name).write_text('test')
                atomic_json(adapter / 'adapter_config.json', dict(base_model_name_or_path=BASE_MODEL, revision=BASE_REVISION))
                identity = dict(
                    adapter=file_hash(adapter / 'adapter_model.safetensors'),
                    adapter_config=file_hash(adapter / 'adapter_config.json'), recipe='recipe',
                    tokenizer={n: file_hash(adapter / n) for n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')})
                atomic_json(path / 'exports' / 'export-identity.json', identity)
                target = path / 'exports' / 'conversion' / 'model-bf16.gguf'
                target.parent.mkdir()
                target.write_text('GGUF test')
                atomic_json(target.parent / 'complete.json', dict(status='COMPLETE', exact_transformed_payloads=True,
                    source_commit='329b6160f513915f1c607dbfae3d5ce864a64a4f', source_manifest_sha256='merged', sha256=file_hash(target)))
                atomic_json(path / 'exports' / 'export-source.json', dict(identity=identity, merged_manifest_sha256='merged'))
                atomic_json(path / 'exports' / 'export-publication.json', dict(schema_version='export-publication-v1', status='COMPLETE',
                    adapter_sha256=identity['adapter'], adapter_config_sha256=identity['adapter_config'],
                    tokenizer_files=identity['tokenizer'], recipe_sha256='recipe', merged_manifest_sha256='merged',
                    gguf_sha256=file_hash(target), conversion_complete_sha256=file_hash(target.parent / 'complete.json')))
                (adapter / 'private-run-history.json').write_text('excluded')
            result = promote(path, root / 'models', 'Example-4B')
            selected = Path(result['model'])
            self.assertEqual(set(p.name for p in selected.iterdir()), {'adapter', 'model-bf16.gguf', 'model-info.json'})
            self.assertFalse((selected / 'adapter' / 'private-run-history.json').exists())
            with self.assertRaises(FileExistsError):
                promote(path, root / 'models', 'Example-4B')
            proof = path / 'exports' / 'export-publication.json'
            saved = json.loads(proof.read_text())
            atomic_json(proof, {**saved, 'merged_manifest_sha256': 'another-model'})
            with self.assertRaisesRegex(ValueError, 'publication chain'):
                promote(path, root / 'models', 'WrongModel-4B')
            atomic_json(proof, saved)
            (path / 'adapter' / 'adapter_model.safetensors').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'selected adapter'):
                promote(path, root / 'models', 'Another-4B')
            with self.assertRaises(ValueError):
                promote(path, root / 'models', '../escape')
