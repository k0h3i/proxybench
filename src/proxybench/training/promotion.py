"""Publish only the selected model formats into retained storage."""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from proxybench.runstate import Run, atomic_json, file_hash


def promote(run_dir, models_dir, name):
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', name) or name.endswith('.')
            or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('Model name must be a short portable filename')
    root = Path(models_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    # Atomic directory creation reserves this name against another publisher.
    reservation = root / ('.' + name + '.publishing')
    reservation.mkdir()
    stage = None
    try:
        if destination.exists():
            raise FileExistsError('Model destination exists; choose a new name')
        with Run(run_dir) as run:
            adapter = run.path / 'adapter'
            gguf = run.path / 'exports' / 'conversion' / 'model-bf16.gguf'
            completion = gguf.parent / 'complete.json'
            if not gguf.is_file():
                raise ValueError('Export the selected adapter before promotion')
            value = json.loads(completion.read_text()) if completion.is_file() else {}
            if (value.get('status') != 'COMPLETE' or value.get('exact_transformed_payloads') is not True
                    or value.get('source_commit') != '329b6160f513915f1c607dbfae3d5ce864a64a4f'
                    or value.get('sha256') != file_hash(gguf)):
                raise ValueError('A verified complete conversion is required before promotion')
            names = ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
                     'tokenizer_config.json', 'chat_template.jinja']
            sources = [adapter / name for name in names] + [gguf]
            if any(not p.is_file() or p.is_symlink() for p in sources):
                raise ValueError('Model files must be complete regular files')
            config = json.loads((adapter / 'adapter_config.json').read_text())
            export_identity = json.loads((run.path / 'exports' / 'export-identity.json').read_text())
            if (export_identity.get('adapter') != file_hash(adapter / 'adapter_model.safetensors')
                    or export_identity.get('adapter_config') != file_hash(adapter / 'adapter_config.json')):
                raise ValueError('Export does not belong to the selected adapter')
            tokenizer = {n: file_hash(adapter / n) for n in names if n not in {'adapter_model.safetensors', 'adapter_config.json'}}
            source = json.loads((run.path / 'exports' / 'export-source.json').read_text())
            publication = json.loads((run.path / 'exports' / 'export-publication.json').read_text())
            expected = dict(schema_version='export-publication-v1', status='COMPLETE',
                adapter_sha256=export_identity['adapter'], adapter_config_sha256=export_identity['adapter_config'],
                tokenizer_files=tokenizer, recipe_sha256=export_identity['recipe'],
                merged_manifest_sha256=value['source_manifest_sha256'], gguf_sha256=value['sha256'],
                conversion_complete_sha256=file_hash(completion))
            if (publication != expected or export_identity.get('tokenizer') != tokenizer
                    or source != dict(identity=export_identity, merged_manifest_sha256=value['source_manifest_sha256'])):
                raise ValueError('The adapter and converted model publication chain differs')
            from proxybench.training.runtime import BASE_MODEL, BASE_REVISION
            if config.get('base_model_name_or_path') != BASE_MODEL or config.get('revision') != BASE_REVISION:
                raise ValueError('Adapter metadata must name the pinned portable base')
            stage = Path(tempfile.mkdtemp(prefix='.' + name + '-', dir=root))
            (stage / 'adapter').mkdir()
            hashes = {}
            for source in sources:
                relative = Path('adapter') / source.name if source.parent == adapter else Path('model-bf16.gguf')
                target = stage / relative
                before = file_hash(source)
                shutil.copyfile(source, target)
                if file_hash(target) != before or file_hash(source) != before:
                    raise ValueError('Model bytes changed during promotion')
                hashes[str(relative)] = before
            atomic_json(stage / 'model-info.json', dict(
                schema_version='model-info-v1', name=name, base_model=BASE_MODEL,
                base_revision=BASE_REVISION, files=hashes,
                input_identity=run.state.get('training_identity', run.state['identity']),
                load_validation='pending-user-launched-gpu-tests'))
            if destination.exists():
                raise FileExistsError('Model destination appeared during promotion')
            os.rename(stage, destination)
            stage = None
        return dict(status='COMPLETE', model=str(destination))
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        reservation.rmdir()
