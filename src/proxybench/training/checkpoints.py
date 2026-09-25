"""Publish validated adapter checkpoints without overwriting prior artifacts."""
import os
from pathlib import Path
import uuid

from proxybench.execution.resources import durable_json


def validate_checkpoint(directory, expected=None, *, allow_temporary=False):
    from proxybench.training.adapters import digest, read_json
    directory = Path(directory)
    if '.incomplete-' in directory.name and not allow_temporary:
        raise ValueError('Checkpoint publication is incomplete')
    complete = read_json((directory / 'complete.json').read_bytes())
    if complete.get('status') != 'COMPLETE':
        raise ValueError('Checkpoint is incomplete')
    if digest(directory / 'manifest.json') != complete['manifest_sha256']:
        raise ValueError('Checkpoint manifest hash differs')
    manifest = read_json((directory / 'manifest.json').read_bytes())
    if expected is not None and manifest['identity'] != expected:
        raise ValueError('Checkpoint identity differs')
    files = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if files != set(manifest['files']) | {'manifest.json', 'complete.json'}:
        raise ValueError('Checkpoint file inventory differs')
    for name, sha256 in manifest['files'].items():
        if digest(directory / name) != sha256:
            raise ValueError(f'Checkpoint file hash differs: {name}')
    return manifest


def publish_adapter(model, tokenizer, directory, identity=None):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from proxybench.training.adapters import digest, require_same_adapter
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # The claim also prevents concurrent writers from replacing a completed directory.
    claim = directory.with_name(directory.name + '.claim')
    with claim.open('x') as stream:
        stream.write(str(os.getpid()))
    temporary = directory.with_name(directory.name + '.incomplete-' + uuid.uuid4().hex)
    temporary.mkdir()
    model.save_pretrained(str(temporary), safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(str(temporary))
    saved = load_file(str(temporary / 'adapter_model.safetensors'))
    if not saved or any('lora_' not in name for name in saved):
        raise ValueError('Saved state contains unintended parameters')
    require_same_adapter(saved, get_peft_model_state_dict(model, save_embedding_layers=False))
    inventory = {name: {'shape': list(t.shape), 'dtype': str(t.dtype)} for name, t in saved.items()}
    files = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()}
    durable_json(temporary / 'manifest.json', dict(identity=identity or {}, tensors=inventory, files=files))
    for path in temporary.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    durable_json(temporary / 'complete.json', dict(status='COMPLETE', manifest_sha256=digest(temporary / 'manifest.json')))
    validate_checkpoint(temporary, identity or {}, allow_temporary=True)
    if directory.exists():
        raise FileExistsError(directory)
    temporary.rename(directory)
    fd = os.open(directory.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    claim.unlink()
    return validate_checkpoint(directory, identity or {})
