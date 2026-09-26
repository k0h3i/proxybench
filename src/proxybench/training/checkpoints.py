"""Publish validated adapter checkpoints without overwriting prior artifacts."""
from contextlib import nullcontext
import os
from pathlib import Path
import re
import uuid

from proxybench.execution.resources import durable_json


def plain_path(value, label='Checkpoint path'):
    if not isinstance(value, (str, Path)) or not str(value) or '\x00' in str(value):
        raise ValueError(f'{label} is invalid')
    path = Path(value).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'{label} cannot contain parent traversal or symbolic links')
    return path


def read_object(path, label):
    from proxybench.training.adapters import read_json
    try:
        source = plain_path(path, label)
        if not source.is_file() or source.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f'{label} must be a regular metadata file of at most 16 MiB')
        value = read_json(source.read_bytes())
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f'{label} is missing or invalid') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


def require_hash(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{label} must be a SHA-256 hash')


def require_position(value, total, label):
    if type(value) is not int or not 0 <= value <= total:
        raise ValueError(f'{label} must be an integer from 0 to {total}')


def validate_checkpoint(directory, expected=None, *, allow_temporary=False, hash_files=True):
    """Validate publication metadata, then optionally hash the full inventory."""
    from proxybench.training.adapters import digest
    directory = plain_path(directory)
    if '.incomplete-' in directory.name and not allow_temporary:
        raise ValueError('Checkpoint publication is incomplete')
    complete = read_object(directory / 'complete.json', 'Checkpoint completion record')
    if complete.get('status') != 'COMPLETE':
        raise ValueError('Checkpoint is incomplete')
    require_hash(complete.get('manifest_sha256'), 'Checkpoint manifest hash')
    manifest = read_object(directory / 'manifest.json', 'Checkpoint manifest')
    if digest(directory / 'manifest.json') != complete['manifest_sha256']:
        raise ValueError('Checkpoint manifest hash differs')
    if not isinstance(manifest.get('identity'), dict):
        raise ValueError('Checkpoint identity must be an object')
    if expected is not None and manifest['identity'] != expected:
        raise ValueError('Checkpoint identity differs')
    if not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise ValueError('Checkpoint file inventory is missing or invalid')
    for name, sha256 in manifest['files'].items():
        relative = Path(name)
        if (not relative.parts or relative.is_absolute() or '..' in relative.parts
                or relative.as_posix() != name or name in {'manifest.json', 'complete.json'}
                or '\x00' in name):
            raise ValueError('Checkpoint inventory contains an invalid path')
        require_hash(sha256, 'Checkpoint file hash')
    paths = list(directory.rglob('*'))
    if any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in paths):
        raise ValueError('Checkpoint inventory contains a symbolic link or unsupported file')
    files = {p.relative_to(directory).as_posix() for p in paths if p.is_file()}
    if files != set(manifest['files']) | {'manifest.json', 'complete.json'}:
        raise ValueError('Checkpoint file inventory differs')
    if hash_files:
        for name, sha256 in manifest['files'].items():
            if digest(directory / name) != sha256:
                raise ValueError(f'Checkpoint file hash differs: {name}')
    return manifest


def publish_adapter(model, tokenizer, directory, identity=None, *, measurements=None):
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
    measure = measurements.stage if measurements else lambda _: nullcontext()
    with measure('checkpoint_serialization'):
        model.save_pretrained(str(temporary), safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(str(temporary))
    with measure('checkpoint_readback'):
        saved = load_file(str(temporary / 'adapter_model.safetensors'))
    with measure('checkpoint_comparison'):
        if not saved or any('lora_' not in name for name in saved):
            raise ValueError('Saved state contains unintended parameters')
        require_same_adapter(saved, get_peft_model_state_dict(model, save_embedding_layers=False))
    inventory = {name: {'shape': list(t.shape), 'dtype': str(t.dtype)} for name, t in saved.items()}
    with measure('checkpoint_hashing'):
        files = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()}
    with measure('checkpoint_synchronization'):
        durable_json(temporary / 'manifest.json', dict(identity=identity or {}, tensors=inventory, files=files))
        for path in temporary.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        durable_json(temporary / 'complete.json', dict(status='COMPLETE', manifest_sha256=digest(temporary / 'manifest.json')))
    with measure('checkpoint_hashing'):
        validate_checkpoint(temporary, identity or {}, allow_temporary=True)
    if directory.exists():
        raise FileExistsError(directory)
    with measure('checkpoint_publication'):
        temporary.rename(directory)
        fd = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        claim.unlink()
    with measure('checkpoint_hashing'):
        return validate_checkpoint(directory, identity or {})
