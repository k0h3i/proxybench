"""Convert a validated temporary model to BF16 GGUF and inspect its tensors."""
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from proxybench.execution.resources import durable_json, host_memory
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.adapters import digest, read_json
from proxybench.training.runtime import phase

SOURCE_COMMIT = '329b6160f513915f1c607dbfae3d5ce864a64a4f'
SOURCE_ARCHIVE_SHA256 = '9d46c7ce4da17fa584df7d906209ff79f2b827a97b9fe57cd9652b29c55a6ca5'
SOURCE_ARCHIVE_NAME = '.proxybench-source.tar.gz'
SOURCE_ARCHIVE_ROOT = 'llama.cpp-b10909-mix-bea84f7'


def validate_converter_source(source):
    source = Path(source).resolve()
    archive = source / SOURCE_ARCHIVE_NAME
    if archive.is_symlink() or not archive.is_file() or digest(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError('Converter source archive checksum differs')
    expected = {SOURCE_ARCHIVE_NAME: SOURCE_ARCHIVE_SHA256}
    directories = set()
    with tarfile.open(archive) as bundle:
        seen = set()
        for member in bundle:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] != SOURCE_ARCHIVE_ROOT or path in seen):
                raise ValueError('Converter source archive inventory differs')
            seen.add(path)
            relative = PurePosixPath(*path.parts[1:])
            if relative == PurePosixPath('.'):
                if not member.isdir():
                    raise ValueError('Converter source archive root differs')
                continue
            name = relative.as_posix()
            if member.isdir():
                directories.add(name)
            elif member.isfile() and name != SOURCE_ARCHIVE_NAME:
                with bundle.extractfile(member) as stream:
                    expected[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
            else:
                raise ValueError('Converter source archive has an unsupported file')
            directories.update(parent.as_posix() for parent in relative.parents
                               if parent != PurePosixPath('.'))
    actual_files, actual_directories = set(), set()
    for path in source.rglob('*'):
        name = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError('Converter source contains a symbolic link')
        if path.is_dir():
            actual_directories.add(name)
        elif path.is_file():
            actual_files.add(name)
            if name not in expected or digest(path) != expected[name]:
                raise ValueError(f'Converter source file differs: {name}')
        else:
            raise ValueError(f'Converter source has an unsupported file: {name}')
    if actual_files != set(expected) or actual_directories != directories:
        raise ValueError('Converter source extracted inventory differs')
    # Python can reuse modules already in memory instead of the verified source.
    for name, module in tuple(sys.modules.items()):
        if name in ('gguf', 'conversion') or name.startswith(('gguf.', 'conversion.')):
            origin = getattr(module, '__file__', None)
            module_root = source / 'gguf-py' if name.split('.')[0] == 'gguf' else source
            module_path = module_root.joinpath(*name.split('.'))
            candidates = (module_path.with_suffix('.py'), module_path / '__init__.py')
            if (not origin or Path(origin).resolve() not in candidates
                    or Path(origin).resolve().relative_to(source).as_posix() not in expected):
                raise ValueError(f'Converter import is outside the verified source: {name}')
            if hasattr(module, '__path__') and list(module.__path__) != [str(module_path)]:
                raise ValueError(f'Converter package search path differs: {name}')
    return dict(source_commit=SOURCE_COMMIT, archive_sha256=SOURCE_ARCHIVE_SHA256,
                files=expected)


def convert(model, output, config):
    phase('conversion')
    model, output = Path(model), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = validate_checkpoint(model)
    temporary = output/'model.incomplete.gguf'
    target = output/'model-bf16.gguf'
    proof = output/'publication.json'
    if target.exists():
        if not proof.exists():
            raise ValueError('GGUF exists without a verified publication record')
        saved = read_json(proof.read_bytes())
        if (saved['source_manifest_sha256'] != digest(model/'manifest.json')
                or saved['sha256'] != digest(target)):
            raise ValueError('Interrupted GGUF publication differs')
        durable_json(output/'complete.json', saved)
        return target
    largest = max(math.prod(v['shape'])*(2 if v['dtype'] == 'torch.bfloat16' else 4) for v in manifest['tensors'].values())
    if host_memory()['available_bytes'] < max(4*1024**3, 8*largest+1024**3):
        raise ValueError('Conversion memory estimate does not fit')
    if shutil.disk_usage(output).free < max(config['minimum_disk_bytes'], 3*manifest['identity']['total_tensor_bytes']+10*1024**3):
        raise ValueError('Conversion disk estimate does not fit')
    source = Path(config['converter_source']).resolve()
    source_identity = validate_converter_source(source)
    # These are disposable files from an interrupted conversion of the same input.
    temporary.unlink(missing_ok=True)
    scratch = output/'conversion-temp'
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', TMPDIR=str(scratch.resolve()),
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join((str(source), str(source/'gguf-py'))))
    command = [sys.executable, str(source/'convert_hf_to_gguf.py'), str(model), '--outfile', str(temporary),
               '--outtype', 'bf16', '--use-temp-file', '--no-mtp']
    durable_json(output/'command.json', dict(command=command, source_manifest_sha256=digest(model/'manifest.json'),
                                           converter_archive_sha256=source_identity['archive_sha256']))
    # Inherit the worker's streams and process group. Supervision owns conversion too.
    subprocess.run(command, env=env, check=True)
    phase('payload-inspection')
    validate_converter_source(source)
    sys.path[:0] = [str(source), str(source/'gguf-py')]
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        import numpy as np
        import gguf
        from safetensors import safe_open
        from conversion import get_model_class
        converter = object.__new__(get_model_class('Qwen3_5ForCausalLM', mmproj=False))
    finally:
        sys.dont_write_bytecode = previous_bytecode
    converter.hparams = read_json((model/'config.json').read_bytes())
    converter.block_count = converter.hparams['num_hidden_layers']
    converter.tensor_map = gguf.get_tensor_name_map(converter.model_arch, converter.block_count)
    converter.fuse_qkv = False
    converter.fuse_gate_up_exps = False
    reader = gguf.GGUFReader(temporary)
    actual = {t.name: t for t in reader.tensors}
    if len(actual) != len(reader.tensors):
        raise ValueError('Duplicate converted tensor')
    index = read_json((model/'model.safetensors.index.json').read_bytes())['weight_map']
    checked = {}
    for name, shard in index.items():
        with safe_open(model/shard, framework='pt', device='cpu') as stream:
            original = stream.get_tensor(name).float()
            block = int(name.split('.')[2]) if name.startswith('model.layers.') else None
            for new_name, transformed in converter.modify_tensors(original, name, block):
                if new_name in checked or new_name not in actual:
                    raise ValueError('Unexpected converted tensor inventory')
                tensor = actual[new_name]
                if (tensor.tensor_type not in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32)
                        or list(tensor.shape) != list(reversed(transformed.shape))):
                    raise ValueError('Converted tensor precision or shape differs')
                wanted = gguf.quants.quantize(transformed.numpy(), tensor.tensor_type)
                expected = hashlib.sha256(memoryview(np.ascontiguousarray(wanted)).cast('B')).hexdigest()
                found = hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()
                if found != expected:
                    raise ValueError('Converted payload differs from the pinned transformation')
                flat = tensor.data.reshape(-1)
                for start in range(0, len(flat), 65536):
                    if not np.isfinite(gguf.quants.dequantize(flat[start:start+65536], tensor.tensor_type)).all():
                        raise ValueError('Nonfinite converted tensor')
                checked[new_name] = dict(dtype=tensor.tensor_type.name, shape=[int(v) for v in tensor.shape], sha256=found)
                del wanted, transformed
            del original
    if set(checked) != set(actual):
        raise ValueError('Incomplete converted tensor inspection')
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    file_hash = digest(temporary)
    result = dict(status='COMPLETE', path=str(target.resolve()), sha256=file_hash,
                 source_manifest_sha256=digest(model/'manifest.json'), tensors=checked,
                 exact_transformed_payloads=True, source_commit=SOURCE_COMMIT,
                 converter_archive_sha256=source_identity['archive_sha256'])
    durable_json(proof, result)
    temporary.rename(target)
    durable_json(output/'complete.json', result)
    return target
