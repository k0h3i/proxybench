"""Convert a validated temporary model to BF16 GGUF and inspect its tensors."""
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from proxybench.execution.resources import durable_json, host_memory
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.adapters import digest, read_json
from proxybench.training.runtime import phase

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
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    top = subprocess.check_output(['git', '-C', str(source), 'rev-parse', '--show-toplevel'], text=True).strip()
    clean = subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD'], check=False).returncode == 0
    if (revision != '329b6160f513915f1c607dbfae3d5ce864a64a4f'
            or Path(top).resolve() != source or not clean):
        raise ValueError('Converter source revision differs')
    # These are disposable files from an interrupted conversion of the same input.
    temporary.unlink(missing_ok=True)
    scratch = output/'conversion-temp'
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', TMPDIR=str(scratch.resolve()))
    command = [sys.executable, str(source/'convert_hf_to_gguf.py'), str(model), '--outfile', str(temporary),
               '--outtype', 'bf16', '--use-temp-file', '--no-mtp']
    durable_json(output/'command.json', dict(command=command, source_manifest_sha256=digest(model/'manifest.json')))
    # Inherit the worker's streams and process group. Supervision owns conversion too.
    subprocess.run(command, env=env, check=True)
    phase('payload-inspection')
    sys.path[:0] = [str(source), str(source/'gguf-py')]
    import numpy as np
    import gguf
    from safetensors import safe_open
    from conversion import get_model_class
    converter = object.__new__(get_model_class('Qwen3_5ForCausalLM', mmproj=False))
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
                 exact_transformed_payloads=True, source_commit='329b6160f513915f1c607dbfae3d5ce864a64a4f')
    durable_json(proof, result)
    temporary.rename(target)
    durable_json(output/'complete.json', result)
    return target
