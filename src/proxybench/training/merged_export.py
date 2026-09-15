"""Write a disposable merged model in bounded safetensors shards."""
import os
from pathlib import Path
import shutil
import uuid

from proxybench.execution.resources import durable_json, host_memory
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.optimization import tensor_hash
from proxybench.training.smoke import digest


def publish_merged(model, tokenizer, directory, expected_hashes, *, shard_bytes=256 * 1024**2):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    if model.config.architectures != ['Qwen3_5ForCausalLM']:
        raise ValueError('Expected the actual text-only architecture')
    parameters = dict(model.named_parameters())
    state = model.state_dict()
    aliases = {}
    for name in set(state) - set(parameters):
        if (name != 'lm_head.weight' or not model.config.tie_word_embeddings
                or state[name].data_ptr() != state['model.embed_tokens.weight'].data_ptr()):
            raise ValueError(f'Unexplained state tensor: {name}')
        aliases[name] = 'model.embed_tokens.weight'
    if set(parameters) != set(expected_hashes) or any('lora_' in name for name in parameters):
        raise ValueError('Merged parameter inventory differs')
    largest = max(p.numel() * p.element_size() for p in parameters.values())
    total = sum(p.numel() * p.element_size() for p in parameters.values())
    required = 3 * max(largest, shard_bytes) + 1024**3
    available = host_memory()['available_bytes']
    free_disk = shutil.disk_usage(directory.parent).free
    if available < max(required, 4 * 1024**3) or free_disk < 3 * total + 10 * 1024**3:
        raise ValueError('Serialization memory or disk estimate does not fit')
    claim = directory.with_name(directory.name + '.claim')
    with claim.open('x') as stream:
        stream.write(str(os.getpid()))
    temporary = directory.with_name(directory.name + '.incomplete-' + uuid.uuid4().hex)
    temporary.mkdir()
    shards, batch, size, inventory, weight_map = [], {}, 0, {}, {}

    def flush():
        nonlocal batch, size
        if not batch:
            return
        name = f'model-{len(shards) + 1:05d}.safetensors'
        save_file(batch, str(temporary / name), metadata={'format': 'pt'})
        weight_map.update({key: name for key in batch})
        shards.append(name)
        batch, size = {}, 0

    for name, parameter in parameters.items():
        nbytes = parameter.numel() * parameter.element_size()
        if size and size + nbytes > shard_bytes:
            flush()
        if not torch.isfinite(parameter).all() or tensor_hash(parameter) != expected_hashes[name]:
            raise ValueError(f'Merged value differs: {name}')
        inventory[name] = dict(shape=list(parameter.shape), dtype=str(parameter.dtype), sha256=expected_hashes[name])
        batch[name] = parameter.detach().to(device='cpu', copy=True).contiguous()
        size += nbytes
        if size >= shard_bytes:
            flush()
    flush()
    model.config.save_pretrained(temporary)
    model.generation_config.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    durable_json(temporary / 'model.safetensors.index.json', dict(metadata={'total_size': total}, weight_map=weight_map))
    observed = set()
    for name in shards:
        with safe_open(temporary / name, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key in observed or tensor_hash(stream.get_tensor(key)) != expected_hashes[key]:
                    raise ValueError('Serialized tensor differs or repeats')
                observed.add(key)
    if observed != set(expected_hashes):
        raise ValueError('Serialized tensor inventory differs')
    files = {p.name: digest(p) for p in temporary.iterdir() if p.is_file()}
    identity = dict(kind='disposable-merged-model', expected_hashes=expected_hashes,
                    tied_aliases=aliases, shard_bytes=shard_bytes, total_tensor_bytes=total,
                    required_host_bytes=required, available_host_bytes=available, free_disk_bytes=free_disk)
    durable_json(temporary / 'manifest.json', dict(identity=identity, tensors=inventory, files=files))
    for path in temporary.iterdir():
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    durable_json(temporary / 'complete.json', dict(status='COMPLETE', manifest_sha256=digest(temporary / 'manifest.json')))
    validate_checkpoint(temporary, identity, allow_temporary=True)
    temporary.rename(directory)
    fd = os.open(directory.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    claim.unlink()
    return validate_checkpoint(directory, identity)
