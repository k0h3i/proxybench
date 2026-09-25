"""Shared adapter preparation and exact tensor checks."""
import hashlib
from pathlib import Path
from proxybench.training.labels import read_json
from proxybench.training.sequences import pad_sequence

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

class ResponseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(f['input_ids']) for f in features)
        padded = [pad_sequence(f, length, self.pad_token_id) for f in features]
        return {k: torch.tensor([p[k] for p in padded], dtype=torch.long)
                for k in ('input_ids', 'attention_mask', 'labels')}

def generation_input(item):
    return item['input_ids'][:item['input_tokens']]

def adapter_targets(named_modules, linear_type):
    targets, unsupported = [], []
    covered = set()
    for name, module in named_modules:
        parts = name.split('.')
        if any(p in parts for p in ('visual', 'vision', 'vision_tower', 'lm_head', 'embed_tokens')):
            continue
        if 'layers' not in parts:
            continue
        branch = next((p for p in ('self_attn', 'linear_attn', 'mlp') if p in parts), None)
        if branch and isinstance(module, linear_type):
            targets.append(name)
            covered.add(branch)
        elif branch and name.endswith(('proj', 'proj_qkv', 'proj_z', 'proj_a', 'proj_b')):
            unsupported.append(name)
    if covered != {'self_attn', 'linear_attn', 'mlp'}:
        raise ValueError(f'Incomplete language adapter coverage: {sorted(covered)}')
    return targets, unsupported

def require_same_adapter(expected, actual):
    import torch
    if set(expected) != set(actual):
        raise ValueError('Adapter tensor names differ')
    for name, tensor in expected.items():
        other = actual[name]
        if tensor.dtype != other.dtype or not torch.equal(tensor.cpu(), other.cpu()):
            raise ValueError(f'Adapter tensor differs: {name}')
