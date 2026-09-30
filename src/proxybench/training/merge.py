"""Merge adapter weights into a disposable base-model instance."""
import hashlib
import inspect
from pathlib import Path
from proxybench.execution.resources import durable_json, host_memory

def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_isfinite(tensor, *, chunk_elements=1024**2):
    """Inspect every value without allocating a full-sized device temporary."""
    import torch
    if not isinstance(chunk_elements, int) or chunk_elements < 1:
        raise ValueError('Finite-check chunk size must be a positive integer')
    pending = [tensor.detach()]
    while pending:
        part = pending.pop()
        if part.numel() <= chunk_elements:
            if not torch.isfinite(part).all().item():
                return False
        elif part.is_contiguous():
            flat = part.view(-1)
            for start in range(0, flat.numel(), chunk_elements):
                if not torch.isfinite(flat[start:start + chunk_elements]).all().item():
                    return False
        else:
            # Narrow creates views, including for transposed or strided weights.
            dimension = max(range(part.ndim), key=lambda axis: part.shape[axis])
            middle = part.shape[dimension] // 2
            pending.append(part.narrow(dimension, middle, part.shape[dimension] - middle))
            pending.append(part.narrow(dimension, 0, middle))
    return True

def merge_model(model, prompt, output, *, phase=None):
    """Inspect and merge one disposable instance with structural diagnostics."""
    import torch
    import time
    started = time.monotonic()
    # Only one GPU model exists. Host hashing copies one tensor at a time.
    largest = max(p.numel() * p.element_size() for p in model.parameters())
    required = 4 * largest + 1024**3
    available = host_memory()['available_bytes']
    if available < max(required, 4 * 1024**3):
        raise ValueError('Merge host-memory estimate does not fit')
    api = str(inspect.signature(model.merge_and_unload))
    if 'safe_merge' not in inspect.signature(model.merge_and_unload).parameters:
        raise ValueError('Installed merge API lacks safe_merge')
    base = model.get_base_model()
    adapted = {name: module for name, module in base.named_modules() if hasattr(module, 'lora_A')}
    device_required = 3 * max(module.get_base_layer().weight.numel() * 4 for module in adapted.values()) + 2 * 1024**3
    device_available = None
    if next(base.parameters()).is_cuda:
        device_available = torch.cuda.mem_get_info()[0]
        if device_available < device_required:
            raise ValueError('Merge device-memory estimate does not fit')
    before, samples, targets = {}, {}, set()
    for name, module in adapted.items():
        targets.add(name + '.weight')
        active = list(module.active_adapters)
        if active != ['default'] or list(module.lora_A) != ['default']:
            raise ValueError('Unexpected active adapter or multiple adapter application')
        weight = module.get_base_layer().weight
        a, b = module.lora_A['default'].weight, module.lora_B['default'].weight
        # Fixed positions include one calculation per adapted projection.
        points = [(0, 0), (weight.shape[0]//2, weight.shape[1]//2), (weight.shape[0]-1, weight.shape[1]-1)]
        values = []
        for row, col in points:
            delta = (b[row].float() @ a[:, col].float()) * module.scaling['default']
            values.append(dict(row=row, col=col, base=weight[row,col].item(),
                               delta_float32=delta.item(), expected_float32=(weight[row,col].float()+delta).item()))
        samples[name] = dict(dtype=str(weight.dtype), active=active, scaling=module.scaling['default'], points=values)
    if phase is not None:
        phase('hashing base weights')
    for name, p in base.named_parameters():
        if 'lora_' not in name:
            before[name.replace('.base_layer.', '.')] = tensor_hash(p)
    ids = torch.tensor([prompt], dtype=torch.long, device=next(base.parameters()).device)
    if phase is not None:
        phase('reference inference')
    with torch.inference_mode():
        reference = base(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
    if not torch.isfinite(reference).all():
        raise ValueError('Nonfinite reference scores')
    if phase is not None:
        phase('safe merge')
    merged = model.merge_and_unload(safe_merge=True)
    if any('lora_' in name for name, _ in merged.named_parameters()):
        raise ValueError('Adapter tensors remain after merge')
    after = {}
    if phase is not None:
        phase('checking merged weights')
    for name, p in merged.named_parameters():
        if not tensor_isfinite(p):
            raise ValueError('Nonfinite merged tensor')
        after[name] = tensor_hash(p)
    if set(before) != set(after):
        raise ValueError('Merged tensor inventory differs')
    changed = {name for name in before if before[name] != after[name]}
    if changed != targets:
        raise ValueError('Unexpected changed or unchanged merge tensors')
    for name, module in merged.named_modules():
        if name in samples:
            for point in samples[name]['points']:
                point['actual'] = module.weight[point['row'], point['col']].item()
    if phase is not None:
        phase('merged inference')
    with torch.inference_mode():
        candidate = merged(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
    if not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite merged scores')
    durable_json(Path(output) / 'merge.json', dict(api=api, host_required_bytes=required,
        host_available_bytes=available, device_required_bytes=device_required,
        device_available_bytes=device_available, samples=samples, before=before, after=after,
        changed=sorted(changed), tied_word_embeddings=merged.config.tie_word_embeddings,
        score_max_absolute_difference=(candidate-reference).abs().max().item(),
        score_mean_absolute_difference=(candidate-reference).abs().mean().item(),
        reference_argmax=reference.argmax().item(), merged_argmax=candidate.argmax().item(),
        elapsed_seconds=time.monotonic()-started))
    return merged
