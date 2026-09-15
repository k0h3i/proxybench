"""Run the fixed preservation, reload, and Python measurement phases."""
import hashlib
import inspect
from pathlib import Path

from proxybench.execution.resources import durable_json, host_memory
from proxybench.extraction.measured_generation import generate
from proxybench.training.checkpoints import publish_adapter, validate_checkpoint
from proxybench.training.smoke import digest, generation_input, read_json, require_same_adapter

PANEL = (1, 6, 2, 7, 12, 14)
RELOAD = PANEL[:3]
TIMING = (1, 7, 14)


def admit(estimate_seconds, *, used_seconds, total_seconds=5400, phase_seconds=1800, reserve_seconds=0):
    estimate = estimate_seconds * 1.25
    if estimate > phase_seconds or estimate + reserve_seconds > total_seconds - used_seconds:
        raise ValueError('Budget admission failed')
    return dict(estimated_seconds=estimate_seconds, allowance=1.25, admitted_seconds=estimate,
                prior_seconds=used_seconds, reserve_seconds=reserve_seconds)


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def merge_model(model, prompt, output):
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
    for name, p in base.named_parameters():
        if 'lora_' not in name:
            before[name.replace('.base_layer.', '.')] = tensor_hash(p)
    ids = torch.tensor([prompt], dtype=torch.long, device=next(base.parameters()).device)
    with torch.inference_mode():
        reference = base(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
    if not torch.isfinite(reference).all():
        raise ValueError('Nonfinite reference scores')
    merged = model.merge_and_unload(safe_merge=True)
    if any('lora_' in name for name, _ in merged.named_parameters()):
        raise ValueError('Adapter tensors remain after merge')
    after = {}
    for name, p in merged.named_parameters():
        if not torch.isfinite(p).all():
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
    with torch.inference_mode():
        candidate = merged(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
    if not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite merged scores')
    durable_json(Path(output) / 'merge.json', dict(api=api, host_required_bytes=required,
        host_available_bytes=available, samples=samples, before=before, after=after,
        changed=sorted(changed), tied_word_embeddings=merged.config.tie_word_embeddings,
        score_max_absolute_difference=(candidate-reference).abs().max().item(),
        score_mean_absolute_difference=(candidate-reference).abs().mean().item(),
        reference_argmax=reference.argmax().item(), merged_argmax=candidate.argmax().item(),
        elapsed_seconds=time.monotonic()-started))
    return merged


def run_phase(args, *, config, rows, manifest, provenance, items, bounds, model, tokenizer,
              train, memory, phase, inference_mode):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import set_seed
    identity = dict(configuration=config, revision=provenance['revision'], dataset_sha256=config['dataset_sha256'],
                    examples=manifest['examples'], order=bounds['order'], steps=30, pilot_updates_included=False)
    if config.get('panel') != list(PANEL):
        raise ValueError('Optimization panel differs')
    if args.phase == 'train-save':
        phase('training')
        set_seed(config['seed'])
        train(bounds['order'])
        phase('saving_adapter')
        publish_adapter(model, tokenizer, args.output / 'adapter', identity)
    else:
        checkpoint = validate_checkpoint(args.adapter, identity)
        state = load_file(str(args.adapter / 'adapter_model.safetensors'))
        set_peft_model_state_dict(model, state)
        require_same_adapter(state, get_peft_model_state_dict(model, save_embedding_layers=False))
        durable_json(args.output / 'reload-integrity.json', dict(exact_tensors=True,
                     checkpoint_manifest_sha256=digest(args.adapter / 'manifest.json'), tensors=checkpoint['tensors']))
    inference_mode()
    operations = {}
    for name, module in model.named_modules():
        for attribute in ('chunk_gated_delta_rule', 'recurrent_gated_delta_rule',
                          'fused_recurrent_gated_delta_rule', 'causal_conv1d_fn', 'causal_conv1d_update'):
            function = getattr(module, attribute, None)
            if function is not None:
                operations[name + '.' + attribute] = dict(module=getattr(function, '__module__', None),
                                                         name=getattr(function, '__qualname__', str(function)))
    durable_json(args.output / 'runtime.json', dict(operations=operations, use_cache=True,
        devices=sorted({str(p.device) for p in model.parameters()}),
        dtypes=sorted({str(p.dtype) for p in model.parameters()}), memory=memory()))

    def request(index, label, *, maximum=1792, forced=False, deadline=240, streaming=True):
        value = generate(model, tokenizer, generation_input(items[index]), args.output / f'{label}-{index}.json',
                         index=index, maximum=maximum, forced=forced, deadline=deadline, streaming=streaming,
                         expected=read_json(rows[index]['messages'][1]['content']))
        print(__import__('json').dumps(dict(label=label, index=index, status=value['status'],
              format_valid=value['format_valid'], timing=value['timing'])), flush=True)
        return value

    def warm(index, label):
        value = request(index, label + '-warmup', maximum=64, forced=True)
        if value['status'] != 'PROBE_COMPLETE':
            raise ValueError('Warmup did not complete')

    def natural(index, label, deadline=240):
        warm(index, label)
        return request(index, label, deadline=deadline)

    def probes(label):
        for index in TIMING:
            warm(index, label)
            for repeat in range(2):
                value = request(index, f'{label}-probe{repeat}', maximum=256, forced=True)
                if value['status'] != 'PROBE_COMPLETE':
                    raise ValueError('Fixed-length probe did not complete')

    phase(args.phase)
    if args.phase == 'train-save':
        for index in RELOAD:
            value = natural(index, 'reference')
            if value['status'] != 'COMPLETE':
                raise ValueError('In-memory reference did not complete')
    elif args.phase == 'reload-panel':
        for index in RELOAD:
            value = natural(index, 'reference')
            expected = read_json((args.reference / f'reference-{index}.json').read_bytes())
            if value['status'] != 'COMPLETE' or value['token_ids'] != expected['token_ids']:
                durable_json(args.output / 'determinism-failure.json', dict(index=index, exact_tensors=True,
                             status='DETERMINISM_DISCREPANCY'))
                raise ValueError(f'Reloaded generation differs for row {index}')
        durable_json(args.output / 'reload-complete.json', dict(exact_tensors=True, matching_generations=True))
        for index in PANEL[3:]:
            if natural(index, 'reference')['status'] != 'COMPLETE':
                raise ValueError('Unmerged panel reference did not complete')
    elif args.phase == 'diagnostics':
        with model.disable_adapter():
            probes('disabled')
        probes('enabled')
        warm(1, 'observer')
        request(1, 'observer-stream', maximum=256, forced=True)
        request(1, 'observer-return', maximum=256, forced=True, streaming=False)
    elif args.phase in ('merged', 'final-candidate'):
        model = merge_model(model, generation_input(items[14]), args.output)
        model.eval()
        if args.phase == 'merged':
            probes('merged')
        indices = (14, 1, 6, 2, 7, 12) if args.phase == 'merged' else tuple(i for i in range(15) if i not in PANEL)
        for index in indices:
            value = natural(index, 'candidate', deadline=60)
            if value['status'] != 'COMPLETE' or not value['format_valid']:
                durable_json(args.output / 'candidate-failed.json', dict(index=index, status=value['status']))
                return
    elif args.phase == 'final-reference':
        for index in range(15):
            if index not in PANEL and natural(index, 'reference')['status'] != 'COMPLETE':
                raise ValueError('Final reference did not complete')
    durable_json(args.output / 'complete.json', dict(status='COMPLETE', phase=args.phase, memory=memory()))
