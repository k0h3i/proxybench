"""Bounded Qwen baseline or disposable synthetic adapter probe, never filing training."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

from proxybench.execution.runner import write_json


def phase(name):
    path = Path(os.environ['PROXYBENCH_PHASE_FILE'])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'phase': name, 'started_at': time.time()}))
    temporary.replace(path)


def load_model(config, model_path, output):
    import torch
    # Triton builds a small C launcher even when CUDA kernels are prebuilt.
    if not os.environ.get('CC') and not (shutil.which('gcc') or shutil.which('clang')):
        import ziglang
        compiler = output.resolve() / 'triton-cc'
        compiler.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -m ziglang cc "$@"\n')
        compiler.chmod(0o755)
        os.environ['CC'] = str(compiler)
    import triton
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'cuda':
        raise RuntimeError('The Qwen preparation runtime requires a working Triton CUDA backend')
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    runtime = {'fast_path_available': modeling_qwen3_5.is_fast_path_available,
               'triton_backend': target.backend, 'triton_arch': target.arch,
               'compiler': os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang'),
               'packages': {name: importlib.metadata.version(name) for name in
                            ('torch', 'transformers', 'bitsandbytes')}}
    for name in ('flash-linear-attention', 'fla-core', 'causal-conv1d', 'triton'):
        try:
            runtime['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime['packages'][name] = None
    write_json(output / 'runtime.json', runtime)
    if not runtime['fast_path_available']:
        raise RuntimeError('Qwen optimized GPU routines are unavailable. Install the pinned GPU dependencies before execution.')

    if config['model'] != 'Qwen/Qwen3.5-9B' or config['revision'] != 'c202236235762e1c871ad0ccb60c8ee5ba337b9a':
        raise ValueError('The probe requires the selected pinned model')
    provenance = json.loads((model_path / 'provenance-with-weights.json').read_text())
    if provenance['revision'] != config['revision']:
        raise ValueError('Local model revision differs')
    for name, entry in provenance['files'].items():
        with (model_path / name).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256']:
            raise ValueError('Pinned model file hash differs: ' + name)
    phase('loading')
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     llm_int8_skip_modules=['visual', 'lm_head'])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, device_map={'': 0},
        dtype=torch.bfloat16, quantization_config=quantization, attn_implementation='sdpa')
    torch.cuda.synchronize()
    visual = [(n, p) for n, p in model.named_parameters() if '.visual.' in n]
    if not visual:
        raise ValueError('The canonical multimodal model did not load its vision parameters')
    linear_layers = [module for module in model.modules()
                     if isinstance(module, modeling_qwen3_5.Qwen3_5GatedDeltaNet)]
    if not linear_layers or any(
        module.causal_conv1d_fn is not modeling_qwen3_5.causal_conv1d_fn
        or module.causal_conv1d_update is not modeling_qwen3_5.causal_conv1d_update
        or module.recurrent_gated_delta_rule is not modeling_qwen3_5.fused_recurrent_gated_delta_rule
        or module.chunk_gated_delta_rule is not modeling_qwen3_5.chunk_gated_delta_rule
        for module in linear_layers
    ):
        raise RuntimeError('Qwen layers do not bind the optimized GPU routines')
    write_json(output / 'optimized-layers.json', {'verified_layer_count': len(linear_layers)})
    write_json(output / 'loading.json', {'elapsed_seconds': time.monotonic() - start,
                                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                       'vision_loaded': True,
                                       'vision_parameter_storage_bytes': sum(p.numel() * p.element_size() for _, p in visual),
                                       'vision_modules': [n for n, _ in visual],
                                       'model_class': type(model).__name__})
    return model, tokenizer


def synthetic_probe(model, tokenizer, config, output, synthetic_path):
    import torch
    from torch.utils.checkpoint import checkpoint
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    phase('synthetic_optimizer')
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    data = json.loads(synthetic_path.read_text())
    if data.get('kind') != 'synthetic-only-v1' or data.get('contains_filing_labels') is not False:
        raise ValueError('Only explicitly synthetic source-target data is allowed')
    inputs, labels, mask = data['input_ids'], data['labels'], data['attention_mask']
    if len(inputs) != config['context_tokens'] or len(labels) != len(inputs) or len(mask) != len(inputs):
        raise ValueError('Synthetic sequence must exercise the declared maximum length')
    if any(m != 1 for m in mask) or any(y != -100 and y != x for x, y in zip(inputs, labels)):
        raise ValueError('Invalid synthetic labels or unexercised padded context')
    supervised = [i for i, y in enumerate(labels) if y != -100]
    if not supervised or supervised[0] <= 0 or tokenizer.eos_token_id not in [labels[i] for i in supervised]:
        raise ValueError('Loss mask lacks response supervision or termination')
    if any(y != -100 for y in labels[:data['input_tokens']]) or any(y == -100 for y in labels[data['input_tokens']:]):
        raise ValueError('Prompt or response loss mask differs from the frozen boundary')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={'use_reentrant': False})
    # Keep frozen visual and language weights at the proposed compute precision.
    # prepare_model_for_kbit_training promotes nonquantized weights to float32.
    for parameter in model.parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    targets = [name for name, module in model.named_modules()
               if '.language_model.' in name and name.rsplit('.', 1)[-1] in config['adapter_suffixes']]
    if not targets or any('.visual.' in name for name in targets):
        raise ValueError('Adapter targets do not identify language-only modules')
    model = get_peft_model(model, LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
                                           lora_dropout=0.0, target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any('lora_' not in n or '.language_model.' not in n or '.visual.' in n for n in trainable):
        raise ValueError('Unintended trainable parameters')
    before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    frozen_samples = {n: p.detach().flatten()[:16].cpu().clone() for n, p in frozen.items()}
    write_json(output / 'adapter-targets.json', {'modules': targets, 'trainable': list(trainable),
                                                'trainable_parameters': sum(p.numel() for p in trainable.values()),
                                                'vision_frozen': True})
    batch = {k: torch.tensor([data[k]], device='cuda') for k in ('input_ids', 'attention_mask')}
    target = torch.tensor(labels, device='cuda')
    positions = torch.tensor([i - 1 for i in supervised], device='cuda')
    expected = target[positions + 1]
    optimizer = torch.optim.AdamW(trainable.values(), lr=config['learning_rate'], weight_decay=0.0)
    model.train()
    model.config.use_cache = False
    losses, gradient_checks = [], []
    base = model.get_base_model()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config['gradient_accumulation']):
            hidden = base.model(**batch, use_cache=False).last_hidden_state[0]
            # Checkpoint vocabulary projection in small chunks to preserve the complete
            # response loss without keeping a context-by-vocabulary logits tensor.
            def loss_chunk(states, answers):
                logits = base.lm_head(states).float()
                return torch.nn.functional.cross_entropy(logits, answers, reduction='sum')
            loss = sum(checkpoint(loss_chunk, hidden[positions[i:i + 128]], expected[i:i + 128],
                                  use_reentrant=False) for i in range(0, len(supervised), 128)) / len(supervised)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite synthetic loss')
            losses.append(float(loss.detach()))
            (loss / config['gradient_accumulation']).backward()
        gradients = [p.grad for p in trainable.values() if p.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in gradients)
        nonzero = any(bool(torch.count_nonzero(g)) for g in gradients)
        if not finite or not nonzero:
            raise ValueError('Adapter gradients must be finite with at least one nonzero gradient')
        gradient_checks.append({'finite': finite, 'any_nonzero': nonzero})
        optimizer.step()
    torch.cuda.synchronize()
    changed = [n for n, p in trainable.items() if not torch.equal(before[n], p.detach().cpu())]
    if not changed or any(p.requires_grad for p in frozen.values()):
        raise ValueError('No adapter update or changed frozen status')
    if any(not torch.equal(frozen_samples[n], p.detach().flatten()[:16].cpu()) for n, p in frozen.items()):
        raise ValueError('Frozen base parameter sample changed')
    write_json(output / 'optimizer.json', {'steps': 2, 'losses': losses, 'gradient_checks': gradient_checks,
                                         'changed_adapter_tensors': changed, 'supervised_tokens': len(supervised),
                                         'context_tokens': len(inputs), 'gradient_accumulation': config['gradient_accumulation'],
                                         'elapsed_seconds': time.monotonic() - start,
                                         'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                         'frozen_base_samples_equal': True, 'padding_supervised_tokens': 0})
    phase('adapter_reload')
    torch.cuda.reset_peak_memory_stats()
    reload_started = time.monotonic()
    model.save_pretrained(output / 'disposable-adapter', safe_serialization=True)
    saved = load_file(output / 'disposable-adapter/adapter_model.safetensors')
    # Clear the active tensors so equality also establishes that reload happened.
    with torch.no_grad():
        for p in trainable.values():
            p.zero_()
    set_peft_model_state_dict(model, saved)
    reloaded = get_peft_model_state_dict(model)
    if set(saved) != set(reloaded) or any(not torch.equal(v, reloaded[k].cpu()) for k, v in saved.items()):
        raise ValueError('Reloaded adapter tensors differ from saved tensors')
    write_json(output / 'reload.json', {'tensors_equal': True, 'tensor_count': len(saved),
                                       'elapsed_seconds': time.monotonic() - reload_started,
                                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                       'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    phase('adapter_generation')
    model.eval()
    from proxybench.extraction.local_model import generate_one
    generate_one(model, tokenizer, synthetic_path.parent / 'prompt.txt', synthetic_path.parent / 'input.json',
                 output / 'generation', config | {'max_new_tokens': 128})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'synthetic'])
    parser.add_argument('configuration', type=Path)
    parser.add_argument('model_path', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if 'PROXYBENCH_PHASE_FILE' not in os.environ:
        parser.error('Run this probe under proxybench.execution.resources')
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.configuration.read_text())
    from transformers import set_seed
    set_seed(config['seed'])
    write_json(args.output / 'environment.json', {name: importlib.metadata.version(name) for name in
                                                ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'tokenizers')})
    write_json(args.output / 'configuration.json', config)
    try:
        model, tokenizer = load_model(config, args.model_path, args.output)
        if args.mode == 'baseline':
            from proxybench.extraction.local_model import generate_one
            directories = sorted(p for p in args.inputs.iterdir() if p.is_dir())
            if len(directories) != 6 or [p.name for p in directories] != [f'cal-{i:03d}' for i in range(1, 7)]:
                raise ValueError('The initial baseline requires the same six development inputs')
            model.eval()
            for item in directories:
                phase('inference_' + item.name)
                generate_one(model, tokenizer, item / 'prompt.txt', item / 'input.json', args.output / item.name, config)
        else:
            synthetic_probe(model, tokenizer, config, args.output, args.inputs)
    except Exception as error:
        write_json(args.output / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
