"""User-launched GPU phases for the fixed historical pilot."""

import json
import os
from pathlib import Path
import re
import time

from proxybench.execution.resources import durable_json
from proxybench.training.checkpoints import publish_adapter, validate_checkpoint
from proxybench.training.historical import binding, prepare_sequences
from proxybench.training.optimization import merge_model, tensor_hash
from proxybench.training.smoke import ResponseCollator, adapter_targets, digest, generation_input, read_json, require_same_adapter
from proxybench.training.trajectory import publish_state, require_clean_stop, restore_state, train_updates


def phase(name):
    print(f'Phase: {name}', flush=True)
    if os.environ.get('PROXYBENCH_PHASE_FILE'):
        durable_json(os.environ['PROXYBENCH_PHASE_FILE'], dict(phase=name))


def stopping():
    return bool(os.environ.get('PROXYBENCH_STOP_FILE') and Path(os.environ['PROXYBENCH_STOP_FILE']).exists())


def forecast_training_seconds(history, order, items):
    """Estimate unfinished updates from recent GPU timings and known token lengths."""
    completed = len(history)
    if completed < 36 or completed >= len(order):
        return None
    recent = history[-12:]
    recent_tokens = sum(items[row['index']]['combined_tokens'] for row in recent)
    pending_tokens = sum(items[index]['combined_tokens'] for index in order[completed:])
    # Leave room for checkpoint saves and final adapter publication.
    return 1.25 * sum(row['update_seconds'] for row in recent) / recent_tokens * pending_tokens + 120


def gpu_phase(root, name, output, config):
    if name not in {'training', 'original-export', 'trained-export'}:
        raise ValueError('Expected training or an evaluation export')
    # Import order is required by the pinned environment.
    phase('loading')
    begin = time.monotonic()
    from unsloth import FastLanguageModel
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import set_seed
    from unsloth_zoo.loss_utils import fused_linear_cross_entropy
    from proxybench.training.merged_export import publish_merged

    prepared = read_json((root/'prepared.json').read_bytes())
    run_state = read_json((root/'state.json').read_bytes())
    identity = dict(run=run_state['identity'], order=binding(prepared['order']))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or '3090' not in torch.cuda.get_device_name(0):
        raise ValueError('The pilot requires the local RTX 3090 with BF16 support')
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Path(config['model']).resolve()), max_seq_length=5120, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False, text_only=True,
        local_files_only=True, use_gradient_checkpointing='unsloth', random_state=42)
    if prepare_sequences(tokenizer, prepared['rows'], config) != prepared['items']:
        raise ValueError('Loaded tokenizer changes the frozen sequences')
    model.config.architectures = [type(model).__name__]
    if model.config.architectures != ['Qwen3_5ForCausalLM']:
        raise ValueError('Unexpected text model architecture')
    pristine = {n: tensor_hash(p) for n, p in model.named_parameters()}
    if name != 'original-export':
        targets, unsupported = adapter_targets(model.named_modules(), torch.nn.Linear)
        if unsupported:
            raise ValueError('Unsupported adapter projections')
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules='(?:'+'|'.join(re.escape(n) for n in targets)+')',
            lora_alpha=16, lora_dropout=0, bias='none', use_gradient_checkpointing='unsloth',
            random_state=42, max_seq_length=5120, temporary_location=str(output/'temporary-buffers'))
        trainables = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not trainables or any('lora_' not in n for n in trainables):
            raise ValueError('Unexpected trainable parameters')
        for target in targets:
            if not any(target+'.lora_' in n for n in trainables):
                raise ValueError('Missing adapter projection')
        for _, module in model.named_modules():
            if hasattr(module, 'lora_A') and (list(module.lora_A) != ['default'] or module.scaling['default'] != 2):
                raise ValueError('Unexpected adapter scale or adapter count')
        durable_json(output/'targets.json', dict(targets=targets, trainables=list(trainables)))
    base = model.get_base_model() if name != 'original-export' else model
    backbone, head = base.model, base.get_output_embeddings()
    collator = ResponseCollator(tokenizer.pad_token_id)

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        if free < config['limits']['device_margin_bytes']:
            raise ValueError('Free device memory fell below 2 GiB')
        return dict(allocated_bytes=torch.cuda.memory_allocated(), free_bytes=free, total_bytes=total)

    def loss(item):
        values = {k: v.to('cuda') for k, v in collator([item]).items()}
        hidden = backbone(input_ids=values['input_ids'], attention_mask=values['attention_mask'],
                          use_cache=False, return_dict=True).last_hidden_state
        return fused_linear_cross_entropy(hidden, head.weight, values['labels'])

    def weights():
        return get_peft_model_state_dict(model, save_embedding_layers=False)

    loading_seconds = time.monotonic()-begin
    durable_json(output/'loaded.json', dict(seconds=loading_seconds, memory=memory()))
    if stopping():
        return

    if name == 'training':
        FastLanguageModel.for_training(model, use_gradient_checkpointing='unsloth')
        model.train()
        base.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=1e-4, weight_decay=0)
        order = prepared['order']
        completed, history, diagnostics = 0, [], {}
        journal = root/'training-journal.json'
        if journal.exists():
            previous = read_json(journal.read_bytes())
            checkpoint = Path(previous['checkpoint'])
            completed = require_clean_stop(previous, checkpoint, identity)
            restored = restore_state(model, optimizer, checkpoint, identity=identity, order=order, completed=completed,
                                     get_weights=weights, set_weights=lambda s: set_peft_model_state_dict(model, s))
            history, diagnostics = restored['history'], restored['diagnostics']
            durable_json(output/'restored.json', dict(completed=completed, checkpoint=str(checkpoint), exact_state=True))

        def save(step, rows, clean):
            phase('saving')
            checkpoints = root/'checkpoints'
            suffix = '-stop-'+output.name if clean else ''
            path = checkpoints/f'step-{step:03}{suffix}'
            publish_state(model, optimizer, tokenizer, path, identity=identity, order=order,
                          completed=step, history=rows, get_weights=weights, diagnostics=diagnostics,
                          save_adapter=lambda p: publish_adapter(model, tokenizer, p, dict(identity, completed=step)))
            phase('training')
            return path

        def forecast_stop(step, rows):
            if config.get('profile') != 'historical-system-v1':
                return
            estimate = forecast_training_seconds(rows, order, prepared['items']['training'])
            if estimate is None:
                return
            deadline = min(float(os.environ['PROXYBENCH_PHASE_DEADLINE']),
                           float(os.environ['PROXYBENCH_TOTAL_DEADLINE']))
            if time.monotonic() + estimate >= deadline and not stopping():
                durable_json(os.environ['PROXYBENCH_STOP_FILE'],
                             dict(reason='TRAINING_FORECAST', completed=step,
                                  remaining_estimate_seconds=estimate))
                print(f'Training forecast exceeds remaining budget after update {step}. '
                      'Saving a clean stop.', flush=True)

        phase('training')
        result = train_updates(model, optimizer, order,
                    lambda i: (loss(prepared['items']['training'][i]), prepared['items']['training'][i]['response_tokens']),
                    journal, start=completed, history=history, stop=stopping, save=save,
                    checkpoint_interval=config.get('checkpoint_interval', config['updates']//4),
                    checkpoint_steps=({config['training_examples'], config['updates']}
                                      if config.get('profile') == 'historical-system-v1' else ()),
                    before_update=lambda step: phase('compilation') if step == 1 else phase('training') if step == 2 else None,
                    after_update=forecast_stop,
                    memory=memory, report=lambda message: print(message, flush=True))
        durable_json(output/'training-result.json', result)
        if result['status'] == 'CLEAN_STOP':
            return
        phase('saving')
        after = {n.replace('.base_layer.', '.'): tensor_hash(p)
                 for n, p in base.named_parameters() if 'lora_' not in n}
        if after != pristine:
            raise ValueError('Training changed a frozen original parameter')
        publish_adapter(model, tokenizer, root/'adapter', dict(identity, completed=config['updates']))
        durable_json(output/'complete.json', dict(status='COMPLETE', updates=config['updates'],
                     loading_seconds=loading_seconds, history=result['history'],
                     adapter_manifest_sha256=digest(root/'adapter/manifest.json')))
        return

    if name == 'trained-export':
        phase('restoring')
        validate_checkpoint(root/'adapter', dict(identity, completed=config['updates']))
        saved = load_file(str(root/'adapter/adapter_model.safetensors'))
        set_peft_model_state_dict(model, saved)
        require_same_adapter(saved, weights())
        from transformers import AutoTokenizer
        restored_tokenizer = AutoTokenizer.from_pretrained(root/'adapter', local_files_only=True)
        if prepare_sequences(restored_tokenizer, prepared['rows'], config) != prepared['items']:
            raise ValueError('Restored adapter tokenizer changes sequences')
        durable_json(output/'reload.json', dict(exact_tensors=True, adapter_sha256=digest(root/'adapter/manifest.json')))
    phase('serialization')
    if name == 'trained-export':
        FastLanguageModel.for_inference(model)
        model.eval()
        model = merge_model(model, generation_input(prepared['items']['training'][0]), output)
        expected = read_json((output/'merge.json').read_bytes())['after']
    else:
        expected = pristine
    publish_merged(model, tokenizer, output/'model', expected)
    durable_json(output/'complete.json', dict(status='COMPLETE', loading_seconds=loading_seconds))
