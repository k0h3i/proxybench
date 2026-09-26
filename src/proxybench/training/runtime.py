"""Portable training, adapter export, and bounded inference commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

from proxybench.execution.resources import durable_json
from proxybench.extraction.runtime import load_config
from proxybench.training.adapters import ResponseCollator, adapter_targets, digest, require_same_adapter
from proxybench.training.preparation import prepare_sequences

BASE_MODEL = 'Qwen/Qwen3.5-4B'
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BASE_MANIFEST_SHA256 = '199f252959418e8e7c006b14c7280db752d1e8b10b3428571bb10f9b28407641'


def binding(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tokenizer_identity(directory):
    root = Path(directory)
    names = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
             'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model', 'merges.txt', 'vocab.json', 'vocab.txt')
    if not (root/'tokenizer.json').is_file() or not (root/'tokenizer_config.json').is_file():
        raise ValueError('Tokenizer files are incomplete')
    return {name: digest(root/name) for name in names if (root/name).is_file()}


def resource_floor(run):
    from proxybench.execution.resources import reconcile_captures
    observed = reconcile_captures(run.path)
    run.state['consumed_seconds'] = max(run.state['consumed_seconds'], observed)
    run.save()
    return observed


def phase(name, *, measurements=None, **fields):
    if measurements is None:
        print(f'Phase: {name}', flush=True)
    else:
        measurements.phase(name, **fields)
    if os.environ.get('PROXYBENCH_PHASE_FILE'):
        durable_json(os.environ['PROXYBENCH_PHASE_FILE'], dict(phase=name))


def stopping():
    return bool(os.environ.get('PROXYBENCH_STOP_FILE') and Path(os.environ['PROXYBENCH_STOP_FILE']).exists())


def require_plain_path(value):
    """Reject paths that depend on symbolic links or parent traversal."""
    path = Path(value).expanduser().absolute()
    if '..' in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Model paths cannot contain parent traversal or symbolic links')
    return path


def base_model_path(config):
    root = require_plain_path(config['base_path'])
    artifacts = Path.cwd().resolve() / 'artifacts'
    if root.is_relative_to(artifacts) and root != artifacts / 'models/Qwen3.5-4B':
        raise ValueError('Use the retained Qwen3.5-4B base model directory')
    return root


def base_snapshot(config):
    """Authenticate the complete local base snapshot without network access."""
    from proxybench.training.labels import read_json
    if config['model_id'] != BASE_MODEL or config['model_revision'] != BASE_REVISION:
        raise ValueError('Unsupported base model or revision')
    root = base_model_path(config)
    manifest_path = require_plain_path(config['base_manifest'])
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_MANIFEST_SHA256:
        raise ValueError('Base model manifest differs from its pinned identity')
    manifest = read_json(raw)
    if (manifest.get('model_id') != BASE_MODEL or manifest.get('model_revision') != BASE_REVISION
            or not isinstance(manifest.get('files'), dict) or not manifest['files']):
        raise ValueError('Unsupported base model manifest')
    expected = manifest['files']
    for name, expected_hash in expected.items():
        relative = Path(name)
        if (not relative.parts or relative.is_absolute() or '..' in relative.parts
                or relative.as_posix() != name or relative.parts[0] == '.cache'
                or not isinstance(expected_hash, str) or not re.fullmatch('[0-9a-f]{64}', expected_hash)):
            raise ValueError('Invalid base model manifest file')
        path = require_plain_path(root / relative)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Base model snapshot has a missing or escaped file')
        if digest(path) != expected_hash:
            raise ValueError('Base model file differs from its pinned hash')
    metadata = {'.cache/huggingface/.gitignore', '.cache/huggingface/CACHEDIR.TAG',
                f'.cache/huggingface/trees/{BASE_REVISION}.json'}
    metadata.update(f'.cache/huggingface/download/{name}{suffix}'
                    for name in expected for suffix in ('.metadata', '.lock'))
    allowed = expected.keys() | metadata
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Base model snapshot cannot contain symbolic links')
        if path.is_dir():
            continue
        if not path.is_file() or path.relative_to(root).as_posix() not in allowed:
            raise ValueError('Base model snapshot contains an unexpected file')
    return str(root)


def load_base(config, snapshot=None):
    from unsloth import FastLanguageModel
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Training requires a CUDA GPU with BF16 support')
    snapshot = snapshot or base_snapshot(config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=snapshot, max_seq_length=config['context_tokens'], dtype=torch.bfloat16,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False, text_only=True,
        local_files_only=True, use_gradient_checkpointing='unsloth', random_state=config['seed'])
    model.config.architectures = [type(model).__name__]
    if model.config.architectures != ['Qwen3_5ForCausalLM']:
        raise ValueError('Unexpected base model architecture')
    return model, tokenizer


def attach_adapter(model, config, output):
    from unsloth import FastLanguageModel
    import torch
    targets, unsupported = adapter_targets(model.named_modules(), torch.nn.Linear)
    if unsupported:
        raise ValueError('Unsupported adapter projections')
    model = FastLanguageModel.get_peft_model(
        model, r=config['rank'], target_modules='(?:'+'|'.join(re.escape(n) for n in targets)+')',
        lora_alpha=config['alpha'], lora_dropout=config['dropout'], bias='none',
        use_gradient_checkpointing='unsloth', random_state=config['seed'],
        max_seq_length=config['context_tokens'], temporary_location=str(output/'temporary-buffers'))
    if any('lora_' not in n for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError('Unexpected trainable parameters')
    # Saving uses a portable base identity. Loading still pins base_snapshot above.
    model.peft_config['default'].base_model_name_or_path = BASE_MODEL
    model.peft_config['default'].revision = BASE_REVISION
    return model


def train(dataset, run_dir, config, resume=False):
    """Worker API. Resume only a trusted local checkpoint after a clean stop."""
    from proxybench.training.measurements import MeasurementRecorder, event_line
    measurements = MeasurementRecorder(emit=lambda event: print(event_line(event), flush=True))
    try:
        result = _train(dataset, run_dir, config, resume, measurements)
    except BaseException:
        measurements.finish('FAILED')
        raise
    else:
        measurements.finish('COMPLETE' if result['status'] == 'TRAINED' else 'CLEAN_STOP',
                            result_published=True, resume_eligible=result['status'] == 'CLEAN_STOP')
        return result
    finally:
        # Captures are accounting evidence. Never rewrite training provenance on rejection.
        capture = os.environ.get('PROXYBENCH_PHASE_FILE')
        if capture:
            durable_json(Path(capture).parent / 'training-measurements.json', measurements.snapshot())


def _train(dataset, run_dir, config, resume, measurements):
    from proxybench.training.resume import training_preflight
    from proxybench.training.trajectory import load_state, restore_loaded_state, publish_state, train_updates, require_clean_stop
    from proxybench.training.measurements import capture_runtime_identity, compare_runtime_identity
    from proxybench.training.runtime_facts import optimizer_identity, device_and_kernels, DeviceComputeTimer
    phase('preparation', measurements=measurements)
    with measurements.stage('preparation'):
        accepted = training_preflight(dataset, run_dir, config, resume=resume)
        rows, identity, order = accepted['rows'], accepted['identity'], accepted['order']
        completed = accepted['completed']
        saved = (load_state(accepted['checkpoint'], identity=identity, order=order, completed=completed)
                 if resume else None)
        prior_runtime = saved['diagnostics']['runtime_identity'] if saved else None
        origin_runtime = saved['diagnostics']['origin_runtime_identity'] if saved else None
    epochs = [len(rows['training']) * epoch for epoch in range(1, config['epochs'] + 1)]
    counters = dict(global_step=completed, sample_position=completed, planned_updates=len(order),
                    epoch_boundaries=epochs, batch_size=config['batch_size'],
                    gradient_accumulation_steps=config['accumulation'],
                    effective_batch_size=config['batch_size'] * config['accumulation'],
                    resume_eligible=bool(resume),
                    last_checkpoint=(dict(path=str(accepted['checkpoint']), global_step=completed) if resume else None))
    phase('preparation', measurements=measurements, **counters)
    # CPU checkpoint checks finish before imports that initialize the GPU stack.
    from unsloth import FastLanguageModel
    import torch
    from transformers import set_seed, AutoTokenizer
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from unsloth_zoo.loss_utils import fused_linear_cross_entropy
    from proxybench.training.checkpoints import publish_adapter
    from proxybench.training.merge import tensor_hash
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    import shutil
    if shutil.disk_usage(output).free < config['minimum_disk_bytes']:
        raise ValueError('Training disk reserve is insufficient')
    set_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with measurements.stage('preparation'):
        snapshot = base_snapshot(config)
        prepared_tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        items = prepare_sequences(prepared_tokenizer, rows, config)
    phase('loading', measurements=measurements)
    # Peaks cover loading and training in this worker, never inherited resume peaks.
    torch.cuda.init()
    measurements.begin_memory_window('before_model_loading_through_publication', reset=torch.cuda.reset_peak_memory_stats,
                                     global_step=completed)
    with measurements.stage('model_loading'):
        model, tokenizer = load_base(config, snapshot)
        if prepare_sequences(tokenizer, rows, config) != items:
            raise ValueError('Loaded tokenizer changes prepared sequences')
    with measurements.stage('initial_base_hashing'):
        pristine = {n: tensor_hash(p) for n, p in model.named_parameters()}
    with measurements.stage('adapter_attachment'):
        model = attach_adapter(model, config, output)
        base = model.get_base_model()
        backbone, head = base.model, base.get_output_embeddings()
        collator = ResponseCollator(tokenizer.pad_token_id)
        FastLanguageModel.for_training(model, use_gradient_checkpointing='unsloth')
        model.train()
        base.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config['learning_rate'], weight_decay=config['weight_decay'])
    hardware, kernels = device_and_kernels(torch)
    profile = os.environ.get('PROXYBENCH_PROFILE_CUDA') == '1'
    if profile and not os.environ.get('PROXYBENCH_STOP_FILE'):
        raise ValueError('Device profiling requires the bounded supervisor')
    compute_timer = DeviceComputeTimer(torch) if profile else None
    kernels['device_event_profiling'] = profile
    runtime = capture_runtime_identity(
        model=dict(model_id=config['model_id'], revision=config['model_revision'],
                   manifest_sha256=BASE_MANIFEST_SHA256, architecture=type(base).__name__,
                   training_loop='custom'),
        tokenizer=tokenizer_identity(snapshot), optimizer=optimizer_identity(model, optimizer),
        hardware=hardware, kernels=kernels)
    runtime_comparison = compare_runtime_identity(prior_runtime, runtime)
    record = dict(runtime_identity=runtime, comparison=runtime_comparison,
                  origin_runtime_identity=origin_runtime if saved else runtime)
    attempts = output / 'training-attempts'
    attempts.mkdir(exist_ok=True)
    durable_json(attempts / (measurements.attempt_id + '.json'), record)
    if resume and runtime_comparison['status'] != 'MATCH':
        print('Runtime identity differs or is incomplete. Exact CUDA continuation requires new acceptance.', flush=True)
    weights = lambda: get_peft_model_state_dict(model, save_embedding_layers=False)
    journal = output/'training-journal.json'
    history = []
    # Restore random state after every model, optimizer, and runtime inspection operation.
    if saved is not None:
        with measurements.stage('restoration'):
            restore_loaded_state(model, optimizer, saved, order=order, completed=completed,
                                 get_weights=weights, set_weights=lambda v: set_peft_model_state_dict(model, v))
        history = saved['history']
    else:
        durable_json(output/'training-inputs.json', accepted['inputs'])

    def memory(*, enforce=True):
        if enforce:
            torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        if enforce and free < config['limits']['device_margin_bytes']:
            raise ValueError('Device memory margin exhausted')
        values = dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_device_bytes=free)
        measurements.memory_sample(**values)
        return dict(**values, free_bytes=free, total_bytes=total)

    def loss(index):
        item = items['training'][index]
        values = {k: v.to('cuda') for k, v in collator([item]).items()}
        hidden = backbone(input_ids=values['input_ids'], attention_mask=values['attention_mask'],
                          use_cache=False, return_dict=True).last_hidden_state
        return fused_linear_cross_entropy(hidden, head.weight, values['labels']), item['response_tokens']

    def save(step, history, clean):
        phase('saving checkpoint', measurements=measurements, global_step=step, sample_position=step,
              resume_eligible=False)
        path = output/'checkpoints'/f'step-{step:06}-{uuid.uuid4().hex[:8]}'
        publish_state(model, optimizer, tokenizer, path, identity=identity, order=order,
                      completed=step, history=history, get_weights=weights, diagnostics=record,
                      measurements=measurements,
                      save_adapter=lambda p: publish_adapter(model, tokenizer, p, dict(identity, completed=step),
                                                           measurements=measurements))
        phase('training', measurements=measurements,
              last_checkpoint=dict(path=str(path), global_step=step), resume_eligible=False)
        return path

    phase('training', measurements=measurements, **dict(counters, resume_eligible=False))
    with measurements.stage('full_loop'):
        result = train_updates(model, optimizer, order, loss, journal, start=completed, history=history,
                               stop=stopping, save=save, checkpoint_interval=config['checkpoint_interval'],
                               memory=memory, report=None, measurements=measurements,
                               sequence_tokens=lambda index: sum(items['training'][index]['attention_mask']),
                               epoch_boundaries=epochs, compute_timer=compute_timer)
    if result['status'] == 'TRAINED':
        phase('validating final adapter', measurements=measurements)
        with measurements.stage('final_validation'):
            after = {n.replace('.base_layer.', '.'): tensor_hash(p) for n,p in base.named_parameters() if 'lora_' not in n}
            if after != pristine:
                raise ValueError('Training changed frozen base parameters')
        phase('publishing final adapter', measurements=measurements)
        with measurements.stage('final_publication'):
            publish_adapter(model, tokenizer, output/'adapter', dict(identity, completed=len(order)),
                            measurements=measurements)
    else:
        previous = json.loads(journal.read_text())
        require_clean_stop(previous, previous.get('checkpoint'), identity, total=len(order))
    # Read final publication peaks without adding a display synchronization or a new stop policy.
    memory(enforce=False)
    durable_json(output/'training-result.json', result)
    return result


def export_adapter(adapter, output, config):
    from unsloth import FastLanguageModel
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from proxybench.training.merge import merge_model
    from proxybench.training.merged_export import publish_merged
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    phase('loading')
    model, tokenizer = load_base(config)
    model = attach_adapter(model, config, output)
    saved = load_file(str(Path(adapter)/'adapter_model.safetensors'))
    set_peft_model_state_dict(model, saved)
    require_same_adapter(saved, get_peft_model_state_dict(model, save_embedding_layers=False))
    from transformers import AutoTokenizer
    restored = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    messages = [dict(role='system', content=Path(config['system_prompt']).read_text()), dict(role='user', content='BEGIN MARKED TARGET\nSynthetic vote: FOR\nEND MARKED TARGET')]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if restored.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False) != prompt:
        raise ValueError('Adapter tokenizer differs from the pinned base tokenizer')
    FastLanguageModel.for_inference(model)
    model.eval()
    phase('serialization')
    model = merge_model(model, prompt, output)
    expected = json.loads((output/'merge.json').read_text())['after']
    return publish_merged(model, restored, output/'merged', expected)


def launch(operation, run_dir, config, *, resource_ceiling=None, **arguments):
    from proxybench.execution.live import supervise
    from proxybench.execution.resources import ledger_entries, reconcile_captures
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    reconcile_captures(root)
    request = root/f'{operation}-request-{uuid.uuid4().hex}.json'
    durable_json(request, dict(operation=operation, config=config, run_dir=str(root), **arguments))
    phase_used = sum(row['elapsed_seconds'] for row in ledger_entries(root/'resources.jsonl') if row.get('phase') == operation)
    environment = dict(UNSLOTH_COMPILE_LOCATION=str((root/'cache'/'unsloth').resolve()),
                       TRITON_CACHE_DIR=str((root/'cache'/'triton').resolve()),
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
                       HF_HUB_DISABLE_TELEMETRY='1')
    limits = dict(config['limits'])
    if resource_ceiling is not None:
        limits['total_seconds'] = min(limits['total_seconds'], resource_ceiling)
    return supervise([sys.executable, '-m', 'proxybench.training.runtime', '_work', str(request)],
                     root/f'execution-{uuid.uuid4().hex[:8]}', limits,
                     ledger=root/'resources.jsonl', phase=operation, phase_used=phase_used, environment=environment)


def export_complete(adapter, run_dir, config):
    from proxybench.execution.resources import reconcile_captures
    root = Path(run_dir)
    reconcile_captures(root)
    complete = root/'conversion'/'complete.json'
    if not complete.exists():
        return None
    config = {**config, 'limits': {**config['limits'], 'phase_seconds': config.get('export_seconds', 1800)}}
    identity = dict(adapter=digest(Path(adapter)/'adapter_model.safetensors'),
                    adapter_config=digest(Path(adapter)/'adapter_config.json'),
                    tokenizer=tokenizer_identity(adapter), recipe=binding(config))
    if json.loads((root/'export-identity.json').read_text()) != identity:
        raise ValueError('Completed export inputs changed')
    value = json.loads(complete.read_text())
    source_binding = json.loads((root/'export-source.json').read_text())
    target = root/'conversion'/'model-bf16.gguf'
    if (value.get('status') != 'COMPLETE' or value.get('exact_transformed_payloads') is not True
            or value.get('source_commit') != '329b6160f513915f1c607dbfae3d5ce864a64a4f'
            or source_binding['identity'] != identity
            or source_binding['merged_manifest_sha256'] != value['source_manifest_sha256']
            or value['sha256'] != digest(target)):
        raise ValueError('Completed export publication chain differs')
    publication = dict(schema_version='export-publication-v1', status='COMPLETE',
        adapter_sha256=identity['adapter'], adapter_config_sha256=identity['adapter_config'],
        tokenizer_files=identity['tokenizer'], recipe_sha256=identity['recipe'],
        merged_manifest_sha256=value['source_manifest_sha256'], gguf_sha256=value['sha256'],
        conversion_complete_sha256=digest(complete))
    path = root/'export-publication.json'
    if path.exists() and json.loads(path.read_text()) != publication:
        raise ValueError('Completed export publication changed')
    durable_json(path, publication)
    return target


def export_model(adapter, run_dir, config, *, remaining_seconds=None):
    from proxybench.training.checkpoints import validate_checkpoint
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    recovered = export_complete(adapter, root, config)
    if recovered is not None:
        import shutil
        if (root/'merged').exists():
            shutil.rmtree(root/'merged')
        return recovered
    from proxybench.execution.resources import ledger_entries
    prior = sum(row['elapsed_seconds'] for row in ledger_entries(root/'resources.jsonl'))
    ceiling = None if remaining_seconds is None else prior + remaining_seconds
    config = {**config, 'limits': {**config['limits'], 'phase_seconds': config.get('export_seconds', 1800)}}
    identity = dict(adapter=digest(Path(adapter)/'adapter_model.safetensors'),
                    adapter_config=digest(Path(adapter)/'adapter_config.json'), tokenizer=tokenizer_identity(adapter), recipe=binding(config))
    identity_path = root/'export-identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Export inputs changed')
    else:
        durable_json(identity_path, identity)
    source_record = root/'export-source.json'
    complete = root/'conversion'/'complete.json'
    if complete.exists():
        value = json.loads(complete.read_text())
        target = root/'conversion'/'model-bf16.gguf'
        if value.get('status') != 'COMPLETE' or digest(target) != value['sha256']:
            raise ValueError('Completed GGUF export changed')
    else:
        if (root/'merged').exists():
            validate_checkpoint(root/'merged')
        else:
            status = launch('export', root, config, adapter=str(adapter), resource_ceiling=ceiling)
            if status != 'EXITED':
                raise RuntimeError(f'Adapter export stopped: {status}')
        merged_hash = digest(root/'merged'/'manifest.json')
        source_binding = dict(identity=identity, merged_manifest_sha256=merged_hash)
        if source_record.exists() and json.loads(source_record.read_text()) != source_binding:
            raise ValueError('Export merged-model binding changed')
        durable_json(source_record, source_binding)
        cpu_config = {**config, 'limits': {**config['limits'], 'cpu_only': True}}
        status = launch('convert', root, cpu_config, resource_ceiling=ceiling)
        if status != 'EXITED':
            raise RuntimeError(f'Model conversion stopped: {status}')
        target = root/'conversion'/'model-bf16.gguf'
    value = json.loads(complete.read_text())
    source_binding = json.loads(source_record.read_text())
    if (value.get('status') != 'COMPLETE' or value.get('exact_transformed_payloads') is not True
            or value.get('source_commit') != '329b6160f513915f1c607dbfae3d5ce864a64a4f'
            or source_binding['identity'] != identity
            or source_binding['merged_manifest_sha256'] != value['source_manifest_sha256']
            or value['sha256'] != digest(target)):
        raise ValueError('Export publication chain differs')
    publication = dict(schema_version='export-publication-v1', status='COMPLETE',
        adapter_sha256=identity['adapter'], adapter_config_sha256=identity['adapter_config'],
        tokenizer_files=identity['tokenizer'], recipe_sha256=identity['recipe'],
        merged_manifest_sha256=value['source_manifest_sha256'], gguf_sha256=value['sha256'],
        conversion_complete_sha256=digest(complete))
    publication_path = root/'export-publication.json'
    if publication_path.exists() and json.loads(publication_path.read_text()) != publication:
        raise ValueError('Saved export publication changed')
    durable_json(publication_path, publication)
    import shutil
    if (root/'merged').exists():
        shutil.rmtree(root/'merged')
    return target


def training_status(root, status):
    result = Path(root)/'training-result.json'
    if status == 'EXITED' and result.exists():
        return json.loads(result.read_text())['status']
    return status


def resume_training(run_dir):
    from proxybench.runstate import Run
    from proxybench.training.resume import training_preflight
    with Run(run_dir) as run:
        if run.state.get('operation') != 'train':
            raise ValueError('This is not a training run')
        config = run.state.get('configuration')
        if not isinstance(run.state.get('dataset'), str) or not run.state['dataset']:
            raise ValueError('Training run dataset path is missing or invalid')
        training_preflight(run.state['dataset'], run_dir, config, resume=True, run_state=run.state)
        from proxybench.execution.resources import ledger_entries
        resource_floor(run)
        root_used = sum(row['elapsed_seconds'] for row in ledger_entries(Path(run_dir)/'resources.jsonl'))
        ceiling = root_used + run.state['resource_limit_seconds'] - run.state['consumed_seconds']
        try:
            status = launch('train', run_dir, config, dataset=run.state['dataset'], resume=True, resource_ceiling=ceiling)
            run.state['status'] = training_status(run_dir, status)
        finally:
            resource_floor(run)
        return run.state


def infer_run(run):
    from proxybench.extraction.runtime import source_messages, supervised_generate_answers, runtime_identity
    from proxybench.execution.resources import reconcile_captures
    config = run.state['configuration']
    if digest(config['model']) != run.state['model_sha256'] or digest(config['system_prompt']) != run.state['identity']['prompt']:
        raise ValueError('Inference inputs changed before resume')
    if (binding(config) != run.state['identity']['recipe']
            or binding(run.state['input']) != run.state['identity']['input']
            or runtime_identity(config) != run.state['identity']['runtime']
            or tokenizer_identity(config['tokenizer']) != run.state['identity']['tokenizer']):
        raise ValueError('Inference request or runtime changed before resume')
    capture = run.path/'capture'
    resource_floor(run)
    for saved in sorted(capture.glob('*/answers/answer-0.json')):
        answer = json.loads(saved.read_text())
        request = json.loads((saved.parent.parent/'request.json').read_text())
        expected = run.state.get('inference_attempts', {}).get(saved.parent.parent.name)
        if request != expected:
            raise ValueError('Saved inference request differs')
        if answer.get('status') in {'COMPLETE', 'LENGTH_STOP'}:
            durable_json(run.path/'answer.json', answer)
            run.state['status'] = answer['status']
            return answer
    remaining = run.state['resource_limit_seconds'] - run.state['consumed_seconds']
    seconds = min(config['limits']['phase_seconds'], remaining)
    run.reserve('inference', seconds)
    worker_config = {**config, 'limits': {**config['limits'], 'phase_seconds': seconds, 'total_seconds': seconds}}
    output = capture/uuid.uuid4().hex
    run.state.setdefault('inference_attempts', {})[output.name] = dict(
        messages=[source_messages(run.state['input'], config)], config=worker_config)
    run.save()
    import time
    start = time.monotonic()
    try:
        answer = supervised_generate_answers([source_messages(run.state['input'], config)], output, worker_config)[0]
        durable_json(run.path/'answer.json', answer)
        run.state['status'] = answer['status']
        return answer
    finally:
        run.settle('inference', time.monotonic()-start)
        resource_floor(run)


def resume_runtime(run_dir):
    from proxybench.runstate import Run
    with Run(run_dir) as run:
        operation = run.state.get('operation')
        if operation == 'infer':
            infer_run(run)
        elif operation == 'export':
            resource_floor(run)
            try:
                run.state['model'] = str(export_model(run.state['adapter'], run_dir, run.state['configuration'],
                    remaining_seconds=run.state['resource_limit_seconds']-run.state['consumed_seconds']))
                run.state['status'] = 'COMPLETE'
            finally:
                resource_floor(run)
        else:
            raise ValueError('Unsupported runtime resume operation')
        run.save()
        return run.state


def add_cli(subparsers):
    from proxybench.training.validation import validate_runtime
    p = subparsers.add_parser('validate-runtime')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--config', default='configs/inference.json')
    p.add_argument('--training-config', default='configs/training.json')
    p.add_argument('--adapter', default='artifacts/models/ProxyType-4B/adapter')
    p.set_defaults(handler=validate_runtime)
    for operation in ('train', 'export', 'infer'):
        p = subparsers.add_parser(operation)
        p.add_argument('--config', default='configs/inference.json' if operation == 'infer' else 'configs/training.json')
        p.add_argument('--run-dir', required=True)
        if operation == 'train':
            p.add_argument('--dataset', default='data/training-dataset')
        elif operation == 'export':
            p.add_argument('--adapter', required=True)
        else:
            p.add_argument('--input', required=True)
            p.add_argument('--model')
        p.set_defaults(handler=cli)


def cli(args):
    from proxybench.runstate import Run
    from proxybench.execution.resources import ledger_entries
    config = load_config(args.config)
    if getattr(args, 'model', None):
        config['model'] = str(Path(args.model).resolve())
    operation = 'infer' if hasattr(args, 'input') else 'train' if hasattr(args, 'dataset') else 'export'
    identity = dict(operation=operation, recipe=binding(config), prompt=digest(config['system_prompt']))
    if operation == 'infer':
        from proxybench.extraction.runtime import runtime_identity
        path = Path(args.input)
        if path.stat().st_size > 1024*1024:
            raise ValueError('Input exceeds 1 MiB')
        identity.update(input=binding(path.read_text()), runtime=runtime_identity(config),
                        tokenizer=tokenizer_identity(config['tokenizer']))
    if operation == 'train':
        from proxybench.training.dataset import read_release
        _, manifest = read_release(args.dataset)
        identity['dataset'] = binding(manifest)
    with Run(args.run_dir, create=True, identity=identity, config=config) as run:
        run.state['operation'] = operation
        if operation == 'infer':
            path = Path(args.input)
            if path.stat().st_size > 1024*1024:
                raise ValueError('Input exceeds 1 MiB')
            run.state['input'] = path.read_text()
            run.state['model_sha256'] = digest(config['model'])
        if operation == 'export':
            run.state['adapter'] = str(Path(args.adapter).resolve())
        if operation == 'train':
            run.state['dataset'] = str(Path(args.dataset).resolve())
        run.save()
        try:
            if operation == 'infer':
                infer_run(run)
            elif operation == 'train':
                from proxybench.training.resume import training_preflight
                training_preflight(run.state['dataset'], args.run_dir, config, run_state=run.state)
                status = launch(operation, args.run_dir, config, dataset=run.state['dataset'], resume=False)
                run.state['status'] = training_status(args.run_dir, status)
            else:
                run.state['model'] = str(export_model(args.adapter, args.run_dir, config))
                run.state['status'] = 'COMPLETE'
            return run.state
        finally:
            if operation != 'infer':
                resource_floor(run)
            run.save()


def main():
    if sys.argv[1] == '_infer':
        from proxybench.extraction.runtime import generate_answers
        request = json.loads(Path(sys.argv[2]).read_text())
        generate_answers(request['messages'], sys.argv[3], request['config'])
    elif sys.argv[1] == '_work':
        request = json.loads(Path(sys.argv[2]).read_text())
        if request['operation'] == 'train':
            train(request['dataset'], request['run_dir'], request['config'], request['resume'])
        elif request['operation'] == 'validate-adapter':
            from proxybench.training.validation import validate_adapter
            validate_adapter(request['adapter'], request['run_dir'], request['config'])
        elif request['operation'] == 'export':
            export_adapter(request['adapter'], request['run_dir'], request['config'])
        else:
            from proxybench.training.conversion import convert
            convert(Path(request['run_dir'])/'merged', Path(request['run_dir'])/'conversion', request['config'])
    else:
        p = argparse.ArgumentParser()
        add_cli(p.add_subparsers(required=True))
        cli(p.parse_args())


if __name__ == '__main__':
    main()
