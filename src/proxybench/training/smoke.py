"""Bounded Qwen3.5-4B adapter test on explicitly accepted development labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

from proxybench.annotation.bindings import review_binding
from proxybench.training.labels import dumps, from_review, read_json, validate
from proxybench.training.sequences import pad_sequence, sequence

MODEL_ID = 'Qwen/Qwen3.5-4B'
DATA_SHA256 = '5d469a1221b72b3be92451c7422f8f4dd3537e574e4a88861a89a7166ad28807'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def identity(configuration):
    if configuration['model_id'] != MODEL_ID or configuration['dataset_sha256'] != DATA_SHA256:
        raise ValueError('Wrong model or dataset identity for this bounded test')
    expected = dict(examples=15, seed=42, rank=8, alpha=16, learning_rate=1e-4,
                    steps=30, context_ceiling=12288, generation_ceiling=4096)
    if any(configuration.get(k) != v for k, v in expected.items()):
        raise ValueError('Configuration changes the approved bounded test')


def accepted_rows(data, configuration):
    identity(configuration)
    if digest(data) != DATA_SHA256:
        raise ValueError('Dataset bytes differ from the accepted dataset')
    manifest = read_json(Path(data).with_suffix('.manifest.json').read_bytes())
    rows = [read_json(line) for line in Path(data).read_bytes().splitlines()]
    if len(rows) != 15 or len(manifest['examples']) != 15 or manifest['split'] != 'development':
        raise ValueError('Expected exactly 15 development examples')
    identities = set()
    checked_files = {}
    for i, (row, metadata) in enumerate(zip(rows, manifest['examples'], strict=True)):
        if metadata['dataset_row'] != i or metadata['split'] != 'development':
            raise ValueError('Dataset ordering or split differs from acceptance')
        if hashlib.sha256((dumps(row) + '\n').encode()).hexdigest() != metadata['row_sha256']:
            raise ValueError('Accepted row hash differs')
        messages = row['messages']
        if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
            raise ValueError('Expected one user message and one assistant message')
        for message in messages:
            if any(marker in message['content'] for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')):
                raise ValueError('Reserved template marker in accepted content')
        validate(read_json(messages[1]['content']))
        source = metadata['source']
        key = (source['accession'], source['selected_index'])
        if key in identities:
            raise ValueError('Repeated source target')
        identities.add(key)
        for prefix in ('primary', 'votes'):
            path = source[prefix + '_path']
            if path not in checked_files:
                checked_files[path] = digest(path)
            if checked_files[path] != source[prefix + '_sha256']:
                raise ValueError('Original source hash differs')
        review_path = Path(metadata['review_export_path'])
        approval = read_json(Path(metadata['acceptance_path']).read_bytes())
        if (approval['decision'] != 'ACCEPTED' or approval['export_sha256'] != digest(review_path)
                or metadata['packet_id'] not in approval['accepted_packet_ids']):
            raise ValueError('Acceptance does not bind the exact review')
        packet_id = metadata['packet_id']
        packets = read_json((Path(metadata['run']) / 'review/packet-set.json').read_bytes())
        packet = next(p for p in packets if p['manifest']['packet_id'] == packet_id)
        answer = read_json(review_path.read_bytes())['packets'][packet_id]
        if (source != packet['manifest'] or answer['reviewed'] is not True
                or answer['input_binding'] != review_binding(packet)
                or messages[0]['content'] != packet['model_input']
                or messages[1]['content'] != dumps(from_review(answer['fields']))):
            raise ValueError('Example differs from its reviewed source or accepted response')
    return rows, manifest


def prepare(tokenizer, rows, configuration):
    prepared = []
    for row in rows:
        prompt, response = [m['content'] for m in row['messages']]
        item = sequence(tokenizer, prompt, response, context_cap=configuration['context_ceiling'])
        suffix = item['input_ids'][item['input_tokens']:]
        decoded = tokenizer.decode(suffix, skip_special_tokens=False)
        termination = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
        if decoded != response + termination + '\n' and decoded != response + termination:
            raise ValueError('Supervised region changes the accepted response or termination')
        prepared.append(item)
    response_max = max(p['response_tokens'] for p in prepared)
    allowance = math.ceil((response_max + 256) / 128) * 128
    context = math.ceil(max(max(p['combined_tokens'] for p in prepared),
                            max(p['input_tokens'] for p in prepared) + allowance) / 256) * 256
    if allowance > configuration['generation_ceiling'] or context > configuration['context_ceiling']:
        raise ValueError(f'Measured requirements exceed ceilings: generation={allowance}, context={context}')
    rng = random.Random(configuration['seed'])
    order = []
    for _ in range(2):
        epoch = list(range(len(rows)))
        rng.shuffle(epoch)
        order.extend(epoch)
    return prepared, dict(generation_tokens=allowance, context_tokens=context, order=order)


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


def diagnostics(directory, prefix):
    outputs = [read_json((directory / f'{prefix}-{i}.json').read_bytes()) for i in range(15)]
    losses = read_json((directory / f'{prefix}-losses.json').read_bytes())
    tokens = sum(row['response_tokens'] for row in losses)
    return {
        'attempts': len(outputs),
        'format_valid': sum(row['format_valid'] for row in outputs),
        'exact_target': sum(row['exact_target'] for row in outputs),
        'length_stopped': sum(row['termination'] == 'length' for row in outputs),
        'token_weighted_response_loss': sum(row['loss'] * row['response_tokens'] for row in losses) / tokens,
        'source_content_assessment': 'Requires separate source review',
    }


def require_same_adapter(expected, actual):
    import torch
    if set(expected) != set(actual):
        raise ValueError('Adapter tensor names differ')
    for name, tensor in expected.items():
        other = actual[name]
        if tensor.dtype != other.dtype or not torch.equal(tensor.cpu(), other.cpu()):
            raise ValueError(f'Adapter tensor differs: {name}')


def save_adapter(model, tokenizer, directory):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(directory), safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(str(directory))
    saved = load_file(str(directory / 'adapter_model.safetensors'))
    if not saved or any('lora_' not in name for name in saved):
        raise ValueError('Saved state contains unintended parameters')
    require_same_adapter(saved, get_peft_model_state_dict(model, save_embedding_layers=False))


class GenerationRecorder:
    """Flush generated token IDs so a stopped worker retains its partial answer."""
    def __init__(self, path, index):
        self.path, self.index = Path(path), index
        self.first, self.count = True, 0

    def __enter__(self):
        self.stream = self.path.open('x')
        self.started = time.monotonic()
        return self

    def __exit__(self, *args):
        self.stream.close()

    def put(self, value):
        if self.first:
            self.first = False
            return
        for token in value.reshape(-1).tolist():
            self.stream.write(json.dumps(token) + '\n')
            self.count += 1
        self.stream.flush()
        if self.count % 256 == 0:
            print(json.dumps({'generation_index': self.index, 'generated_tokens': self.count,
                              'elapsed_seconds': time.monotonic() - self.started}), flush=True)

    def end(self):
        print(json.dumps({'generation_index': self.index, 'generation_complete': True,
                          'generated_tokens': self.count}), flush=True)


def worker(args):
    # Import Unsloth before Transformers or PEFT so its supported patches apply.
    import os
    import time
    import gc
    import importlib.metadata
    import platform
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['HF_HOME'] = str(Path('artifacts/cache/huggingface').resolve())
    os.environ['UNSLOTH_COMPILE_LOCATION'] = str(Path('artifacts/cache/unsloth').resolve())
    os.environ['TRITON_CACHE_DIR'] = str(Path('artifacts/cache/triton').resolve())
    from unsloth import FastLanguageModel
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import set_seed
    from unsloth_zoo.loss_utils import fused_linear_cross_entropy

    config = read_json(args.configuration.read_bytes())
    identity(config)
    rows, manifest = accepted_rows(args.data, config)
    provenance = read_json(args.provenance.read_bytes())
    if provenance['model_id'] != MODEL_ID or provenance['revision'] != config['model_revision']:
        raise ValueError('Wrong pinned model identity')
    for name, record in provenance['files'].items():
        if digest(args.model / name) != record['sha256']:
            raise ValueError(f'Model file hash differs: {name}')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    set_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(args.output / 'execution.json', {
        'phase': args.phase, 'configuration': config, 'model_revision': provenance['revision'],
        'dataset_sha256': DATA_SHA256, 'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('unsloth', 'unsloth_zoo', 'torch', 'transformers', 'peft', 'triton', 'cut_cross_entropy')},
        'cuda': torch.version.cuda, 'worker_sha256': digest(__file__),
        'compiler': os.environ.get('CC'), 'sample_seconds': config['limits']['sample_seconds']})

    def phase(name):
        value = {'phase': name, 'elapsed_seconds': time.monotonic() - started}
        print(json.dumps(value), flush=True)
        if os.environ.get('PROXYBENCH_PHASE_FILE'):
            path = Path(os.environ['PROXYBENCH_PHASE_FILE'])
            temporary = path.with_suffix('.tmp')
            save(temporary, value)
            temporary.replace(path)

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result = dict(free_bytes=free, total_bytes=total,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if free < config['limits']['device_margin_bytes'] or total - result['peak_reserved_bytes'] < config['limits']['device_margin_bytes']:
            raise RuntimeError('Device memory margin failed at worker checkpoint')
        return result

    phase('loading')
    from transformers import AutoTokenizer
    original_tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    items, bounds = prepare(original_tokenizer, rows, config)
    save(args.output / 'prepared.json', {'items': items, 'bounds': bounds, 'examples': manifest['examples']})
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model.resolve()), max_seq_length=bounds['context_tokens'],
        dtype=torch.bfloat16, load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
        text_only=True, local_files_only=True, use_gradient_checkpointing='unsloth',
        random_state=config['seed'])
    actual_items, actual_bounds = prepare(tokenizer, rows, config)
    if actual_items != items or actual_bounds != bounds:
        raise ValueError('Loader changed the pinned tokenizer sequences')
    targets, unsupported = adapter_targets(model.named_modules(), torch.nn.Linear)
    loaded_class = type(model).__name__
    # The official nested text configuration omits this descriptive field.
    # Unsloth generation reads it, so identify the actual loaded architecture.
    model.config.architectures = [loaded_class]
    model = FastLanguageModel.get_peft_model(
        model, r=config['rank'], target_modules='(?:' + '|'.join(re.escape(n) for n in targets) + ')', lora_alpha=config['alpha'],
        lora_dropout=0, bias='none', use_gradient_checkpointing='unsloth',
        random_state=config['seed'], max_seq_length=bounds['context_tokens'],
        temporary_location=str(args.output / 'temporary-buffers'))
    trainables = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainables or any('lora_' not in n for n in trainables):
        raise ValueError('Unintended trainable parameters')
    for target in targets:
        if not any(target + '.lora_' in n for n in trainables):
            raise ValueError(f'No trainable adapter for {target}')
    save(args.output / 'model.json', dict(loaded_class=loaded_class, targets=targets,
        unsupported=unsupported, trainables={n: {'shape': list(p.shape), 'dtype': str(p.dtype)} for n,p in trainables.items()},
        attention_kernels={n: str(getattr(m, 'chunk_gated_delta_rule')) for n,m in model.named_modules()
                           if hasattr(m, 'chunk_gated_delta_rule')},
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}), memory=memory()))
    base = model.get_base_model()
    backbone = base.model
    head = base.get_output_embeddings()
    collator = ResponseCollator(tokenizer.pad_token_id)

    def batch(item):
        value = collator([item])
        if value['labels'][0].tolist() != item['labels'] or value['input_ids'][0].tolist() != item['input_ids']:
            raise ValueError('Collator changed exact labels or input')
        return {k: v.to('cuda') for k,v in value.items()}

    def response_loss(item):
        value = batch(item)
        hidden = backbone(input_ids=value['input_ids'], attention_mask=value['attention_mask'],
                          use_cache=False, return_dict=True).last_hidden_state
        # The supported loss function applies the next-token shift exactly once.
        return fused_linear_cross_entropy(hidden, head.weight, value['labels'])

    def training_mode():
        FastLanguageModel.for_training(model, use_gradient_checkpointing='unsloth')
        model.train()
        base.config.use_cache = False

    def inference_mode():
        FastLanguageModel.for_inference(model)
        model.eval()

    def generate(index, label, force=False):
        inference_mode()
        prompt = generation_input(items[index])
        ids = torch.tensor([prompt], dtype=torch.long, device='cuda')
        begin = time.monotonic()
        with GenerationRecorder(args.output / f'{label}-{index}.tokens.jsonl', index) as recorder, torch.inference_mode():
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=bounds['generation_tokens'],
                min_new_tokens=bounds['generation_tokens'] if force else 0,
                do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, streamer=recorder)
        tokens = result[0, len(prompt):].tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        ended = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
        text = tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
        parsed, error = None, None
        try:
            parsed = read_json(text)
            validate(parsed)
        except (ValueError, TypeError) as exc:
            error = str(exc)
        value = dict(index=index, source=manifest['examples'][index]['source'], token_ids=tokens,
                     raw=raw, text=text, termination='eos' if ended else 'length',
                     elapsed_seconds=time.monotonic()-begin, memory=memory(),
                     format_valid=ended and error is None, format_error=error,
                     exact_target=ended and error is None and parsed == read_json(rows[index]['messages'][1]['content']))
        del result, ids
        gc.collect()
        torch.cuda.empty_cache()
        return value

    def losses(label):
        training_mode()
        model.eval()
        result = []
        for i,item in enumerate(items):
            with torch.no_grad():
                loss = response_loss(item)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite evaluation loss')
            value = dict(index=i, loss=loss.item(), response_tokens=sum(n != -100 for n in item['labels'][1:]))
            result.append(value)
            save(args.output / f'{label}-losses.json', result)
        training_mode()
        return result

    def frozen_fingerprints():
        result = {}
        for n,p in model.named_parameters():
            if not p.requires_grad and (n.endswith('weight') or 'norm' in n):
                sample = p.detach().reshape(-1)[::max(1,p.numel()//32)].contiguous().view(torch.uint8).cpu().numpy().tobytes()
                result[n] = hashlib.sha256(sample).hexdigest()
        return result

    def train(order):
        training_mode()
        optimizer = torch.optim.AdamW(list(trainables.values()), lr=config['learning_rate'], weight_decay=0)
        frozen_before = frozen_fingerprints()
        save(args.output / 'frozen-before.json', frozen_before)
        with (args.output / 'steps.jsonl').open('x') as log:
            for step,index in enumerate(order, 1):
                begin = time.monotonic()
                before = {n:p.detach().clone() for n,p in trainables.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = response_loss(items[index])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise ValueError('Frozen parameter received a gradient')
                norm = torch.nn.utils.clip_grad_norm_(list(trainables.values()), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in trainables.values()):
                    raise ValueError('Nonfinite adapter parameter')
                changed = any(not torch.equal(before[n],p) for n,p in trainables.items())
                if not changed:
                    raise ValueError('No adapter tensor changed')
                value = dict(step=step, index=index, target=manifest['examples'][index]['source'],
                             loss=loss.item(), gradient_norm=norm.item(), learning_rate=config['learning_rate'],
                             adapter_updated=changed, elapsed_seconds=time.monotonic()-begin, memory=memory())
                log.write(json.dumps(value) + '\n')
                log.flush()
                print(json.dumps({k:v for k,v in value.items() if k not in ('target','memory')}), flush=True)
                del before, loss
        frozen_after = frozen_fingerprints()
        save(args.output / 'frozen-after.json', frozen_after)
        if frozen_before != frozen_after:
            raise ValueError('Frozen base fingerprint changed')
        del optimizer

    phase('loss_fixture')
    training_mode()
    torch.manual_seed(42)
    hidden = torch.randn(2, 8, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, 64, device='cuda', dtype=torch.bfloat16)
    labels = torch.randint(0,128,(2,8),device='cuda')
    labels[:, :3] = -100
    labels[1, -2:] = -100
    fused = fused_linear_cross_entropy(hidden, weight, labels)
    logits = torch.nn.functional.linear(hidden, weight).float()
    ordinary = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,128), labels[:,1:].reshape(-1))
    torch.testing.assert_close(fused, ordinary, atol=0.05, rtol=0.005)
    fused_grad = torch.autograd.grad(fused, hidden, retain_graph=True)[0]
    ordinary_grad = torch.autograd.grad(ordinary, hidden)[0]
    torch.testing.assert_close(fused_grad, ordinary_grad, atol=0.01, rtol=0.05)
    save(args.output / 'loss-fixture.json', {'fused':fused.item(), 'ordinary':ordinary.item(), 'single_shift_and_mask_pass':True})
    del hidden, weight, labels, fused, ordinary, logits, fused_grad, ordinary_grad
    if args.phase == 'pilot':
        phase('pilot_training')
        longest = max(range(len(items)), key=lambda i: items[i]['combined_tokens'])
        train([longest, longest])
        phase('pilot_response_loss')
        losses('pilot')
        phase('pilot_generation_capacity')
        longest_prompt = max(range(len(items)), key=lambda i: items[i]['input_tokens'])
        value = generate(longest_prompt, 'capacity', force=True)
        save(args.output / 'capacity.json', value)
        if len(value['token_ids']) != bounds['generation_tokens']:
            raise ValueError('Capacity probe did not exercise the full output allowance')
        save(args.output / 'complete.json', {'status':'COMPLETE', 'disposable_steps':2, 'memory':memory()})
        return
    if args.phase == 'reload':
        state = load_file(str(args.adapter / 'adapter_model.safetensors'))
        set_peft_model_state_dict(model, state)
        restored = get_peft_model_state_dict(model, save_embedding_layers=False)
        require_same_adapter(state, restored)
        phase('reload_generation')
        for index in (1,6,2):
            value = generate(index, 'reload')
            save(args.output / f'reload-{index}.json', value)
            expected = read_json((args.adapter.parent / f'final-{index}.json').read_bytes())
            if value['token_ids'] != expected['token_ids']:
                raise ValueError(f'Reloaded generation differs for row {index}')
        save(args.output / 'complete.json', {'status':'COMPLETE','exact_tensors':True,'matching_generations':True})
        return
    phase('baseline_generation')
    with model.disable_adapter():
        for i in range(15):
            save(args.output / f'baseline-{i}.json', generate(i, 'baseline'))
        phase('baseline_loss')
        losses('baseline')
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(config['seed'])
    phase('training')
    train(bounds['order'])
    phase('saving_adapter')
    save_adapter(model, tokenizer, args.output / 'adapter')
    save(args.output / 'adapter-checkpoint.json', dict(steps=30, model_revision=provenance['revision'],
         dataset_sha256=DATA_SHA256, order=bounds['order'], pilot_updates_included=False))
    phase('final_loss')
    losses('final')
    phase('final_generation')
    for i in range(15):
        save(args.output / f'final-{i}.json', generate(i, 'final'))
    phase('checking_saved_adapter')
    saved = load_file(str(args.output / 'adapter/adapter_model.safetensors'))
    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    require_same_adapter(saved, state)
    save(args.output / 'diagnostics.json', {phase: diagnostics(args.output, phase) for phase in ('baseline', 'final')})
    save(args.output / 'complete.json', dict(status='COMPLETE', steps=30, configuration=config,
         model_revision=provenance['revision'], dataset_sha256=DATA_SHA256, order=bounds['order'],
         pilot_updates_included=False, memory=memory()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['pilot', 'main', 'reload'])
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    if args.phase == 'reload' and args.adapter is None:
        parser.error('Reload requires --adapter')
    existed = args.output.exists()
    try:
        worker(args)
    except Exception as exc:
        if not existed and args.output.exists():
            save(args.output / 'failure.json', {'error_type':type(exc).__name__, 'error':str(exc)})
        raise


if __name__ == '__main__':
    main()
