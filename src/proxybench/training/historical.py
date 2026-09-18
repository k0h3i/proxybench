"""Freeze historical pilot inputs and admit the complete remaining schedule."""

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from proxybench.annotation.historical import check_packet
from proxybench.execution.resources import durable_json, ledger_entries
from proxybench.training.dataset import read_release
from proxybench.training.labels import sha
from proxybench.training.sequences import sequence
from proxybench.training.smoke import digest, read_json
from proxybench.training.trajectory import sample_order

# Reservations already include the reviewed 25 percent allowance.
PHASES = [('training', 'gpu', 1200), ('original-export', 'gpu', 900),
          ('original-conversion', 'cpu', 600), ('trained-export', 'gpu', 900),
          ('trained-conversion', 'cpu', 600), ('baseline', 'gpu', 1650), ('final', 'gpu', 1350)]
EVALUATION_PHASES = tuple(n for n, _, _ in PHASES if n != 'training')

ACCEPTED_RELEASES = {
    'data/annotations/sol-historical-20260917-b10': dict(
        training_examples=96, development_examples=24,
        release_hashes={
            'manifest.json': '2c430590e945919b7768eb2dc725d6fbaf8479be1e99cd5bfb84d479d87053fc',
            'training.jsonl': '3846a5a1ba7834430c6a30d71ea0442a630d90b04a816940eaca43c5a3aa0225',
            'development.jsonl': '7f5b90585cf9ae36a54800bae9b8898566460e91787829f95d42fae2779c710e',
            'policy.md': '04e4950a4ad5e45b34f543ead1971459722fe61d49b7117684445849d9568bae'}),
    'data/annotations/sol-historical-20260917-b18': dict(
        training_examples=174, development_examples=42,
        release_hashes={
            'manifest.json': '61967e39aba5f5d1a8249fe955c097419f5eea400f687004ffe009f5ac4db4e2',
            'training.jsonl': '1932588c3eb4bd578aff8bae87bcab49dc3fc9fff7c116aef11e87dd20a1f58d',
            'development.jsonl': '229e03d17e8c75454b7078b1be3ea447603f21dd02fe49a3c6c5ebbb5ba0ab8f',
            'policy.md': '04e4950a4ad5e45b34f543ead1971459722fe61d49b7117684445849d9568bae'}),
}


def binding(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def inventory(paths):
    return {str(p): digest(p) for p in sorted(set(map(Path, paths))) if p.is_file()}


def check_files(files):
    for name, expected in files.items():
        if digest(name) != expected:
            raise ValueError(f'File identity changed: {name}')


def read_configuration(path):
    config = read_json(Path(path).read_bytes())
    expected = dict(epochs=2, batch_size=1, accumulation=1, seed=42, rank=8, alpha=16, dropout=0,
                    learning_rate=.0001, weight_decay=0, max_grad_norm=1.0,
                    context_tokens=5120, input_tokens=3328, response_tokens=1792,
                    python_request_seconds=240, engine_request_seconds=60)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Configuration differs from the reviewed recipe')
    release = ACCEPTED_RELEASES.get(config.get('release'))
    if release is None:
        raise ValueError('Configuration does not select a reviewed dataset release')
    counts = {key: config.get(key) for key in ('training_examples', 'development_examples')}
    if counts != {key: release[key] for key in counts} or config.get('updates') != config['training_examples'] * config['epochs']:
        raise ValueError('Configuration counts differ from the selected release')
    if (config['limits']['total_seconds'] != 7200 or config['cpu_limits']['total_seconds'] != 3600
            or any(config[k]['phase_seconds'] != 1800 or config[k]['grace_seconds'] != 30
                   or config[k]['fixed_phase'] is not True for k in ('limits', 'cpu_limits'))):
        raise ValueError('Configuration differs from the reviewed resource ceilings')
    if (config['minimum_disk_bytes'] < 60*1024**3 or config['limits']['device_margin_bytes'] < 2*1024**3
            or config['cpu_limits'].get('cpu_only') is not True
            or config['limits'].get('cpu_only', False)
            or any(config[k]['start_host_bytes'] < 4*1024**3 or config[k]['stop_host_bytes'] < 1024**3
                   or config[k]['sample_seconds'] > 1 for k in ('limits', 'cpu_limits'))):
        raise ValueError('Configuration weakens the reviewed memory or disk limits')
    # Pin data and model independently of editable configuration values.
    if (config['release_hashes'] != release['release_hashes'] or config['model_id'] != 'Qwen/Qwen3.5-4B'
            or config['model_revision'] != '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'):
        raise ValueError('Pinned data or model identity differs')
    return config


def prepare_sequences(tokenizer, rows, config):
    result = {}
    for split in ('training', 'development'):
        result[split] = []
        for row in rows[split]:
            prompt, answer = (m['content'] for m in row['messages'])
            for value in (prompt, answer):
                if any(marker in value for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')):
                    raise ValueError('Reserved template marker in source or answer')
            item = sequence(tokenizer, prompt, answer, context_cap=config['context_tokens'])
            suffix = item['input_ids'][item['input_tokens']:]
            eos = tokenizer.eos_token_id
            # Supervise through the end token, excluding template whitespace after it.
            if suffix.count(eos) != 1:
                raise ValueError('Expected one answer termination token')
            end = item['input_tokens'] + suffix.index(eos) + 1
            for key in ('input_ids', 'attention_mask', 'labels'):
                item[key] = item[key][:end]
            item['response_tokens'] = end-item['input_tokens']
            item['combined_tokens'] = end
            if tokenizer.decode(item['input_ids'][item['input_tokens']:-1], skip_special_tokens=False) != answer:
                raise ValueError('Answer loss mask changes the accepted answer')
            if item['input_tokens'] > config['input_tokens'] or item['response_tokens'] > config['response_tokens']:
                raise ValueError('Accepted sequence exceeds a reviewed limit')
            result[split].append(item)
    return result


def panel_indices(items):
    prompt = max(range(len(items)), key=lambda i: (items[i]['input_tokens'], -i))
    answer = max((i for i in range(len(items)) if i != prompt), key=lambda i: (items[i]['response_tokens'], -i))
    return [prompt, answer, next(i for i in range(len(items)) if i not in (prompt, answer))]


def runtime_files(config):
    pin = read_json(Path(config['runtime_pin']).read_bytes())
    if pin['source_commit'] != '329b6160f513915f1c607dbfae3d5ce864a64a4f':
        raise ValueError('Pinned engine revision differs')
    check_files(pin['files'])
    approved_path = Path(config['runtime_pin']).with_name('panel-approved.json')
    approved = read_json(approved_path.read_bytes())
    if read_json(Path(config['engine_configuration']).read_bytes()) != approved['configuration']:
        raise ValueError('Engine configuration differs from the tested runtime pin')
    archive = Path(config['engine_root'])/'llama.cpp-source-commit-329b6160f513915f1c607dbfae3d5ce864a64a4f.tar.gz'
    if digest(archive) != '9d46c7ce4da17fa584df7d906209ff79f2b827a97b9fe57cd9652b29c55a6ca5':
        raise ValueError('Pinned converter source archive differs')
    with tarfile.open(archive) as saved:
        for member in saved:
            if member.isfile() and member.name.endswith('.py'):
                path = Path(config['engine_root'])/'source'/member.name
                if digest(path) != sha(saved.extractfile(member).read()):
                    raise ValueError('Converter code differs from its pinned source archive')
    expected_versions = {'unsloth': '2026.9.4', 'unsloth_zoo': '2026.9.3', 'transformers': '5.5.0',
                         'torch': '2.12.1', 'peft': '0.21.0', 'triton': '3.7.1', 'cut_cross_entropy': '25.1.1'}
    if any(importlib.metadata.version(name) != version for name, version in expected_versions.items()):
        raise ValueError('Training packages differ from the tested environment')
    paths = [Path(p) for p in pin['files']] + [approved_path, archive]
    # Hash installed code and native libraries, not only package version strings.
    for root in (Path(config['environment']), Path(config['toolchain']), Path(config['engine_root']) / 'source'):
        paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix != '.pyc')
    paths.extend([Path(sys.executable).resolve(), Path(config['runtime_pin']), Path(config['engine_configuration'])])
    return inventory(paths)


def prepare_run(root, config, configuration_path):
    from transformers import AutoTokenizer
    release = Path(config['release'])
    check_files({str(release/name): value for name, value in config['release_hashes'].items()})
    rows, manifest = read_release(release)
    expected_counts = {key: config[key + '_examples'] for key in ('training', 'development')}
    if set(rows) != set(expected_counts) or {k: len(v) for k, v in rows.items()} != expected_counts:
        raise ValueError('Dataset row counts differ from the reviewed configuration')
    policy = (release/'policy.md').read_text()
    for meta in manifest['examples']:
        check_packet(Path.cwd(), meta['packet'], policy)
    provenance = read_json(Path(config['provenance']).read_bytes())
    if provenance['model_id'] != config['model_id'] or provenance['revision'] != config['model_revision']:
        raise ValueError('Original model provenance differs')
    model_files = {str(Path(config['model'])/name): value['sha256'] for name, value in provenance['files'].items()}
    check_files(model_files)
    if any('adapter' in p.name for p in Path(config['model']).iterdir()):
        raise ValueError('Original model directory contains an adapter')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    items = prepare_sequences(tokenizer, rows, config)
    tokenizer.save_pretrained(root/'tokenizer')
    prepared = dict(items=items, rows=rows, examples=manifest['examples'],
                    order=sample_order(config['training_examples'], config['epochs'], config['seed']),
                    panel=panel_indices(items['training']), eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id)
    durable_json(root/'prepared.json', prepared)
    print(f'CPU sequences: {config["training_examples"]} training, '
          f'{config["development_examples"]} development, {config["updates"]} updates. '
          'Hashing runtime files.', flush=True)
    code = inventory(Path('src/proxybench').rglob('*.py'))
    inputs = inventory([*release.rglob('*'), Path(config['provenance']), Path(configuration_path),
                        *(Path(m['packet']['manifest']['source_path']) for m in manifest['examples'])])
    runtime = runtime_files(config)
    identity = dict(schema='historical-pilot-v1', configuration=config,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    code=code, inputs=inputs, model=model_files, runtime=runtime,
                    executable=str(Path(sys.executable).absolute()), python=sys.version,
                    versions={d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                              if d.metadata['Name']},
                    prepared_sha256=digest(root/'prepared.json'), tokenizer=inventory((root/'tokenizer').rglob('*')))
    durable_json(root/'identity.json', identity)
    durable_json(root/'state.json', dict(status='TRAINING_READY', operation='separate', completed_phases={}, attempts={},
                                       identity=binding(identity), reservations={n: s for n, _, s in PHASES}))


def validate_identity(root, config):
    identity = read_json((root/'identity.json').read_bytes())
    state = read_json((root/'state.json').read_bytes())
    if binding(identity) != state['identity'] or identity['configuration'] != config:
        raise ValueError('Run identity or configuration differs')
    # Keep the preparation revision as provenance. Exact file inventories below
    # reject executable changes without blocking later documentation commits.
    if digest(root/'prepared.json') != identity['prepared_sha256']:
        raise ValueError('Prepared sequences changed')
    if str(Path(sys.executable).absolute()) != identity['executable'] or sys.version != identity['python']:
        raise ValueError('Python runtime changed')
    for key in ('code', 'inputs', 'model', 'tokenizer'):
        check_files(identity[key])
    if runtime_files(config) != identity['runtime']:
        raise ValueError('Runtime file inventory changed')
    if inventory(Path('src/proxybench').rglob('*.py')) != identity['code']:
        raise ValueError('Source code inventory changed')
    for completed in state['completed_phases'].values():
        check_files(completed['files'])
    for attempts in state.get('partial_phases', {}).values():
        for attempt in attempts:
            check_files(attempt['files'])
    return identity, state


def reuse_prepared_run(root, config):
    """Reuse intact prepared data before training and preserve previous run records."""
    identity = read_json((root/'identity.json').read_bytes())
    state = read_json((root/'state.json').read_bytes())
    if (state['status'] not in {'PREPARED', 'EXPORT_REJECTED'} or state.get('operation')
            or 'training' in state.get('attempts', {}) or (root/'training-journal.json').exists()
            or (root/'adapter').exists() or (root/'previous-run.json').exists()):
        raise ValueError('Reuse requires prepared data before retained training starts')
    if binding(identity) != state['identity'] or identity['configuration'] != config:
        raise ValueError('Run identity or configuration differs')
    if (digest(root/'prepared.json') != identity['prepared_sha256']
            or str(Path(sys.executable).absolute()) != identity['executable'] or sys.version != identity['python']):
        raise ValueError('Prepared sequences or Python runtime changed')
    for key in ('inputs', 'model', 'tokenizer'):
        check_files(identity[key])
    if runtime_files(config) != identity['runtime']:
        raise ValueError('Runtime file inventory changed')
    changed = {'src/proxybench/training/historical.py', 'src/proxybench/training/historical_run.py',
               'src/proxybench/training/historical_worker.py'}
    check_files({k: v for k, v in identity['code'].items() if k not in changed})
    code = inventory(Path('src/proxybench').rglob('*.py'))
    if set(code) != set(identity['code']):
        raise ValueError('Source file inventory changed')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for name in changed:
        if sha(subprocess.check_output(['git', 'show', f'{revision}:{name}'])) != code[name]:
            raise ValueError('Commit and test the separate-operation code before reusing this run')
    for completed in state['completed_phases'].values():
        check_files(completed['files'])
    ledgers = {}
    for resource in ('cpu', 'gpu'):
        path = root/f'{resource}-ledger.jsonl'
        if path.with_suffix('.active.json').exists():
            raise ValueError('A worker is still active')
        ledger_entries(path)
        ledgers[str(path)] = digest(path)
    durable_json(root/'previous-run.json', dict(identity=identity, state=state, ledger_hashes=ledgers))
    identity.update(code=code, git_commit=revision)
    identity['inputs'].update(inventory([root/'previous-run.json']))
    state.update(status='TRAINING_READY', operation='separate', identity=binding(identity))
    state.pop('error', None)
    state.pop('active_phase', None)
    durable_json(root/'identity.json', identity)
    durable_json(root/'state.json', state)
    check_files(ledgers)


def admit_schedule(state, entries, *, phase, operation=None):
    index = next(i for i, p in enumerate(PHASES) if p[0] == phase)
    resource = PHASES[index][1]
    used = sum(e['elapsed_seconds'] for e in entries[resource])
    phase_used = sum(e['elapsed_seconds'] for e in entries[resource] if e.get('phase') == phase)
    remaining = {n: state['reservations'][n] for n, r, _ in PHASES[index:] if r == resource
                 and (operation is None or n in operation)
                 and n not in state['completed_phases']}
    reserve = 120 if resource == 'gpu' else 0
    if (used+sum(remaining.values())+reserve > (7200 if resource == 'gpu' else 3600)
            or phase_used+remaining[phase] > 1800):
        raise ValueError('The complete remaining schedule does not fit the cumulative or phase budget')
    return dict(resource=resource, used_seconds=used, phase_used_seconds=phase_used,
                remaining=remaining, stop_reserve_seconds=reserve)


def environment(config, root):
    return dict(CC=str((Path(config['toolchain'])/'bin/cc').resolve()),
                CXX=str((Path(config['toolchain'])/'bin/c++').resolve()),
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                WANDB_DISABLED='true', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
                HF_HOME=str((root/'cache/huggingface').resolve()),
                UNSLOTH_COMPILE_LOCATION=str((root/'cache/unsloth').resolve()),
                TRITON_CACHE_DIR=str((root/'cache/triton').resolve()), PYTHONUNBUFFERED='1')
