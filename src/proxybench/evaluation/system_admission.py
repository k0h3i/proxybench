"""Bind new evaluation code to a completed, unchanged historical training run."""

from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys

from proxybench.execution.resources import durable_json, ledger_entries
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.historical import binding, check_files, inventory, runtime_files
from proxybench.training.smoke import digest, generation_input, read_json
from proxybench.training.labels import sha

SCORER = 'historical-system-v1'
EVALUATION_CODE = {
    'src/proxybench/evaluation/pilot_review.py',
    'src/proxybench/evaluation/system_admission.py',
    'src/proxybench/evaluation/system_labels.py',
    'src/proxybench/training/historical.py',
    'src/proxybench/training/historical_run.py',
    'src/proxybench/execution/live.py',
    'src/proxybench/extraction/llama_server.py',
    'src/proxybench/training/historical_engine.py',
}


def ledger_prefixes(root):
    prefixes = {}
    for resource in ('cpu', 'gpu'):
        path = root/f'{resource}-ledger.jsonl'
        if path.with_suffix('.active.json').exists() or path.with_suffix('.lock').exists():
            raise ValueError('An execution ledger is active')
        ledger_entries(path)
        contents = path.read_bytes() if path.exists() else b''
        prefixes[resource] = dict(bytes=len(contents), sha256=sha(contents))
    return prefixes


def check_prefixes(root, prefixes):
    for resource, prefix in prefixes.items():
        path = root/f'{resource}-ledger.jsonl'
        contents = path.read_bytes() if path.exists() else b''
        if sha(contents[:prefix['bytes']]) != prefix['sha256'] or len(contents) < prefix['bytes']:
            raise ValueError('The prior resource ledger changed')
        ledger_entries(path)


def preserved_training(root, config, *, first_admission=False):
    identity = read_json((root/'identity.json').read_bytes())
    state = read_json((root/'state.json').read_bytes())
    if (config.get('profile') != 'historical-system-v1' or identity['configuration'] != config
            or state['identity'] != binding(identity) or state.get('operation') != 'separate'
            or 'training' not in state['completed_phases']):
        raise ValueError('Evaluation requires the completed, unchanged system-profile run')
    if first_admission and (state['status'] != 'TRAINED' or set(state['completed_phases']) != {'training'}):
        raise ValueError('Admit evaluation before its first GPU phase')
    if digest(root/'prepared.json') != identity['prepared_sha256']:
        raise ValueError('Prepared training input changed')
    if str(Path(sys.executable).absolute()) != identity['executable'] or sys.version != identity['python']:
        raise ValueError('Python runtime changed')
    for key in ('inputs', 'model', 'tokenizer'):
        check_files(identity[key])
    if runtime_files(config) != identity['runtime']:
        raise ValueError('Pinned runtime changed')
    check_files(state['completed_phases']['training']['files'])
    prepared = read_json((root/'prepared.json').read_bytes())
    validate_checkpoint(root/'adapter', dict(run=state['identity'], order=binding(prepared['order']),
                                             completed=config['updates']))
    journal = read_json((root/'training-journal.json').read_bytes())
    if (journal.get('status') != 'TRAINED' or journal.get('completed') != config['updates']
            or journal.get('pending_step') is not None):
        raise ValueError('Training journal is not complete')
    return identity, state


def current_evaluation_code(identity):
    code = inventory(Path('src/proxybench').rglob('*.py'))
    changed = {name for name in set(code) | set(identity['code']) if code.get(name) != identity['code'].get(name)}
    if not changed <= EVALUATION_CODE or not changed:
        raise ValueError(f'Unreviewed source-code changes: {sorted(changed - EVALUATION_CODE)}')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for name, expected in code.items():
        if sha(subprocess.check_output(['git', 'show', f'{revision}:{name}'])) != expected:
            raise ValueError(f'Evaluation code is not committed: {name}')
    for name, expected in identity['code'].items():
        if sha(subprocess.check_output(['git', 'show', f'{identity["git_commit"]}:{name}'])) != expected:
            raise ValueError(f'Original training code is not retained: {name}')
    return code, revision


def admit_evaluation(root, config):
    root = Path(root)
    path = root/'evaluation-admission.json'
    if path.exists():
        raise FileExistsError(path)
    identity, state = preserved_training(root, config, first_admission=True)
    code, revision = current_evaluation_code(identity)
    admission = dict(schema='historical-system-development-v1', scorer=SCORER,
                     training_identity=state['identity'], training_commit=identity['git_commit'],
                     training_phase=state['completed_phases']['training'],
                     code=code, evaluation_commit=revision, ledger_prefixes=ledger_prefixes(root))
    durable_json(path, admission)
    return admission


def saved_baseline_prefix(output, prepared, expected_count):
    answers = {}
    for path in Path(output).glob('answer-*.json'):
        match = re.fullmatch(r'answer-(\d+)\.json', path.name)
        if match is None:
            raise ValueError('Unexpected baseline answer name')
        index = int(match.group(1))
        if index in answers or index >= expected_count:
            raise ValueError('Duplicate or out-of-range baseline answer')
        answer = read_json(path.read_bytes())
        if (answer.get('index') != index or answer.get('max_new_tokens') != 1792
                or answer.get('prompt_token_ids') != generation_input(prepared['items']['development'][index])):
            raise ValueError(f'Baseline answer {index} has a different request identity')
        answers[index] = answer
    complete = [index for index, answer in sorted(answers.items()) if answer.get('status') in ('COMPLETE', 'LENGTH_STOP')]
    if (not complete or complete != list(range(len(complete)))
            or set(answers) != set(range(len(complete)+1))
            or answers[len(complete)].get('status') not in ('STARTED', 'INTERRUPTED')
            or len(complete) >= expected_count):
        raise ValueError('The saved baseline answers do not form one resumable prefix')
    return len(complete)


def admit_timeout_recovery(root, config):
    """Bind a timed-out development attempt before removing its GPU time ceilings."""
    root = Path(root)
    recovery_path = root/'evaluation-timeout-recovery.json'
    snapshot_path = root/'baseline-timeout-state.json'
    identity, state = preserved_training(root, config)
    if recovery_path.exists() and state['status'] == 'EVALUATION_PAUSED':
        validate_evaluation(root, config)
        return read_json(recovery_path.read_bytes())
    if recovery_path.exists() and not snapshot_path.exists():
        raise ValueError('The timeout recovery lacks its failed-state snapshot')
    if snapshot_path.exists() and read_json(snapshot_path.read_bytes()) != state:
        raise ValueError('The saved failed state differs from the current run')
    original_path = root/'evaluation-admission.json'
    original = read_json(original_path.read_bytes())
    if (original['schema'] != 'historical-system-development-v1' or original['scorer'] != SCORER
            or original['training_identity'] != state['identity']
            or original['training_commit'] != identity['git_commit']
            or original['training_phase'] != state['completed_phases']['training']):
        raise ValueError('The first evaluation admission changed')
    check_prefixes(root, original['ledger_prefixes'])
    for name, expected in original['code'].items():
        if sha(subprocess.check_output(['git', 'show', f'{original["evaluation_commit"]}:{name}'])) != expected:
            raise ValueError(f'The first admitted evaluation commit changed: {name}')
    code, revision = current_evaluation_code(identity)
    if (state['status'] != 'FAILED' or state.get('failed_phase') != 'baseline'
            or state.get('supervisor_status') != 'PHASE_TIMEOUT'
            or state.get('attempts', {}).get('baseline') != 1
            or 'baseline' in state['completed_phases'] or state.get('partial_phases', {}).get('baseline')):
        raise ValueError('Recovery requires the first baseline attempt to end at its time ceiling')
    output = root/'phases/baseline-01'
    result_path = output/'supervisor/result.json'
    result = read_json(result_path.read_bytes())
    gpu = ledger_entries(root/'gpu-ledger.jsonl')
    if (result.get('status') != 'PHASE_TIMEOUT' or result.get('surviving_owned_pids')
            or not gpu or gpu[-1].get('execution_id') != result.get('execution_id')
            or gpu[-1].get('status') != 'PHASE_TIMEOUT'
            or gpu[-1].get('elapsed_seconds') != result.get('elapsed_seconds')
            or (output/'complete.json').exists()):
        raise ValueError('The timed-out baseline result or GPU ledger differs')
    prepared = read_json((root/'prepared.json').read_bytes())
    complete = saved_baseline_prefix(output, prepared, config['development_examples'])
    for completed in state['completed_phases'].values():
        check_files(completed['files'])
    partial = dict(output=str(output.resolve()),
                   files=inventory(p for p in output.rglob('*') if p.is_file()
                                   and 'supervisor' not in p.relative_to(output).parts))
    check_files(partial['files'])
    prefixes = ledger_prefixes(root)
    if not snapshot_path.exists():
        durable_json(snapshot_path, state)
    recovery = dict(schema='historical-system-timeout-recovery-v1', scorer=SCORER,
                    training_identity=state['identity'], first_admission_sha256=digest(original_path),
                    failed_state_sha256=digest(snapshot_path), failed_result_sha256=digest(result_path),
                    code=code, recovery_commit=revision, ledger_prefixes=prefixes,
                    partial=partial, complete_answers=complete,
                    gpu_time_ceiling_seconds=None)
    if recovery_path.exists():
        if read_json(recovery_path.read_bytes()) != recovery:
            raise ValueError('The saved timeout recovery admission differs')
    else:
        durable_json(recovery_path, recovery)
    resumed = deepcopy(state)
    resumed['status'] = 'EVALUATION_PAUSED'
    resumed.pop('active_phase', None)
    resumed['resource_limits'].update(gpu_total=None, gpu_phase=None,
                                      stop_reserve=None, test_reserve=None,
                                      evaluation_unbounded=True)
    resumed['reservations'].update(baseline=None, final=None)
    resumed.setdefault('partial_phases', {}).setdefault('baseline', []).append(partial)
    resumed['timeout_recovery_sha256'] = digest(recovery_path)
    durable_json(root/'state.json', resumed)
    return recovery


def validate_evaluation(root, config):
    root = Path(root)
    identity, state = preserved_training(root, config)
    admission = read_json((root/'evaluation-admission.json').read_bytes())
    if (admission['schema'] != 'historical-system-development-v1' or admission['scorer'] != SCORER
            or admission['training_identity'] != state['identity']
            or admission['training_commit'] != identity['git_commit']
            or admission['training_phase'] != state['completed_phases']['training']):
        raise ValueError('Evaluation admission or code changed')
    check_prefixes(root, admission['ledger_prefixes'])
    recovery_path = root/'evaluation-timeout-recovery.json'
    if recovery_path.exists():
        recovery = read_json(recovery_path.read_bytes())
        snapshot_path = root/'baseline-timeout-state.json'
        old = read_json(snapshot_path.read_bytes())
        if (recovery.get('schema') != 'historical-system-timeout-recovery-v1'
                or recovery.get('scorer') != SCORER
                or recovery.get('training_identity') != state['identity']
                or recovery.get('first_admission_sha256') != digest(root/'evaluation-admission.json')
                or recovery.get('failed_state_sha256') != digest(snapshot_path)
                or recovery.get('failed_result_sha256') != digest(root/'phases/baseline-01/supervisor/result.json')
                or old.get('status') != 'FAILED' or old.get('supervisor_status') != 'PHASE_TIMEOUT'
                or old.get('completed_phases') != {k: state['completed_phases'][k] for k in old['completed_phases']}
                or recovery.get('complete_answers') != saved_baseline_prefix(
                    recovery['partial']['output'], read_json((root/'prepared.json').read_bytes()),
                    config['development_examples'])
                or recovery.get('gpu_time_ceiling_seconds', 1) is not None
                or state.get('timeout_recovery_sha256') != digest(recovery_path)
                or state['resource_limits'].get('evaluation_unbounded') is not True
                or any(state['resource_limits'].get(k, 1) is not None
                       for k in ('gpu_total', 'gpu_phase', 'stop_reserve', 'test_reserve'))
                or any(state['reservations'].get(k, 1) is not None for k in ('baseline', 'final'))
                or recovery['partial'] not in state.get('partial_phases', {}).get('baseline', [])
                or inventory(Path('src/proxybench').rglob('*.py')) != recovery['code']):
            raise ValueError('Timeout recovery admission or code changed')
        check_prefixes(root, recovery['ledger_prefixes'])
        check_files(recovery['partial']['files'])
    elif inventory(Path('src/proxybench').rglob('*.py')) != admission['code']:
        raise ValueError('Evaluation admission or code changed')
    for completed in state['completed_phases'].values():
        check_files(completed['files'])
    for attempts in state.get('partial_phases', {}).values():
        for attempt in attempts:
            check_files(attempt['files'])
    return identity, state, admission
