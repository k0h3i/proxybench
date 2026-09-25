"""Bind new evaluation code to a completed, unchanged historical training run."""

from pathlib import Path
import subprocess
import sys

from proxybench.execution.resources import durable_json, ledger_entries
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.historical import binding, check_files, inventory, runtime_files
from proxybench.training.smoke import digest, read_json
from proxybench.training.labels import sha

SCORER = 'historical-system-v1'
EVALUATION_CODE = {
    'src/proxybench/evaluation/pilot_review.py',
    'src/proxybench/evaluation/system_admission.py',
    'src/proxybench/evaluation/system_labels.py',
    'src/proxybench/training/historical.py',
    'src/proxybench/training/historical_run.py',
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


def validate_evaluation(root, config):
    root = Path(root)
    identity, state = preserved_training(root, config)
    admission = read_json((root/'evaluation-admission.json').read_bytes())
    if (admission['schema'] != 'historical-system-development-v1' or admission['scorer'] != SCORER
            or admission['training_identity'] != state['identity']
            or admission['training_commit'] != identity['git_commit']
            or admission['training_phase'] != state['completed_phases']['training']
            or inventory(Path('src/proxybench').rglob('*.py')) != admission['code']):
        raise ValueError('Evaluation admission or code changed')
    check_prefixes(root, admission['ledger_prefixes'])
    for completed in state['completed_phases'].values():
        check_files(completed['files'])
    for attempts in state.get('partial_phases', {}).values():
        for attempt in attempts:
            check_files(attempt['files'])
    return identity, state, admission
