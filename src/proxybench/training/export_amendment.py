"""Apply the authorized quotation-review amendment to a stopped original panel."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

from proxybench.evaluation.training_labels import EXPORT_COMPARISON_VERSION, export_comparison
from proxybench.execution.resources import durable_json, group_members, ledger_entries
from proxybench.training.historical import binding, check_files, inventory, runtime_files
from proxybench.training.smoke import digest, read_json


# Only these reviewed v1 modules can change during the v2 migration.
PREVIOUS_CODE = {
    'src/proxybench/evaluation/training_labels.py': '0bc6bf8441e37712af771d534d99b7930a5eb43b4af07806559cb792f85927d9',
    'src/proxybench/training/historical_run.py': 'a5a03354d0aba743d6b3cb43b7b57dc0d89bde382d8602ed9e1c621b37638833',
    'src/proxybench/training/historical.py': '5c89e4552883b8243003a74a16d172b5e739dd5f93b3029c7736cd1078ae67c7',
}
ADDED_CODE = {'src/proxybench/training/export_amendment.py'}


def amended_code(identity):
    """Refuse unrelated edits and require the replacement modules to be committed."""
    if any(identity['code'].get(name) != expected for name, expected in PREVIOUS_CODE.items()):
        raise ValueError('This amendment requires the reviewed v1 comparison code')
    check_files({k: v for k, v in identity['code'].items() if k not in PREVIOUS_CODE})
    current = inventory(Path('src/proxybench').rglob('*.py'))
    if set(current) != set(identity['code']) | ADDED_CODE:
        raise ValueError('Unrelated source files changed')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    from proxybench.training.labels import sha
    for name in set(PREVIOUS_CODE) | ADDED_CODE:
        committed = subprocess.check_output(['git', 'show', f'{revision}:{name}'])
        if sha(committed) != current[name]:
            raise ValueError('Commit and test the amendment before applying it')
    return current, revision


def amend_export(root, config, reason):
    """CPU-only migration. Preserve prior identities, outputs, and ledger bytes."""
    if not reason or not reason.strip():
        raise ValueError('Record the authorization and reason with --reason')
    identity = read_json((root/'identity.json').read_bytes())
    state = read_json((root/'state.json').read_bytes())
    expected_phases = {'probe', 'original-export', 'original-conversion', 'original-panel'}
    if (state['status'] != 'EXPORT_REJECTED' or state.get('active_phase') != 'original-panel'
            or set(state['completed_phases']) != expected_phases
            or set(state['attempts']) != expected_phases or state.get('partial_phases')
            or (root/'training-journal.json').exists() or (root/'adapter').exists()
            or (root/'export-amendment').exists()):
        raise ValueError('Only a stopped v1 original comparison before training can be amended')
    if binding(identity) != state['identity'] or identity['configuration'] != config:
        raise ValueError('Run identity or configuration differs')
    if (digest(root/'prepared.json') != identity['prepared_sha256']
            or str(Path(sys.executable).absolute()) != identity['executable'] or sys.version != identity['python']):
        raise ValueError('Prepared sequences or Python runtime changed')
    for key in ('inputs', 'model', 'tokenizer'):
        check_files(identity[key])
    if runtime_files(config) != identity['runtime']:
        raise ValueError('Runtime file inventory changed')
    code, revision = amended_code(identity)
    for completed in state['completed_phases'].values():
        check_files(completed['files'])
    ledgers = {}
    for resource in ('cpu', 'gpu'):
        path = root/f'{resource}-ledger.jsonl'
        if path.with_suffix('.active.json').exists():
            raise ValueError('A resource ledger still has an active worker')
        for entry in ledger_entries(path):
            owner = Path(entry['run'])/'owner.json'
            if (entry.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                    and owner.exists() and group_members(read_json(owner.read_bytes())['pid'])):
                raise ValueError('An owned process group is still running')
        ledgers[str(path)] = digest(path)
    prepared = read_json((root/'prepared.json').read_bytes())
    comparisons = {}
    for index in prepared['panel']:
        answers = [read_json((Path(state['completed_phases'][phase]['output'])/f'answer-{index}.json').read_bytes())
                   for phase in ('original-export', 'original-panel')]
        comparisons[str(index)] = export_comparison(*answers)
    if (any(c['status'] == 'FAILED' for c in comparisons.values())
            or not any(c.get('reason') == 'Malformed quotation comparison' for c in comparisons.values())):
        raise ValueError('Saved answers do not qualify for the quotation-review amendment')
    archive = root/'export-amendment'
    archive.mkdir()
    for name in ('identity.json', 'state.json', 'original-export-gate.json', 'cpu-ledger.jsonl', 'gpu-ledger.jsonl'):
        with (archive/name).open('xb') as stream:
            stream.write((root/name).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    updated = deepcopy(identity)
    updated['inputs'].update(inventory(archive.iterdir()))
    updated.update(code=code, git_commit=revision, amendment=dict(rule=EXPORT_COMPARISON_VERSION,
                   reason=reason, previous_identity=state['identity'], evidence=inventory(archive.iterdir())))
    new_state = deepcopy(state)
    new_state.update(identity=binding(updated), amendment=EXPORT_COMPARISON_VERSION)
    new_state.pop('error', None)
    durable_json(archive/'record.json', dict(rule=EXPORT_COMPARISON_VERSION, reason=reason,
                 previous_identity=state['identity'], new_identity=new_state['identity'],
                 ledger_hashes=ledgers, comparisons=comparisons))
    durable_json(root/'identity.json', updated)
    durable_json(root/'state.json', new_state)
    check_files(ledgers)
    print('Export amendment recorded. Saved outputs and resource ledgers are unchanged. No GPU worker started.', flush=True)
