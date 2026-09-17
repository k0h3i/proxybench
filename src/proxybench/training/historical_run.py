"""Prepare on CPU, then let the user launch the fixed historical pilot."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from proxybench.evaluation.pilot_review import accept_decisions, create_review, finish_report, require_decisions, source_text
from proxybench.evaluation.training_labels import export_comparison
from proxybench.execution.live import supervise
from proxybench.execution.resources import durable_json, ledger_entries
from proxybench.training.historical import (
    PHASES, admit_schedule, binding, check_files, environment, inventory, prepare_run,
    read_configuration, validate_identity,
)
from proxybench.training.smoke import digest, read_json
from proxybench.training.trajectory import require_clean_stop


def entries(root):
    return {resource: ledger_entries(root/f'{resource}-ledger.jsonl') for resource in ('cpu', 'gpu')}


def update_estimates(root, state, prepared):
    """Use observed loading, token lengths, updates, and request durations."""
    completed = state['completed_phases']
    reservations = state['reservations']
    if 'probe' in completed:
        output = Path(completed['probe']['output'])
        probe = read_json((output/'complete.json').read_bytes())
        trained = read_json((output/'training-result.json').read_bytes())
        recent_update = trained['history'][-1]['update_seconds']
        # Keep loading/compilation overhead and scale complete training sequences.
        longest = max(p['combined_tokens'] for p in prepared['items']['training'])
        total_length = sum(p['combined_tokens'] for p in prepared['items']['training'])*2
        dev_length = sum(p['combined_tokens'] for p in prepared['items']['development'])*3
        remaining_length = total_length
        journal = root/'training-journal.json'
        if journal.exists():
            progress = read_json(journal.read_bytes())
            remaining_length = sum(prepared['items']['training'][i]['combined_tokens']
                                   for i in prepared['order'][progress['completed']:])
        measured_training = 1.25*(probe['loading_seconds'] + recent_update*(remaining_length+dev_length)/longest + 90)
        reservations['training'] = measured_training
        rate = probe['python_output_tokens'][0]/probe['python_request_seconds'][0]
        panel_tokens = sum(min(1792, prepared['items']['training'][i]['response_tokens']+256) for i in prepared['panel'])
        reservations['original-export'] = 1.25*(probe['loading_seconds'] + panel_tokens/rate + 120)
    if 'original-export' in completed:
        result = read_json((Path(completed['original-export']['output'])/'complete.json').read_bytes())
        reservations['trained-export'] = 1.25*(result['loading_seconds']+sum(result['python_request_seconds'])+150)
    engine_measurement = next((n for n in ('baseline', 'original-panel') if n in completed), None)
    if engine_measurement:
        result = read_json((Path(completed[engine_measurement]['output'])/'complete.json').read_bytes())
        # Reserve using the slowest measured token rate and full output capacity.
        rate = min(t['output_tokens']/t['seconds'] for t in result['timings'] if t['output_tokens'])
        longest_observed_prompt = max(t['input_tokens'] for t in result['timings'])
        dev_prompt = max(p['input_tokens'] for p in prepared['items']['development'])
        per_request = 1792/rate*max(1, dev_prompt/longest_observed_prompt)
        reservations['baseline'] = 1.25*(result['loading_seconds']+24*per_request+10)
        reservations['final'] = reservations['baseline']
        reservations['trained-panel'] = 1.25*(result['loading_seconds']+3*per_request+10)
    durable_json(root/'state.json', state)


def gate_exports(root, state, prepared, phase_name):
    kind = phase_name.split('-')[0]
    left = Path(state['completed_phases'][kind+'-export']['output'])
    right = Path(state['completed_phases'][phase_name]['output'])
    metadata = [m for m in prepared['examples'] if m['split'] == 'training']
    cases, comparisons = [], {}
    for index in prepared['panel']:
        a, b = left/f'answer-{index}.json', right/f'answer-{index}.json'
        result = export_comparison(read_json(a.read_bytes()), read_json(b.read_bytes()))
        comparisons[str(index)] = result
        if result['status'] == 'FAILED':
            durable_json(root/f'{kind}-export-gate.json', comparisons)
            raise ValueError('Export behavior differs beyond the reviewed equivalence rules')
        if result['status'] == 'REVIEW_REQUIRED':
            cases.append(dict(id=str(index), source=source_text(metadata[index]), answers={'python': str(a), 'engine': str(b)}))
    durable_json(root/f'{kind}-export-gate.json', comparisons)
    if cases:
        review = create_review(root/f'review-{kind}', cases, identity=state['identity'], kind='export')
        state.update(status='REVIEW_REQUIRED', review=str(review.resolve()), next_phase=phase_name)
        durable_json(root/'state.json', state)
        print(f'REVIEW_REQUIRED: {review / "index.html"}', flush=True)
        return False
    return True


def create_final_review(root, state, prepared):
    metadata = [m for m in prepared['examples'] if m['split'] == 'development']
    cases = []
    for index, meta in enumerate(metadata):
        cases.append(dict(id=meta['packet']['manifest']['packet_id'], source=source_text(meta),
                          reference=read_json(prepared['rows']['development'][index]['messages'][1]['content']),
                          answers={model: str(Path(state['completed_phases'][phase_name]['output'])/f'answer-{index}.json')
                                   for model, phase_name in [('original', 'baseline'), ('trained', 'final')]}))
    review = create_review(root/'review-final', cases, identity=state['identity'], kind='evaluation')
    state.update(status='REVIEW_REQUIRED', review=str(review.resolve()), next_phase='report')
    durable_json(root/'state.json', state)
    print(f'REVIEW_REQUIRED: {review / "index.html"}', flush=True)


def validate_transition(action, state):
    required = {'run': {'PREPARED'}, 'continue': {'REVIEW_REQUIRED', 'PAUSED'}, 'resume': {'CLEAN_STOP'},
                'repair': {'FAILED', 'STOPPED'}}
    if state['status'] not in required[action]:
        raise ValueError(f'{action} cannot launch a run in state {state["status"]}')
    if action == 'repair' and (state.get('repairs', 0) >= 1 or state.get('failed_phase') == 'training'):
        raise ValueError('Only one nontraining mechanical repair is permitted')


def launch(root, config, configuration, action, repair_reason=None):
    identity, state = validate_identity(root, config)
    validate_transition(action, state)
    prepared = read_json((root/'prepared.json').read_bytes())
    if action == 'continue' and state['status'] == 'REVIEW_REQUIRED':
        review = Path(state['review'])
        require_decisions(review, state['identity'])
        if state.get('accepted_review_sha256') != digest(review/'accepted.json'):
            raise ValueError('The recorded review decision changed')
        if state['next_phase'] == 'report':
            state['status'] = finish_report(root, state, prepared)
            durable_json(root/'state.json', state)
            print(f'{state["status"]}: {root / "report.json"}', flush=True)
            return 0 if state['status'] == 'COMPLETE' else 1
    if action == 'resume':
        journal = read_json((root/'training-journal.json').read_bytes())
        require_clean_stop(journal, Path(journal['checkpoint']), dict(run=state['identity'], order=binding(prepared['order'])))
    if action == 'repair':
        if not repair_reason:
            raise ValueError('A mechanical repair requires a recorded reason')
        state.update(repairs=state.get('repairs', 0)+1, repair_reason=repair_reason)
    print('Schedule: '+ ' -> '.join(name for name, _, _ in PHASES), flush=True)
    print('Limits: 192 retained updates, 2 disposable probe updates, 120 GPU minutes, 60 CPU minutes, 30 minutes per phase.', flush=True)
    for name, resource, _ in PHASES:
        if name in state['completed_phases']:
            continue
        update_estimates(root, state, prepared)
        try:
            check_files(identity.get('code', {}))
            check_files(identity.get('inputs', {}))
            admission = admit_schedule(state, entries(root), phase=name)
            if shutil.disk_usage(root).free < config['minimum_disk_bytes']:
                raise ValueError('At least 60 GiB of free disk is required before a phase')
            if resource == 'gpu':
                device = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '--id=0'], text=True, timeout=5)
                if '3090' not in device:
                    raise ValueError('The pilot requires the local RTX 3090')
        except (ValueError, subprocess.SubprocessError) as exc:
            state.update(status='ADMISSION_REFUSED', failed_phase=name, error=str(exc))
            durable_json(root/'state.json', state)
            raise
        attempt = state['attempts'].get(name, 0)+1
        state['attempts'][name] = attempt
        output = root/'phases'/f'{name}-{attempt:02}'
        output.mkdir(parents=True, exist_ok=False)
        durable_json(output/'admission.json', admission)
        state.update(status='RUNNING', active_phase=name)
        durable_json(root/'state.json', state)
        command = [sys.executable, '-u', '-m', 'proxybench.training.historical_run', '_worker',
                   '--phase', name, '--configuration', str(configuration), '--run', str(root), '--output', str(output)]
        try:
            result = supervise(command, output/'supervisor', config['cpu_limits' if resource == 'cpu' else 'limits'],
                               ledger=root/f'{resource}-ledger.jsonl', phase=name, phase_used=admission['phase_used_seconds'],
                               environment=environment(config, root))
        except (ValueError, OSError, RuntimeError) as exc:
            state.update(status='FAILED', failed_phase=name, error=str(exc))
            durable_json(root/'state.json', state)
            raise
        complete = output/'complete.json'
        if result != 'EXITED' or not complete.exists():
            status = 'FAILED'
            if name == 'training' and result == 'EXITED' and (root/'training-journal.json').exists():
                journal = read_json((root/'training-journal.json').read_bytes())
                if journal['status'] == 'CLEAN_STOP':
                    require_clean_stop(journal, Path(journal['checkpoint']), dict(run=state['identity'], order=binding(prepared['order'])))
                    status = 'CLEAN_STOP'
            elif 'STOP' in result:
                status = 'STOPPED'
            if name in ('baseline', 'final', 'original-panel', 'trained-panel'):
                state.setdefault('partial_phases', {}).setdefault(name, []).append(
                    dict(output=str(output.resolve()), files=inventory(output.glob('answer-*'))))
            state.update(status=status, failed_phase=name, supervisor_status=result)
            durable_json(root/'state.json', state)
            print(f'{status}: {name}. Completed outputs and cumulative ledgers remain in {root}.', flush=True)
            return 130 if status in ('CLEAN_STOP', 'STOPPED') else 1
        if read_json(complete.read_bytes()).get('status') != 'COMPLETE':
            raise ValueError('Worker completion record is invalid')
        files = inventory(p for p in output.rglob('*') if p.is_file() and 'supervisor' not in p.relative_to(output).parts)
        if name == 'training':
            files.update(inventory((root/'adapter').rglob('*')))
        state['completed_phases'][name] = dict(output=str(output.resolve()), files=files)
        durable_json(root/'state.json', state)
        if name.endswith('panel'):
            try:
                if not gate_exports(root, state, prepared, name):
                    return 3
            except ValueError as exc:
                state.update(status='EXPORT_REJECTED', error=str(exc))
                durable_json(root/'state.json', state)
                raise
        supervisor_result = output/'supervisor/result.json'
        if supervisor_result.exists() and read_json(supervisor_result.read_bytes()).get('stop_requested'):
            state.update(status='PAUSED', next_phase=name)
            durable_json(root/'state.json', state)
            print('PAUSED after a complete phase. Use continue to start the remaining phases.', flush=True)
            return 130
    create_final_review(root, state, prepared)
    return 3


def status_report(root):
    state = read_json((root/'state.json').read_bytes())
    journal = read_json((root/'training-journal.json').read_bytes()) if (root/'training-journal.json').exists() else {}
    used = {k: sum(e['elapsed_seconds'] for e in v) for k, v in entries(root).items()}
    active = {}
    for resource in ('cpu', 'gpu'):
        path = root/f'{resource}-ledger.active.json'
        if path.exists():
            value = read_json(path.read_bytes())
            active[resource] = value['phase']
            used[resource] += max(0, time.monotonic()-value['started_monotonic'])
    probes = [read_json(p.read_bytes()) for p in (root/'phases').glob('probe-*/journal.json')]
    return dict(status=state['status'], active=active, completed_phases=list(state['completed_phases']),
                completed_updates=journal.get('completed', 0), pending_update=journal.get('pending_step'),
                disposable_probe_updates=sum(p['completed'] for p in probes),
                pending_probe_updates=[p['pending_step'] for p in probes if p.get('pending_step')],
                seconds=used, review=state.get('review'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'continue', 'resume', 'status', 'review', 'repair', '_worker'])
    parser.add_argument('--configuration', type=Path, default=Path('configs/qwen35-4b-historical-pilot.json'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--reason')
    parser.add_argument('--phase', choices=['prepare', *(n for n, _, _ in PHASES)])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.run.resolve()
    config = read_configuration(args.configuration)
    if args.action == '_worker':
        if not os.environ.get('PROXYBENCH_PHASE_FILE'):
            raise ValueError('Workers require the resource supervisor')
        if args.phase == 'prepare':
            prepare_run(root, config, args.configuration)
        elif args.phase.endswith('conversion'):
            from proxybench.training.historical_engine import convert
            convert(root, args.phase, args.output, config)
        elif args.phase.endswith('panel') or args.phase in ('baseline', 'final'):
            from proxybench.training.historical_engine import engine
            engine(root, args.phase, args.output, config)
        else:
            from proxybench.training.historical_worker import gpu_phase
            gpu_phase(root, args.phase, args.output, config)
        return 0
    if args.action == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError('Prepare this run before continuing')
    if args.action == 'status':
        print(json.dumps(status_report(root), indent=2))
        return 0
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            os.environ.update(environment(config, root))
            command = [sys.executable, '-u', '-m', 'proxybench.training.historical_run', '_worker', '--phase', 'prepare',
                       '--run', str(root), '--configuration', str(args.configuration)]
            result = supervise(command, root/'prepare-supervisor', config['cpu_limits'], ledger=root/'cpu-ledger.jsonl',
                               phase='prepare', environment=environment(config, root))
            if result != 'EXITED':
                raise ValueError(f'CPU preparation failed: {result}')
            print(f'PREPARED: {root}. No GPU worker started.', flush=True)
        elif args.action == 'review':
            _, state = validate_identity(root, config)
            if state['status'] != 'REVIEW_REQUIRED' or args.decisions is None:
                raise ValueError('A pending review and --decisions file are required')
            directory = Path(state['review'])
            accept_decisions(directory, read_json(args.decisions.read_bytes()), state['identity'])
            state['accepted_review_sha256'] = digest(directory/'accepted.json')
            durable_json(root/'state.json', state)
            print(f'Decisions saved. Model mapping: {directory / "revealed-mapping.json"}', flush=True)
        else:
            return launch(root, config, args.configuration, args.action, args.reason)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
