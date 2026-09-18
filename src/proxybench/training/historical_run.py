"""Run historical training and evaluation as separate user-launched operations."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from proxybench.evaluation.pilot_review import accept_decisions, create_review, finish_report, require_decisions, source_text
from proxybench.execution.live import supervise
from proxybench.execution.resources import durable_json, ledger_entries
from proxybench.training.historical import (
    PHASES, EVALUATION_PHASES, admit_schedule, binding, environment, inventory,
    prepare_run, read_configuration, reuse_prepared_run, validate_identity,
)
from proxybench.training.smoke import digest, read_json
from proxybench.training.trajectory import require_clean_stop


def entries(root):
    return {resource: ledger_entries(root/f'{resource}-ledger.jsonl') for resource in ('cpu', 'gpu')}


def create_final_review(root, state, prepared):
    metadata = [m for m in prepared['examples'] if m['split'] == 'development']
    cases = []
    for index, meta in enumerate(metadata):
        key = meta['packet']['manifest']['packet_id']
        cases.append(dict(id=key, source=source_text(meta),
                          reference=read_json(prepared['rows']['development'][index]['messages'][1]['content']),
                          answers={model: str(Path(state['completed_phases'][phase]['output'])/f'answer-{index}.json')
                                   for model, phase in [('original', 'baseline'), ('trained', 'final')]}))
    review = create_review(root/'review-final', cases, identity=state['identity'], kind='evaluation')
    state.update(status='REVIEW_REQUIRED', review=str(review.resolve()))
    durable_json(root/'state.json', state)
    print(f'REVIEW_REQUIRED: {review / "index.html"}', flush=True)


def validate_transition(action, state, updates=None):
    required = {'train': {'TRAINING_READY', 'CLEAN_STOP'}, 'resume': {'CLEAN_STOP'},
                'evaluate': {'TRAINED', 'EVALUATION_PAUSED', 'REVIEW_REQUIRED'}}
    if state.get('operation') != 'separate' or state['status'] not in required[action]:
        raise ValueError(f'{action} cannot launch a run in state {state["status"]}')
    if action == 'evaluate' and 'training' not in state['completed_phases']:
        count = f' all {updates}' if updates is not None else ''
        raise ValueError(f'Complete{count} training updates before evaluation')


def launch(root, config, configuration, action):
    _, state = validate_identity(root, config)
    validate_transition(action, state, config['updates'])
    prepared = read_json((root/'prepared.json').read_bytes())
    if action == 'evaluate' and state['status'] == 'REVIEW_REQUIRED':
        review = Path(state['review'])
        require_decisions(review, state['identity'])
        if state.get('accepted_review_sha256') != digest(review/'accepted.json'):
            raise ValueError('The recorded review decision changed')
        state['status'] = finish_report(root, state, prepared)
        durable_json(root/'state.json', state)
        print(f'{state["status"]}: {root / "report.json"}', flush=True)
        return 0 if state['status'] == 'COMPLETE' else 1
    training = action in ('train', 'resume')
    if state['status'] == 'CLEAN_STOP':
        journal = read_json((root/'training-journal.json').read_bytes())
        require_clean_stop(journal, Path(journal['checkpoint']), dict(run=state['identity'], order=binding(prepared['order'])))
    phases = ['training'] if training else list(EVALUATION_PHASES)
    print((f'Training: {config["training_examples"]} examples, {config["updates"]} updates, then save and exit.')
          if training else
          (f'Evaluation: original and trained models on the same '
           f'{config["development_examples"]} development examples.'), flush=True)
    print('Limits: 120 cumulative GPU minutes, 60 CPU minutes, 30 minutes per phase.', flush=True)
    for name in phases:
        if name in state['completed_phases']:
            continue
        resource = next(r for n, r, _ in PHASES if n == name)
        admission = admit_schedule(state, entries(root), phase=name, operation=phases)
        if shutil.disk_usage(root).free < config['minimum_disk_bytes']:
            raise ValueError('At least 60 GiB of free disk is required before a phase')
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
            if training and result == 'EXITED' and (root/'training-journal.json').exists():
                journal = read_json((root/'training-journal.json').read_bytes())
                if journal['status'] == 'CLEAN_STOP':
                    require_clean_stop(journal, Path(journal['checkpoint']), dict(run=state['identity'], order=binding(prepared['order'])))
                    status = 'CLEAN_STOP'
            state.update(status=status, failed_phase=name, supervisor_status=result)
            durable_json(root/'state.json', state)
            print(f'{status}: {name}. Saved outputs and resource ledgers remain in {root}.', flush=True)
            return 130 if status == 'CLEAN_STOP' else 1
        completion = read_json(complete.read_bytes())
        if completion.get('status') != 'COMPLETE' or (training and completion.get('updates') != config['updates']):
            state.update(status='FAILED', failed_phase=name, error='Invalid completion record')
            durable_json(root/'state.json', state)
            raise ValueError('Worker completion record is invalid')
        files = inventory(p for p in output.rglob('*') if p.is_file() and 'supervisor' not in p.relative_to(output).parts)
        if training:
            files.update(inventory((root/'adapter').rglob('*')))
        state['completed_phases'][name] = dict(output=str(output.resolve()), files=files)
        state['status'] = 'TRAINED' if training else 'EVALUATION_PAUSED'
        durable_json(root/'state.json', state)
        supervisor_result = output/'supervisor/result.json'
        if not training and supervisor_result.exists() and read_json(supervisor_result.read_bytes()).get('stop_requested'):
            print('EVALUATION_PAUSED after a complete phase. Run evaluate to continue.', flush=True)
            return 130
    if training:
        print(f'TRAINED: {config["updates"]} updates saved in {root / "adapter"}. Training ended.', flush=True)
        return 0
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
    return dict(status=state['status'], active=active, completed_phases=list(state['completed_phases']),
                completed_updates=journal.get('completed', 0), pending_update=journal.get('pending_step'),
                seconds=used, review=state.get('review'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate', 'resume', 'status', 'review', '_worker'])
    parser.add_argument('--configuration', type=Path, default=Path('configs/qwen35-4b-historical-pilot.json'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reuse-prepared', action='store_true')
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--phase', choices=['prepare', *(n for n, _, _ in PHASES)])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.run.resolve()
    config = read_configuration(args.configuration)
    if args.reuse_prepared and args.action != 'prepare':
        raise ValueError('--reuse-prepared applies only to CPU preparation')
    if args.action == '_worker':
        if not os.environ.get('PROXYBENCH_PHASE_FILE'):
            raise ValueError('Workers require the resource supervisor')
        if args.phase == 'prepare':
            prepare_run(root, config, args.configuration)
        elif args.phase.endswith('conversion'):
            from proxybench.training.historical_engine import convert
            convert(root, args.phase, args.output, config)
        elif args.phase in ('baseline', 'final'):
            from proxybench.training.historical_engine import engine
            engine(root, args.phase, args.output, config)
        else:
            from proxybench.training.historical_worker import gpu_phase
            gpu_phase(root, args.phase, args.output, config)
        return 0
    if args.action == 'prepare' and not args.reuse_prepared:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError('Prepare this run before continuing')
    if args.action == 'status':
        print(json.dumps(status_report(root), indent=2))
        return 0
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            if args.reuse_prepared:
                reuse_prepared_run(root, config)
            else:
                os.environ.update(environment(config, root))
                command = [sys.executable, '-u', '-m', 'proxybench.training.historical_run', '_worker', '--phase', 'prepare',
                           '--run', str(root), '--configuration', str(args.configuration)]
                result = supervise(command, root/'prepare-supervisor', config['cpu_limits'], ledger=root/'cpu-ledger.jsonl',
                                   phase='prepare', environment=environment(config, root))
                if result != 'EXITED':
                    raise ValueError(f'CPU preparation failed: {result}')
            print(f'TRAINING_READY: {root}. No GPU worker started.', flush=True)
        elif args.action == 'review':
            _, state = validate_identity(root, config)
            if state['status'] != 'REVIEW_REQUIRED' or args.decisions is None:
                raise ValueError('A pending review and --decisions file are required')
            directory = Path(state['review'])
            accept_decisions(directory, read_json(args.decisions.read_bytes()), state['identity'])
            state['accepted_review_sha256'] = digest(directory/'accepted.json')
            durable_json(root/'state.json', state)
            print('Decisions saved. Run evaluate to write the final report.', flush=True)
        else:
            return launch(root, config, args.configuration, args.action)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
