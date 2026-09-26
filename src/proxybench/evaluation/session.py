"""One supervised model lifetime with recoverable, independently bound answers."""
import json
import re
import sys
import uuid

from proxybench.execution.resources import ledger_entries
from proxybench.runstate import atomic_json, binding
from proxybench.training.labels import read_json, validate


INTERRUPTIONS = {'USER_STOP', 'FORCED_STOP', 'STOP_GRACE_EXHAUSTED',
                 'AGGREGATE_TIMEOUT', 'PHASE_TIMEOUT', 'RECONCILED'}


def terminal_answer(answer):
    return (isinstance(answer, dict) and isinstance(answer.get('status'), str)
            and answer['status'] not in {'', 'STARTED', 'INTERRUPTED'}
            and isinstance(answer.get('text'), str))


def session_request(inputs, config, cases):
    return dict(schema='evaluation-session-v1', identity=inputs['identity'], config=config,
                cases=[dict(id=case['id'], messages=case['messages']) for case in cases])


def failed_capture(folder, index, status):
    """Keep partial capture metadata when the supervisor ends an active request."""
    path = folder / 'answers' / f'answer-{index}.json'
    raw = read_json(path.read_text()) if path.exists() else {}
    tokens = []
    token_path = path.with_suffix('.tokens.jsonl')
    if token_path.exists():
        lines = token_path.read_text().splitlines()
        for position, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if position != len(lines) - 1:
                    raise ValueError('Invalid saved token stream')
                break
            tokens.append(event['token_id'])
    return {**raw, 'index': index, 'status': 'TIMEOUT' if status == 'REQUEST_TIMEOUT' else 'FAILED',
            'text': raw.get('text', ''), 'token_ids': tokens or raw.get('token_ids', []),
            'supervisor_status': status, 'format_valid': False}


def recover_sessions(run, inputs):
    """Call only after capture ownership and ledgers have been reconciled."""
    from proxybench.evaluation.workflow import answer_binding
    from proxybench.extraction.runtime import normalize_length_stop
    cases = {case['id']: case for case in inputs['cases']}
    positions = {case['id']: index for index, case in enumerate(inputs['cases'])}
    for record in run.state.get('evaluation_sessions', []):
        name = record['name']
        if not re.fullmatch(r'session-[0-9a-f]{32}', name):
            raise ValueError('Invalid evaluation session directory')
        folder = run.path / 'evaluation' / 'capture' / name
        request = read_json((folder / 'request.json').read_text())
        if binding(request) != record['request_sha256']:
            raise ValueError('Evaluation session manifest changed')
        requested = request['cases']
        ids = [item['id'] for item in requested]
        if (not ids or len(set(ids)) != len(ids) or any(key not in cases for key in ids)
                or ids != sorted(ids, key=positions.get)
                or request != session_request(inputs, run.state['configuration'], [cases[key] for key in ids])):
            raise ValueError('Evaluation session has stale inputs')
        entries = ledger_entries(folder / 'resources.jsonl')
        if len(entries) > 1:
            raise ValueError('An evaluation session must have one execution')
        elapsed = sum(entry['elapsed_seconds'] for entry in entries)
        status = entries[-1]['status'] if entries else None
        progress_path = folder / 'progress.json'
        progress = read_json(progress_path.read_text()) if progress_path.exists() else None
        if progress is not None and (set(progress) != {'index', 'state'}
                or type(progress['index']) is not int or not 0 <= progress['index'] < len(ids)
                or progress['state'] not in {'STARTED', 'COMPLETE'}):
            raise ValueError('Invalid evaluation session progress')
        if not entries and (progress is not None or (folder / 'answers').exists()):
            raise ValueError('Evaluation capture has no reconciled execution ledger')
        run.settle(name, elapsed)
        for index, key in enumerate(ids):
            path = folder / 'answers' / f'answer-{index}.json'
            raw = read_json(path.read_text()) if path.exists() else None
            if raw is not None and raw.get('index') != index:
                raise ValueError('Evaluation capture has a mismatched case index')
            completed = progress == dict(index=index, state='COMPLETE')
            answer = normalize_length_stop(dict(raw), request['config']) if terminal_answer(raw) else None
            if (answer is not None and answer['status'] in {'FAILED', 'TIMEOUT'}
                    and status in INTERRUPTIONS and not completed):
                answer = None
            active = progress == dict(index=index, state='STARTED')
            if answer is None and active and status is not None and status not in INTERRUPTIONS | {'EXITED'}:
                answer = failed_capture(folder, index, status)
            if answer is None:
                continue
            saved = dict(binding=answer_binding(inputs, cases[key], answer), answer=answer,
                         elapsed_seconds=0, session=name)
            result = run.path / 'evaluation' / 'results' / (binding(key) + '.json')
            if result.exists():
                previous = read_json(result.read_text())
                if previous['binding'] != saved['binding'] or previous['answer'] != answer:
                    raise ValueError('Conflicting evaluation captures')
            else:
                atomic_json(result, saved)
        if status is not None:
            run.state['last_evaluation_session'] = dict(name=name, status=status)


def generate_session(run, inputs):
    """Load the model once for the remaining cases in this evaluation attempt."""
    from proxybench.evaluation.workflow import load_answers
    from proxybench.execution.live import supervise
    answers = load_answers(run, inputs)
    missing = [case for case in inputs['cases'] if case['id'] not in answers]
    if not missing:
        return answers
    for case in missing:
        validate(case['reference'])
    remaining = run.state['resource_limit_seconds'] - run.state['consumed_seconds']
    if remaining <= 0:
        raise ValueError('The cumulative run time limit is exhausted')
    config = run.state['configuration']
    name = 'session-' + uuid.uuid4().hex
    folder = run.path / 'evaluation' / 'capture' / name
    folder.mkdir(parents=True, exist_ok=False)
    request = session_request(inputs, config, missing)
    atomic_json(folder / 'request.json', request)
    run.state.setdefault('evaluation_sessions', []).append(dict(name=name, request_sha256=binding(request)))
    run.save()
    run.reserve(name, remaining)
    limits = {**config['limits'], 'phase_seconds': remaining, 'total_seconds': remaining}
    try:
        supervise([sys.executable, '-m', 'proxybench.evaluation.worker', str(folder.resolve())],
                  folder / 'execution', limits, ledger=folder / 'resources.jsonl', phase='evaluation')
    finally:
        # A live owner blocks recovery and leaves the reservation charged.
        answers = load_answers(run, inputs)
    return answers
