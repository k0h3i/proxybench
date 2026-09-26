"""Process an ordered evaluation session with one supervised model server."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

from proxybench.execution.resources import durable_json
from proxybench.extraction.llama_server import generate, require_prompt_tokens
from proxybench.extraction.runtime import model_server, normalize_length_stop, prompt_tokens


def phase(value):
    path = os.environ.get('PROXYBENCH_PHASE_FILE')
    if path:
        durable_json(path, dict(phase=value))


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Evaluation case exceeded its time limit')
    return seconds


@contextmanager
def supervised_deadline(deadline, request_path):
    """Publish the whole operation's deadline, including prompt preparation."""
    path = os.environ.get('PROXYBENCH_REQUEST_FILE')
    def publish(value):
        if path:
            durable_json(path, dict(request_path=str(request_path.resolve()), deadline_monotonic=value))
    publish(deadline)
    try:
        yield publish
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def failed_answer(path, index, tokenizer, exc):
    """Retain partial token capture when a request raises an ordinary error."""
    answer = json.loads(path.read_text()) if path.exists() else {}
    tokens = answer.setdefault('token_ids', [])
    answer.update(index=index, status='TIMEOUT' if isinstance(exc, TimeoutError) else 'FAILED',
                  format_valid=False, error=str(exc))
    answer.setdefault('text', tokenizer.decode(tokens[:-1] if tokens and tokens[-1] == tokenizer.eos_token_id
                                               else tokens, skip_special_tokens=False))
    answer.setdefault('raw', tokenizer.decode(tokens, skip_special_tokens=False))
    return answer


def run_session(directory):
    directory = Path(directory)
    request = json.loads((directory/'request.json').read_text())
    if request.get('schema') != 'evaluation-session-v1':
        raise ValueError('Unsupported evaluation session schema')
    config = request['config']
    output = directory/'answers'
    output.mkdir(exist_ok=True)
    startup_deadline = time.monotonic() + config['startup_seconds']
    phase('loading')
    with supervised_deadline(startup_deadline, directory/'request.json'):
        # Enter separately so the startup deadline ends before case deadlines begin.
        server = model_server(output, config, startup_deadline=startup_deadline)
        port, tokenizer = server.__enter__()
    try:
        phase('evaluation')
        for index, case in enumerate(request['cases']):
            deadline = time.monotonic() + config['limits']['phase_seconds']
            path = output/f'answer-{index}.json'
            with supervised_deadline(deadline, path) as publish:
                durable_json(directory/'progress.json', dict(index=index, state='STARTED'))
                try:
                    tokens, rendered = prompt_tokens(tokenizer, case['messages'], config)
                    require_prompt_tokens(port, rendered, tokens, timeout=min(10, remaining(deadline)))
                    seconds = min(config['natural_request_seconds'], remaining(deadline))
                    publish(min(deadline, time.monotonic() + seconds))
                    # generate manages a request deadline itself; retain this broader deadline
                    # through its final writes and restore the case deadline afterward.
                    deadline_file = os.environ.pop('PROXYBENCH_REQUEST_FILE', None)
                    try:
                        answer = generate(port, tokenizer, tokens, path, index=index,
                                          maximum=config['response_tokens'], deadline=seconds)
                    finally:
                        if deadline_file is not None:
                            os.environ['PROXYBENCH_REQUEST_FILE'] = deadline_file
                    publish(deadline)
                    remaining(deadline)
                    normalize_length_stop(answer, config)
                except Exception as exc:
                    answer = failed_answer(path, index, tokenizer, exc)
                durable_json(path, answer)
                durable_json(directory/'progress.json', dict(index=index, state='COMPLETE'))
            if answer['status'] not in {'COMPLETE', 'LENGTH_STOP'}:
                return 1
        return 0
    finally:
        server.__exit__(*sys.exc_info())


if __name__ == '__main__':
    raise SystemExit(run_session(sys.argv[1]))
