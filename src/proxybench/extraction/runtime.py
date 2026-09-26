"""Run one pinned local GGUF model on marked source fragments."""
import json
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from proxybench.execution.resources import durable_json
from proxybench.extraction.llama_server import generate, json_request, require_prompt_tokens
from proxybench.training.adapters import digest

RESERVED = ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')


def load_config(path):
    value = json.loads(Path(path).read_text())
    def expand(item):
        if isinstance(item, str):
            return os.path.expandvars(os.path.expanduser(item))
        if isinstance(item, list):
            return [expand(x) for x in item]
        if isinstance(item, dict):
            return {expand(k): expand(v) for k, v in item.items()}
        return item
    value = expand(value)
    import math
    limits = value.get('limits', {})
    for name in ('phase_seconds', 'total_seconds'):
        number = limits.get(name)
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError('Runtime time limits must be finite positive numbers')
    for name in ('startup_seconds', 'natural_request_seconds'):
        if name in value and (type(value[name]) not in (int, float) or not math.isfinite(value[name]) or value[name] <= 0):
            raise ValueError('Request time limits must be finite positive numbers')
    return value


def source_messages(text, config):
    if not isinstance(text, str) or len(text.encode()) > 1024 * 1024:
        raise ValueError('Source input must contain at most 1 MiB of text')
    if any(marker in text for marker in RESERVED):
        raise ValueError('Source contains a reserved model token')
    if (text.count('BEGIN MARKED TARGET') != 1 or text.count('END MARKED TARGET') != 1
            or text.index('BEGIN MARKED TARGET') >= text.index('END MARKED TARGET')):
        raise ValueError('Source must contain exactly one ordered marked target')
    return [dict(role='system', content=Path(config['system_prompt']).read_text()),
            dict(role='user', content=text)]


def prompt_tokens(tokenizer, messages, config):
    if [m.get('role') for m in messages] != ['system', 'user']:
        raise ValueError('Inference requires system and user messages')
    if messages != source_messages(messages[1]['content'], config):
        raise ValueError('Inference system prompt differs from the canonical prompt')
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=False)
    if len(tokens) > config['input_tokens'] or len(tokens) + config['response_tokens'] > config['context_tokens']:
        raise ValueError('Source exceeds the token limit. Select a smaller source range.')
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
    return tokens, rendered


def _read_runtime_file(path, reader):
    try:
        return reader(path)
    except OSError as error:
        problem = 'Missing runtime file' if isinstance(error, FileNotFoundError) else 'Cannot read runtime file'
        raise ValueError(f'{problem}: {path}. Follow docs/preparation.md#install-llamacpp-and-its-cuda-libraries.') from None


def runtime_identity(config):
    """Authenticate local runtime files without importing model packages."""
    manifest_path = Path(config['runtime_manifest'])
    manifest = json.loads(_read_runtime_file(manifest_path, Path.read_text))
    if manifest['source_commit'] != config['source_commit']:
        raise ValueError('Runtime source revision differs')
    root = manifest_path.resolve().parent
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _read_runtime_file(path, digest) != expected:
            raise ValueError('Runtime file differs from its manifest')
    server = Path(config['server']).resolve()
    if not server.is_file():
        raise ValueError(f'Missing server executable: {server}. Follow docs/preparation.md#install-llamacpp-and-its-cuda-libraries.')
    if not server.is_relative_to(root) or str(server.relative_to(root)) not in manifest['files']:
        raise ValueError('Server executable is not bound to the runtime manifest')
    for name, expected in config.get('external_libraries', {}).items():
        if _read_runtime_file(name, digest) != expected:
            raise ValueError('External runtime library differs from its pinned hash')
    return _read_runtime_file(manifest_path, digest)


@contextmanager
def model_server(output_dir, config, *, startup_deadline=None, prepare=None):
    """Keep one authenticated server alive under the caller's supervision."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = runtime_identity(config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], local_files_only=True, trust_remote_code=False)
    if prepare is not None:
        prepare(tokenizer)
    if startup_deadline is not None and time.monotonic() >= startup_deadline:
        raise TimeoutError('Model preparation exceeded its startup time limit')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    arguments = config['server_arguments']
    if any(value.split('=', 1)[0] in {'--host', '--port', '-m', '--model', '--api-key', '--api-key-file'} for value in arguments):
        raise ValueError('Server arguments cannot replace runtime bindings')
    command = [config['server'], '-m', config['model'], '--host', '127.0.0.1', '--port', str(port), *arguments]
    durable_json(output/'runtime.json', dict(runtime_manifest_sha256=identity, model_sha256=digest(config['model']), command=command))
    if startup_deadline is not None and time.monotonic() >= startup_deadline:
        raise TimeoutError('Model authentication exceeded its startup time limit')
    process = None
    try:
        with (output/'server.log').open('xb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log,
                                       env=dict(os.environ, LD_LIBRARY_PATH=config['library_path']))
            began = time.monotonic()
            deadline = startup_deadline if startup_deadline is not None else began + config['startup_seconds']
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Model server stopped during startup')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Model server startup exceeded its time limit')
                try:
                    if json_request(port, '/health', timeout=min(1, remaining)).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            if time.monotonic() >= deadline:
                raise TimeoutError('Model server startup exceeded its time limit')
            yield port, tokenizer
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def normalize_length_stop(answer, config):
    """Recognize a verified response that used the entire output allowance."""
    terminal = answer.get('terminal') or {}
    if (answer['status'] == 'TERMINATION_MISMATCH' and terminal.get('stop_type') == 'limit'
            and terminal.get('tokens_predicted') == len(answer['token_ids']) == config['response_tokens']):
        answer['status'] = 'LENGTH_STOP'
    return answer


def generate_answers(messages, output_dir, config):
    """Worker API. Its caller must provide process and memory supervision."""
    output = Path(output_dir)
    prompts = []
    def prepare(tokenizer):
        prompts.extend(prompt_tokens(tokenizer, row, config) for row in messages)
    with model_server(output, config, prepare=prepare) as (port, tokenizer):
        answers = []
        for index, (tokens, rendered) in enumerate(prompts):
            require_prompt_tokens(port, rendered, tokens)
            answer = generate(port, tokenizer, tokens, output/f'answer-{index}.json', index=index,
                              maximum=config['response_tokens'], deadline=config['natural_request_seconds'])
            normalize_length_stop(answer, config)
            durable_json(output/f'answer-{index}.json', answer)
            answers.append(answer)
        return answers


def supervised_generate_answers(messages, output_dir, config):
    """Run bounded inference and retain raw response files before scoring."""
    from proxybench.execution.live import supervise
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    durable_json(output/'request.json', dict(messages=messages, config=config))
    status = supervise([sys.executable, '-m', 'proxybench.training.runtime', '_infer', str(output/'request.json'), str(output/'answers')],
                       output/'execution', config['limits'], ledger=output/'resources.jsonl', phase='inference')
    if status != 'EXITED':
        raise RuntimeError(f'Inference worker stopped: {status}')
    return [json.loads((output/'answers'/f'answer-{i}.json').read_text()) for i in range(len(messages))]
