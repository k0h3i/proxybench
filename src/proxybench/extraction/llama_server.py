"""Capture native llama.cpp responses with explicit prompt and cache controls."""
import http.client
import json
import os
from pathlib import Path
import time

from proxybench.execution.resources import durable_json
from proxybench.extraction.measured_generation import timing_summary
from proxybench.training.labels import read_json, validate


def request_body(prompt, maximum, forced):
    if not prompt or any(type(token) is not int for token in prompt):
        raise ValueError('A nonempty integer-token prompt is required')
    return dict(prompt=prompt, n_predict=maximum, stream=True, return_tokens=True,
                cache_prompt=False, n_cache_reuse=0, temperature=0, top_k=1,
                top_p=1, min_p=0, typical_p=1, repeat_penalty=1, repeat_last_n=0,
                presence_penalty=0, frequency_penalty=0, seed=42, stop=[],
                ignore_eos=forced, samplers=['temperature'], n_keep=0)


def assess_response(tokens, terminal, *, prompt_length, eos, maximum, forced, elapsed, deadline):
    if elapsed > deadline:
        return 'TIMEOUT'
    if terminal is None or terminal.get('stop') is not True:
        return 'CAPTURE_INCOMPLETE'
    timing = terminal.get('timings', {})
    if (terminal.get('truncated') is not False or terminal.get('tokens_evaluated') != prompt_length
            or timing.get('cache_n') != 0 or timing.get('prompt_n') != prompt_length):
        return 'PROMPT_OR_CACHE_MISMATCH'
    if terminal.get('tokens_predicted') != len(tokens) or len(tokens) > maximum:
        return 'TOKEN_COUNT_MISMATCH'
    if forced:
        return 'PROBE_COMPLETE' if len(tokens) == maximum and terminal.get('stop_type') == 'limit' else 'PROBE_INCOMPLETE'
    return 'COMPLETE' if terminal.get('stop_type') == 'eos' and tokens and tokens[-1] == eos else 'TERMINATION_MISMATCH'


def json_request(port, route, value=None, *, timeout=10):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        raw = None if value is None else json.dumps(value, ensure_ascii=False).encode()
        connection.request('GET' if value is None else 'POST', route, body=raw,
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise ValueError(f'Server HTTP {response.status}: {body[:300]!r}')
        return json.loads(body)
    finally:
        connection.close()


def require_prompt_tokens(port, text, tokens):
    response = json_request(port, '/tokenize', dict(content=text, add_special=False, parse_special=True))
    if response.get('tokens') != tokens:
        raise ValueError('Server prompt token IDs differ from the saved Python prompt')
    return response


def generate(port, tokenizer, prompt, path, *, index, maximum=1792, forced=False, deadline=60, expected=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    body = request_body(prompt, maximum, forced)
    wire = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    path.with_suffix('.request.bin').write_bytes(wire)
    started = time.monotonic()
    request = dict(index=index, status='STARTED', prompt_token_ids=prompt, max_new_tokens=maximum,
                   force_length=forced, fresh_request_state=True, cache_prompt=False,
                   eos_token_id=tokenizer.eos_token_id, deadline_seconds=deadline,
                   started_monotonic=started, exact_token_arrivals=False,
                   maximum_unsaved_tokens=maximum,
                   capture_policy='Flush each SSE line and token group; arrival times belong to chunks')
    durable_json(path, request)
    deadline_file = os.environ.get('PROXYBENCH_REQUEST_FILE')
    if deadline_file:
        durable_json(deadline_file, dict(request_path=str(path.resolve()), deadline_monotonic=started + deadline))
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=deadline)
    tokens, events, terminal, max_chunk_tokens = [], [], None, 0
    try:
        with path.with_suffix('.response.bin').open('xb') as raw, path.with_suffix('.tokens.jsonl').open('x') as saved:
            connection.request('POST', '/completion', body=wire, headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f'Generation HTTP {response.status}: {response.read(1000)!r}')
            while True:
                remaining = started + deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Client request deadline')
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                line = response.readline()
                if not line:
                    break
                raw.write(line)
                raw.flush()
                if not line.startswith(b'data: '):
                    continue
                if line.strip() == b'data: [DONE]':
                    break
                chunk = json.loads(line[6:])
                if 'error' in chunk:
                    raise ValueError(str(chunk['error']))
                arrived = time.monotonic() - started
                ids = chunk.get('tokens', [])
                if not isinstance(ids, list) or any(type(t) is not int for t in ids):
                    raise ValueError('Invalid returned token IDs')
                max_chunk_tokens = max(max_chunk_tokens, len(ids))
                for token in ids:
                    event = dict(token_id=token, seconds=arrived)
                    events.append(event)
                    tokens.append(token)
                    saved.write(json.dumps(event) + '\n')
                saved.flush()
                if chunk.get('stop') is True:
                    terminal = chunk
                    break
        elapsed = time.monotonic() - started
        status = assess_response(tokens, terminal, prompt_length=len(prompt), eos=tokenizer.eos_token_id,
                                 maximum=maximum, forced=forced, elapsed=elapsed, deadline=deadline)
        text = tokenizer.decode(tokens[:-1] if tokens and tokens[-1] == tokenizer.eos_token_id else tokens,
                                skip_special_tokens=False)
        error, parsed = None, None
        try:
            parsed = read_json(text)
            validate(parsed)
        except (ValueError, TypeError) as exc:
            error = str(exc)
        answer = dict(**request)
        answer.update(status=status, token_ids=tokens, text=text, raw=tokenizer.decode(tokens, skip_special_tokens=False),
                      terminal=terminal, terminal_stream_event=terminal is not None, format_error=error,
                      format_valid=status == 'COMPLETE' and error is None,
                      exact_target=status == 'COMPLETE' and error is None and parsed == expected,
                      timing=timing_summary(events, elapsed), maximum_observed_chunk_tokens=max_chunk_tokens)
        # A text chunk can contain multiple tokens. These are chunk-derived windows.
        answer['timing']['decode_tokens_per_second'] = None
        durable_json(path, answer)
        return answer
    except BaseException as exc:
        durable_json(path, {**request, 'status': 'TIMEOUT' if isinstance(exc, TimeoutError) else 'FAILED',
                           'token_ids': tokens, 'format_valid': False, 'terminal_stream_event': False,
                           'timing': timing_summary(events, time.monotonic() - started), 'error': str(exc)})
        raise
    finally:
        connection.close()
        if deadline_file:
            Path(deadline_file).unlink(missing_ok=True)
