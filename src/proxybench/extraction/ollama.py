"""Stream frozen local requests under the existing resource supervisor."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from urllib.request import Request, urlopen

from proxybench.execution.runner import write_json
from proxybench.training.gpu_probe import phase


BASE = 'http://localhost:11434'
MODEL = 'qwen3.5:9b'
DIGEST = '6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7'


def api(path, payload=None, timeout=10):
    request = Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def capture_stream(response, output, expected_tokens, clock=time.monotonic):
    """Persist every received chunk, including output before failure or cancellation."""
    start = clock()
    first = None
    final = None
    with (output / 'stream.jsonl').open('xb') as stream, (output / 'raw.txt').open('x') as raw, \
            (output / 'thinking.txt').open('x') as thinking, (output / 'arrivals.jsonl').open('x') as arrivals:
        for line in response:
            elapsed = clock() - start
            stream.write(line)
            stream.flush()
            chunk = json.loads(line)
            arrivals.write(json.dumps({'elapsed_seconds': elapsed, 'bytes': len(line)}) + '\n')
            arrivals.flush()
            if 'error' in chunk:
                raise RuntimeError(chunk['error'])
            content = chunk.get('response', '')
            if content and first is None:
                first = elapsed
            raw.write(content)
            raw.flush()
            thinking.write(chunk.get('thinking', ''))
            thinking.flush()
            if chunk.get('done'):
                final = chunk
                break
    if final is None:
        raise RuntimeError('Stream ended without a completion event')
    write_json(output / 'final-event.json', final)
    tokens_match = final.get('prompt_eval_count') == expected_tokens
    return {'terminated': final.get('done_reason') == 'stop' and tokens_match,
            'input_tokens': final.get('prompt_eval_count'), 'expected_input_tokens': expected_tokens,
            'prompt_tokens_match': tokens_match, 'output_tokens': final.get('eval_count'),
            'done_reason': final.get('done_reason'), 'first_content_seconds_after_headers': first,
            'stream_seconds': clock() - start,
            'server_total_seconds': final.get('total_duration', 0) / 1e9,
            'server_generation_seconds': final.get('eval_duration', 0) / 1e9,
            'server_load_seconds': final.get('load_duration', 0) / 1e9}


def interrupted(signum, frame):
    raise KeyboardInterrupt('Request cancelled by signal ' + str(signum))


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupted)
    metadata = api('/api/tags')
    write_json(output / 'tags.json', metadata)
    selected = [m for m in metadata['models'] if m['name'] == MODEL and m['digest'] == DIGEST]
    if len(selected) != 1:
        raise ValueError('The pinned local Ollama model is unavailable')
    write_json(output / 'version.json', api('/api/version'))
    write_json(output / 'model.json', api('/api/show', {'model': MODEL}))
    directories = sorted(p for p in inputs.iterdir() if p.is_dir())
    if [p.name for p in directories] != [f'cal-{i:03d}' for i in range(1, 7)]:
        raise ValueError('The probe requires the frozen six inputs')
    try:
        for item in directories:
            phase('inference_' + item.name)
            target = output / item.name
            target.mkdir()
            request_bytes = (item / 'request.json').read_bytes()
            payload = json.loads(request_bytes)
            if payload['model'] != MODEL or payload.get('raw') is not True or payload.get('keep_alive') != 0:
                raise ValueError('Request must use pinned local model, raw prompt, and immediate unload')
            (target / 'request.json').write_bytes(request_bytes)
            binding = json.loads((item / 'binding.json').read_text())
            write_json(target / 'binding.json', binding)
            if hashlib.sha256(payload['prompt'].encode()).hexdigest() != binding['rendered_prompt_sha256']:
                raise ValueError('Frozen rendered prompt differs')
            print(item.name + ' started', flush=True)
            start = time.monotonic()
            request = Request(BASE + '/api/generate', data=request_bytes,
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=60) as response:
                capture = capture_stream(response, target, binding['input_tokens'])
            capture.update(input_id=item.name, elapsed_seconds=time.monotonic() - start,
                           prompt_sha256=binding['prompt_sha256'], runtime='ollama', model_digest=DIGEST)
            write_json(target / 'capture.json', capture)
            print(item.name + ' finished: ' + json.dumps(capture), flush=True)
            if not capture['prompt_tokens_match']:
                raise ValueError('Runtime prompt token count differs from the frozen complete prompt')
    except BaseException as error:
        write_json(output / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        # Disconnect cancels this request at the server. Explicit unload also bounds
        # residency after cancellation; never terminate the shared server process.
        try:
            write_json(output / 'unload.json', api('/api/generate', {'model': MODEL, 'keep_alive': 0}, timeout=2))
            write_json(output / 'processes-after.json', api('/api/ps', timeout=2))
        except Exception as error:
            write_json(output / 'cleanup-error.json', {'message': str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if 'PROXYBENCH_PHASE_FILE' not in os.environ:
        parser.error('Run under proxybench.execution.resources')
    run(args.inputs, args.output)


if __name__ == '__main__':
    main()
