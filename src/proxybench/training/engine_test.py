"""Run the pinned dedicated-server panel within the existing optimization ledger."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from proxybench.execution.resources import durable_json, ledger_entries, supervise
from proxybench.extraction.llama_server import generate, json_request, require_prompt_tokens
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.optimization import PANEL, TIMING, admit
from proxybench.training.smoke import accepted_rows, digest, generation_input, prepare, read_json


def phase_name(mode, attempt):
    if not re.fullmatch(r'[a-z0-9-]+', attempt):
        raise ValueError('Invalid attempt identifier')
    return 'engine-' + mode + ('' if attempt == '1' else '-' + attempt)


def worker(root, mode, config, attempt='1'):
    from transformers import AutoTokenizer
    output = root / phase_name(mode, attempt)
    output.mkdir(exist_ok=False)
    training = read_json(Path('configs/qwen35-4b-optimization.json').read_bytes())
    rows, _ = accepted_rows(Path('data/annotations/direct-sol-15-v1/accepted-labels.jsonl'), training)
    validate_checkpoint(root / 'train-save/adapter')
    completed = read_json((root / 'engine-resume/gguf-complete.json').read_bytes())
    if (completed['status'] != 'COMPLETE' or completed['source_commit'] != config['source_commit']
            or completed['precision'] != config['precision'] or digest(completed['path']) != completed['sha256']):
        raise ValueError('GGUF identity differs')
    pin = read_json((root / 'engine-resume/runtime-pin.json').read_bytes())
    for path, sha256 in pin['files'].items():
        if digest(path) != sha256:
            raise ValueError(f'Runtime file hash differs: {path}')
    tokenizer = AutoTokenizer.from_pretrained(str(root / 'train-save/adapter'), local_files_only=True)
    items, bounds = prepare(tokenizer, rows, training)
    earlier = read_json((root / 'reload-panel/prepared.json').read_bytes())
    if items != earlier['items'] or bounds != earlier['bounds']:
        raise ValueError('Saved prompt sequences differ')
    if mode == 'final':
        approval = read_json((root / 'engine-resume/panel-approved.json').read_bytes())
        if approval['configuration'] != config or approval['gguf_sha256'] != completed['sha256']:
            raise ValueError('Final candidate differs from reviewed panel')
        for name, sha256 in approval['files'].items():
            if digest(name) != sha256:
                raise ValueError('Panel evidence differs')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str((root / 'engine-setup/binary/llama-server').resolve()), '-m', completed['path'],
               '--host', '127.0.0.1', '--port', str(port), *config['server_arguments']]
    environment = dict(os.environ, LD_LIBRARY_PATH=pin['library_path'])
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    durable_json(output / 'launch.json', dict(command=command, configuration=config, gguf_sha256=completed['sha256'],
                 runtime_pin_sha256=digest(root / 'engine-resume/runtime-pin.json'), context=bounds, port=port,
                 library_path=pin['library_path'], python=sys.version))
    process = None
    try:
        with (output / 'server.log').open('xb') as log:
            start = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
            durable_json(output / 'server-owner.json', dict(pid=process.pid, worker_pid=os.getpid(), process_group=os.getpgrp()))
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server startup failed with code {process.returncode}')
                if time.monotonic() - start > config['startup_seconds']:
                    raise TimeoutError('Server startup deadline')
                try:
                    if json_request(port, '/health', timeout=1).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            durable_json(output / 'startup.json', dict(seconds=time.monotonic()-start, props=json_request(port, '/props')))
            text = (output / 'server.log').read_text(errors='replace')
            placement = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', text)
            if not placement or int(placement[-1][0]) != int(placement[-1][1]):
                raise ValueError('Full GPU layer placement was not established')
            agreements = []
            for index, item in enumerate(items):
                prompt = generation_input(item)
                rendered = tokenizer.apply_chat_template([rows[index]['messages'][0]], tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                require_prompt_tokens(port, rendered, prompt)
                agreements.append(dict(index=index, prompt_token_ids=prompt, rendered_prompt=rendered))
            durable_json(output / 'prompt-agreement.json', dict(all_exact=True, rows=agreements))

            def request(index, label, maximum=1792, forced=False):
                result = generate(port, tokenizer, generation_input(items[index]), output / f'{label}-{index}.json',
                                  index=index, maximum=maximum, forced=forced,
                                  deadline=config['probe_request_seconds'] if forced else config['natural_request_seconds'],
                                  expected=read_json(rows[index]['messages'][1]['content']))
                print(json.dumps(dict(index=index, label=label, status=result['status'], format_valid=result['format_valid'],
                                      timing=result['timing'])), flush=True)
                return result

            def warm(index, label):
                if request(index, label+'-warmup', 64, True)['status'] != 'PROBE_COMPLETE':
                    raise ValueError('Engine warmup failed')

            if mode == 'panel':
                for index in TIMING:
                    warm(index, 'probe')
                    for repeat in range(2):
                        if request(index, f'probe{repeat}', 256, True)['status'] != 'PROBE_COMPLETE':
                            raise ValueError('Engine forced probe failed')
            indices = (14, 1, 6, 2, 7, 12) if mode == 'panel' else tuple(i for i in range(15) if i not in PANEL)
            for index in indices:
                warm(index, 'candidate')
                result = request(index, 'candidate')
                if result['status'] != 'COMPLETE' or not result['format_valid']:
                    durable_json(output / 'candidate-failed.json', dict(index=index, status=result['status']))
                    return
            durable_json(output / 'complete.json', dict(status='COMPLETE', quality_review='PENDING', mode=mode))
    finally:
        if process is not None:
            begin = time.monotonic()
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            durable_json(output / 'server-cleanup.json', dict(pid=process.pid, returncode=process.returncode,
                                                           seconds=time.monotonic()-begin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['panel', 'final'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--estimate-seconds', type=float, required=True)
    parser.add_argument('--reserve-seconds', type=float, default=2100)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--attempt', default='1')
    args = parser.parse_args()
    config_path = Path('configs/qwen35-4b-llama.json')
    config = read_json(config_path.read_bytes())
    if args.worker:
        worker(args.run, args.mode, config, args.attempt)
        return
    limits = read_json(Path('configs/qwen35-4b-optimization.json').read_bytes())['limits']
    ledger = args.run / 'gpu-ledger.jsonl'
    entries = ledger_entries(ledger)
    earlier = sum(e['elapsed_seconds'] for e in entries if Path(e['run']).name.startswith('engine-' + args.mode))
    limits['phase_seconds'] -= earlier
    admission = admit(args.estimate_seconds, used_seconds=sum(e['elapsed_seconds'] for e in entries),
                      phase_seconds=limits['phase_seconds'], reserve_seconds=args.reserve_seconds)
    name = phase_name(args.mode, args.attempt)
    output = args.run / (name + '-launch')
    output.mkdir(exist_ok=False)
    durable_json(output / 'admission.json', admission)
    shutil.copyfile(config_path, output / config_path.name)
    for package in ['execution', 'extraction', 'training']:
        for source in Path('src/proxybench', package).glob('*.py'):
            target = output / package / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
    command = [sys.executable, '-m', 'proxybench.training.engine_test', args.mode, '--run', str(args.run),
               '--estimate-seconds', str(args.estimate_seconds), '--reserve-seconds', str(args.reserve_seconds),
               '--attempt', args.attempt, '--worker']
    result = supervise(command, args.run / (name + '-supervisor'), limits, ledger=ledger)
    print(result, flush=True)
    if result != 'EXITED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
