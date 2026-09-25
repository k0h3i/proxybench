"""Convert and evaluate both pilot models with the same pinned local engine."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from proxybench.execution.resources import durable_json, host_memory
from proxybench.extraction.llama_server import generate, json_request, require_prompt_tokens
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.historical import check_files
from proxybench.training.historical_worker import phase, stopping
from proxybench.training.smoke import digest, generation_input, read_json


def phase_output(state, name):
    return Path(state['completed_phases'][name]['output'])


def convert(root, name, output, config):
    phase('conversion')
    state = read_json((root/'state.json').read_bytes())
    kind = name.split('-')[0]
    model = phase_output(state, kind+'-export')/'model'
    manifest = validate_checkpoint(model)
    largest = max(math.prod(v['shape'])*(2 if v['dtype'] == 'torch.bfloat16' else 4) for v in manifest['tensors'].values())
    if host_memory()['available_bytes'] < max(4*1024**3, 8*largest+1024**3):
        raise ValueError('Conversion memory estimate does not fit')
    if shutil.disk_usage(output).free < max(config['minimum_disk_bytes'], 3*manifest['identity']['total_tensor_bytes']+10*1024**3):
        raise ValueError('Conversion disk estimate does not fit')
    source = Path(config['engine_root'])/'source/llama.cpp-b10909-mix-bea84f7'
    temporary = output/'model.incomplete.gguf'
    target = output/'model-bf16.gguf'
    if temporary.exists() or target.exists():
        raise FileExistsError(temporary)
    scratch = output/'conversion-temp'
    scratch.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', TMPDIR=str(scratch.resolve()))
    command = [sys.executable, str(source/'convert_hf_to_gguf.py'), str(model), '--outfile', str(temporary),
               '--outtype', 'bf16', '--use-temp-file', '--no-mtp']
    durable_json(output/'command.json', dict(command=command, source_manifest_sha256=digest(model/'manifest.json')))
    # Inherit the worker's streams and process group. Supervision owns conversion too.
    subprocess.run(command, env=env, check=True)
    phase('payload-inspection')
    sys.path[:0] = [str(source), str(source/'gguf-py')]
    import numpy as np
    import gguf
    from safetensors import safe_open
    from conversion import get_model_class
    converter = object.__new__(get_model_class('Qwen3_5ForCausalLM', mmproj=False))
    converter.hparams = read_json((model/'config.json').read_bytes())
    converter.block_count = converter.hparams['num_hidden_layers']
    converter.tensor_map = gguf.get_tensor_name_map(converter.model_arch, converter.block_count)
    converter.fuse_qkv = False
    converter.fuse_gate_up_exps = False
    reader = gguf.GGUFReader(temporary)
    actual = {t.name: t for t in reader.tensors}
    if len(actual) != len(reader.tensors):
        raise ValueError('Duplicate converted tensor')
    index = read_json((model/'model.safetensors.index.json').read_bytes())['weight_map']
    checked = {}
    for name, shard in index.items():
        with safe_open(model/shard, framework='pt', device='cpu') as stream:
            original = stream.get_tensor(name).float()
            block = int(name.split('.')[2]) if name.startswith('model.layers.') else None
            for new_name, transformed in converter.modify_tensors(original, name, block):
                if new_name in checked or new_name not in actual:
                    raise ValueError('Unexpected converted tensor inventory')
                tensor = actual[new_name]
                if (tensor.tensor_type not in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32)
                        or list(tensor.shape) != list(reversed(transformed.shape))):
                    raise ValueError('Converted tensor precision or shape differs')
                wanted = gguf.quants.quantize(transformed.numpy(), tensor.tensor_type)
                expected = hashlib.sha256(memoryview(np.ascontiguousarray(wanted)).cast('B')).hexdigest()
                found = hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()
                if found != expected:
                    raise ValueError('Converted payload differs from the pinned transformation')
                flat = tensor.data.reshape(-1)
                for start in range(0, len(flat), 65536):
                    if not np.isfinite(gguf.quants.dequantize(flat[start:start+65536], tensor.tensor_type)).all():
                        raise ValueError('Nonfinite converted tensor')
                checked[new_name] = dict(dtype=tensor.tensor_type.name, shape=[int(v) for v in tensor.shape], sha256=found)
                del wanted, transformed
            del original
    if set(checked) != set(actual):
        raise ValueError('Incomplete converted tensor inspection')
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    file_hash = digest(temporary)
    temporary.rename(target)
    durable_json(output/'complete.json', dict(status='COMPLETE', path=str(target.resolve()), sha256=file_hash,
                 source_manifest_sha256=digest(model/'manifest.json'), tensors=checked,
                 exact_transformed_payloads=True, source_commit='329b6160f513915f1c607dbfae3d5ce864a64a4f'))


def normalize_engine_length(answer, path):
    # The older engine experiment treats length stops as a gate failure.
    # This pilot retains them as complete attempts that fail extraction scoring.
    terminal = answer.get('terminal') or {}
    timing = terminal.get('timings', {})
    if (answer['status'] == 'TERMINATION_MISMATCH' and terminal.get('stop_type') == 'limit'
            and terminal.get('stop') is True and terminal.get('truncated') is False
            and terminal.get('tokens_predicted') == len(answer['token_ids']) == answer['max_new_tokens']
            and terminal.get('tokens_evaluated') == len(answer['prompt_token_ids'])
            and timing.get('cache_n') == 0 and timing.get('prompt_n') == len(answer['prompt_token_ids'])):
        answer['status'] = 'LENGTH_STOP'
        durable_json(path, answer)
    return answer


def engine(root, name, output, config):
    from transformers import AutoTokenizer
    state = read_json((root/'state.json').read_bytes())
    prepared = read_json((root/'prepared.json').read_bytes())
    kind = 'original' if name in ('original-panel', 'baseline') else 'trained'
    converted = read_json((phase_output(state, kind+'-conversion')/'complete.json').read_bytes())
    if converted['status'] != 'COMPLETE' or digest(converted['path']) != converted['sha256']:
        raise ValueError('Converted model identity differs')
    pin = read_json(Path(config['runtime_pin']).read_bytes())
    check_files(pin['files'])
    engine_config = read_json(Path(config['engine_configuration']).read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True)
    panel = name.endswith('panel')
    split = 'training' if panel else 'development'
    indices = prepared['panel'] if panel else list(range(config['development_examples']))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str((Path(config['engine_root'])/'binary/llama-server').resolve()), '-m', converted['path'],
               '--host', '127.0.0.1', '--port', str(port), *engine_config['server_arguments']]
    env = dict(os.environ, LD_LIBRARY_PATH=pin['library_path'])
    durable_json(output/'launch.json', dict(command=command, gguf_sha256=converted['sha256'],
                 runtime_pin_sha256=digest(config['runtime_pin']), library_path=pin['library_path']))
    process = None
    try:
        phase('loading')
        with (output/'server.log').open('xb') as log:
            begin = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
            durable_json(output/'owner.json', dict(pid=process.pid, process_group=os.getpgrp()))
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Engine startup failed')
                if time.monotonic()-begin >= engine_config['startup_seconds']:
                    raise TimeoutError('Engine startup deadline')
                try:
                    if json_request(port, '/health', timeout=1).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            loading = time.monotonic()-begin
            placement = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', (output/'server.log').read_text(errors='replace'))
            if not placement or placement[-1][0] != placement[-1][1]:
                raise ValueError('Full GPU layer placement was not established')
            phase('evaluation')
            timings, agreements = [], []
            for count, index in enumerate(indices, 1):
                if stopping():
                    return
                prompt = generation_input(prepared['items'][split][index])
                messages = prepared['rows'][split][index]['messages']
                prompt_messages = messages[:-1] if messages[-1]['role'] == 'assistant' else messages
                rendered = tokenizer.apply_chat_template(prompt_messages,
                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
                require_prompt_tokens(port, rendered, prompt)
                agreements.append(dict(index=index, prompt_token_ids=prompt))
                durable_json(output/'prompt-agreement.json', agreements)
                path = output/f'answer-{index}.json'
                answer = None
                for prior in state.get('partial_phases', {}).get(name, []):
                    saved = Path(prior['output'])/path.name
                    if str(saved) not in prior['files']:
                        continue
                    check_files(prior['files'])
                    candidate = read_json(saved.read_bytes())
                    if candidate.get('status') not in ('COMPLETE', 'LENGTH_STOP'):
                        continue
                    if candidate.get('prompt_token_ids') != prompt or candidate.get('max_new_tokens') != 1792:
                        raise ValueError('Saved partial answer has a different request identity')
                    for sibling in saved.parent.glob(saved.stem+'.*'):
                        if str(sibling) not in prior['files']:
                            raise ValueError('Saved partial capture is not bound to the run')
                        shutil.copyfile(sibling, output/sibling.name)
                    answer = candidate
                    break
                if answer is None:
                    answer = normalize_engine_length(generate(port, tokenizer, prompt, path, index=index,
                                  maximum=1792, deadline=60), path)
                if answer['status'] not in ('COMPLETE', 'LENGTH_STOP'):
                    raise ValueError('Engine request failed its deadline or capture contract')
                timings.append(dict(index=index, seconds=answer['timing']['request_seconds'],
                                    input_tokens=len(prompt), output_tokens=len(answer['token_ids'])))
                print(f'{name} {count}/{len(indices)} | {answer["status"]} | format {answer["format_valid"]} | '
                      f'{answer["timing"]["request_seconds"]:.2f}s', flush=True)
            durable_json(output/'complete.json', dict(status='COMPLETE', loading_seconds=loading, timings=timings))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            durable_json(output/'cleanup.json', dict(pid=process.pid, returncode=process.returncode))
