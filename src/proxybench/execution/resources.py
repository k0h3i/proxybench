"""Supervise one local GPU process group with host, device, and phase limits."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from proxybench.execution.runner import write_json


def host_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {'available_bytes': values['MemAvailable'], 'swap_used_bytes': values['SwapTotal'] - values['SwapFree'],
            'swap_in_pages': int(vm['pswpin']), 'swap_out_pages': int(vm['pswpout'])}


def device_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                             '--format=csv,noheader,nounits', '--id=0'], capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError('Device memory query failed: ' + result.stderr.strip())
    total, used, free = [int(s.strip()) * 1024 * 1024 for s in result.stdout.strip().split(',')]
    return {'total_bytes': total, 'used_bytes': used, 'free_bytes': free}


def process_memory(pid):
    try:
        values = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        return {key: int(values.get(key, '0 kB').split()[0]) * 1024 for key in ('VmRSS', 'VmHWM', 'VmSwap')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def limit_reason(samples_below, available, phase_elapsed, total_elapsed, limits):
    below = samples_below + 1 if available < limits['stop_host_bytes'] else 0
    if below >= limits.get('stop_host_samples', 2):
        return below, 'HOST_MEMORY_LIMIT'
    if phase_elapsed >= limits['phase_seconds']:
        return below, 'PHASE_TIMEOUT'
    if total_elapsed >= limits['total_seconds']:
        return below, 'AGGREGATE_TIMEOUT'
    return below, None


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # The group can retain descendants after its leader exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(command, output, limits, *, ledger):
    output, ledger = Path(output), Path(ledger)
    output.mkdir(parents=True, exist_ok=False)
    used = 0
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            entry = json.loads(line)
            if entry['status'] == 'STARTED':
                raise ValueError('Unclosed execution ledger entry requires operator reconciliation')
            used += entry['elapsed_seconds']
    initial = host_memory()
    device = device_memory()
    if (initial['available_bytes'] < limits['start_host_bytes']
            or device['free_bytes'] < limits['device_margin_bytes']
            or used >= limits['total_seconds']):
        write_json(output / 'result.json', {'status': 'PREFLIGHT_REFUSED', 'host': initial,
                                           'device': device, 'used_seconds': used})
        return 'PREFLIGHT_REFUSED'
    write_json(output / 'configuration.json', {'command': command, 'limits': limits, 'prior_seconds': used,
                                               'initial_host': initial, 'initial_device': device})
    ledger.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive sidecar prevents concurrent candidates from sharing one allocation.
    lock = ledger.with_suffix('.lock')
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    start = time.monotonic()
    process = None
    status = 'SUPERVISOR_FAILED'
    phase, phase_started, below = 'loading', start, 0
    minimum_free = device['free_bytes']
    try:
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr, (output / 'memory.jsonl').open('x') as log:
            env = dict(os.environ, PROXYBENCH_PHASE_FILE=str((output / 'phase.json').resolve()))
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True, env=env)
            while True:
                tick = time.monotonic()
                phase_file = output / 'phase.json'
                if phase_file.exists():
                    current = json.loads(phase_file.read_text())['phase']
                    if current != phase:
                        phase, phase_started = current, tick
                host, gpu = host_memory(), device_memory()
                minimum_free = min(minimum_free, gpu['free_bytes'])
                log.write(json.dumps({'elapsed_seconds': tick - start, 'phase': phase, 'host': host,
                                      'device': gpu, 'process': process_memory(process.pid)}) + '\n')
                log.flush()
                below, reason = limit_reason(below, host['available_bytes'], tick - phase_started,
                                              used + tick - start, limits)
                if gpu['free_bytes'] < limits['device_margin_bytes']:
                    reason = 'DEVICE_MEMORY_LIMIT'
                if reason:
                    status = reason
                    stop_group(process)
                    break
                code = process.poll()
                if code is not None:
                    status = 'EXITED' if code == 0 else 'PROCESS_FAILED'
                    break
                time.sleep(max(0, limits.get('sample_seconds', 1) - (time.monotonic() - tick)))
    finally:
        if process is not None:
            stop_group(process)
        elapsed = time.monotonic() - start
        entry = {'status': status, 'elapsed_seconds': elapsed, 'run': str(output)}
        with ledger.open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        lock.unlink()
        write_json(output / 'result.json', {**entry, 'total_used_seconds': used + elapsed,
                                           'returncode': process.returncode if process else None,
                                           'minimum_device_free_bytes': minimum_free,
                                           'memory_margin_pass': minimum_free >= limits['device_margin_bytes']})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ledger', type=Path, required=True)
    argv = sys.argv[1:]
    if '--' not in argv:
        parser.error('Separate the worker command with --')
    boundary = argv.index('--')
    args = parser.parse_args(argv[:boundary])
    command = argv[boundary + 1:]
    if not command:
        parser.error('A worker command is required')
    config = json.loads(args.configuration.read_text())
    result = supervise(command, args.output, config['limits'], ledger=args.ledger)
    print(result)
    return 0 if result == 'EXITED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
