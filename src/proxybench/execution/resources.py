"""Supervise one local GPU process group with host, device, and phase limits."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from proxybench.execution.runner import write_json


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ledger_entries(ledger):
    entries = [json.loads(line) for line in Path(ledger).read_text().splitlines()] if Path(ledger).exists() else []
    if any(entry['status'] == 'STARTED' for entry in entries):
        raise ValueError('Unclosed execution ledger entry requires operator reconciliation')
    ids = [entry['execution_id'] for entry in entries if 'execution_id' in entry]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate execution ledger entry')
    return entries


def append_entry(ledger, entry):
    with Path(ledger).open('a') as stream:
        stream.write(json.dumps(entry) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def group_members(pgid):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def reconcile(ledger):
    """Charge an abandoned execution once, after its recorded processes exit."""
    ledger = Path(ledger)
    active = ledger.with_suffix('.active.json')
    record = json.loads(active.read_text())
    entries = ledger_entries(ledger)
    same_boot = record['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if same_boot and Path(f"/proc/{record['supervisor_pid']}").exists():
        raise ValueError('Recorded supervisor still exists; reconcile its ownership first')
    owner = Path(record['run']) / 'owner.json'
    if same_boot and owner.exists() and group_members(json.loads(owner.read_text())['pid']):
        raise ValueError('Owned process group survives; stop the recorded group before reconciliation')
    if not any(entry.get('execution_id') == record['execution_id'] for entry in entries):
        elapsed = time.monotonic() - record['started_monotonic'] if same_boot else time.time() - record['started_wall']
        if elapsed < 0:
            raise ValueError('Clock mismatch requires manual elapsed-time reconciliation')
        append_entry(ledger, dict(status='RECONCILED', execution_id=record['execution_id'],
                                 elapsed_seconds=elapsed, run=record['run']))
    active.unlink()
    ledger.with_suffix('.lock').unlink(missing_ok=True)
    return sum(entry['elapsed_seconds'] for entry in ledger_entries(ledger))


def host_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {'available_bytes': values['MemAvailable'], 'swap_used_bytes': values['SwapTotal'] - values['SwapFree'],
            'swap_in_pages': int(vm['pswpin']), 'swap_out_pages': int(vm['pswpout'])}


def device_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm',
                             '--format=csv,noheader,nounits', '--id=0'], capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError('Device memory query failed: ' + result.stderr.strip())
    fields = [s.strip() for s in result.stdout.strip().split(',')]
    total, used, free = [int(s) * 1024 * 1024 for s in fields[:3]]
    activity = {}
    for name, value in zip(('gpu_utilization_percent', 'memory_utilization_percent', 'temperature_c',
                            'power_watts', 'sm_clock_mhz'), fields[3:], strict=True):
        try:
            activity[name] = float(value)
        except ValueError:
            activity[name] = None
    return {'total_bytes': total, 'used_bytes': used, 'free_bytes': free, **activity}


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
    active = ledger.with_suffix('.active.json')
    if active.exists() or ledger.with_suffix('.lock').exists():
        raise ValueError('Active execution requires process and elapsed-time reconciliation')
    output.mkdir(parents=True, exist_ok=False)
    used = sum(entry['elapsed_seconds'] for entry in ledger_entries(ledger))
    cpu = limits.get('cpu_only', False)
    initial = host_memory()
    device = device_memory() if not cpu else {'free_bytes': 0}
    if (initial['available_bytes'] < limits['start_host_bytes']
            or (not cpu and device['free_bytes'] < limits['device_margin_bytes'])
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
    execution_id = uuid.uuid4().hex
    record = dict(status='STARTED', execution_id=execution_id, run=str(output.resolve()),
                  supervisor_pid=os.getpid(), started_monotonic=start, started_wall=time.time(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    durable_json(active, record)
    process = None
    status = 'SUPERVISOR_FAILED'
    phase, phase_started, below = 'loading', start, 0
    minimum_free = device['free_bytes']
    try:
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr, (output / 'memory.jsonl').open('x') as log:
            env = dict(os.environ, PROXYBENCH_PHASE_FILE=str((output / 'phase.json').resolve()),
                       PROXYBENCH_REQUEST_FILE=str((output / 'request.json').resolve()))
            owner, gate = output / 'owner.json', output / 'launch.json'
            wrapper = [sys.executable, '-m', 'proxybench.execution.owned', str(owner), str(gate), *command]
            process = subprocess.Popen(wrapper, stdout=stdout, stderr=stderr, start_new_session=True, env=env)
            durable_json(active, {**record, 'worker_pid': process.pid})
            durable_json(gate, {'execution_id': execution_id})
            while True:
                tick = time.monotonic()
                phase_file = output / 'phase.json'
                if phase_file.exists():
                    current = json.loads(phase_file.read_text())['phase']
                    if current != phase:
                        phase, phase_started = current, tick
                host, gpu = host_memory(), device_memory() if not cpu else {'free_bytes': 0}
                minimum_free = min(minimum_free, gpu['free_bytes'])
                log.write(json.dumps({'elapsed_seconds': tick - start, 'phase': phase, 'host': host,
                                      'device': gpu, 'process': process_memory(process.pid)}) + '\n')
                log.flush()
                below, reason = limit_reason(below, host['available_bytes'], tick - (start if limits.get('fixed_phase') else phase_started),
                                              used + tick - start, limits)
                if not cpu and gpu['free_bytes'] < limits['device_margin_bytes']:
                    reason = 'DEVICE_MEMORY_LIMIT'
                request_file = output / 'request.json'
                if request_file.exists():
                    try:
                        request = json.loads(request_file.read_text())
                        if tick >= request['deadline_monotonic']:
                            reason = 'REQUEST_TIMEOUT'
                    except FileNotFoundError:
                        pass
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
        cleanup_start = time.monotonic()
        if process is not None:
            stop_group(process)
        survivors = group_members(process.pid) if process else []
        if survivors:
            status = 'OWNERSHIP_UNRESOLVED'
        pending = output / 'request.json'
        if pending.exists():
            request_path = Path(json.loads(pending.read_text())['request_path'])
            request = json.loads(request_path.read_text())
            if request.get('status') == 'STARTED':
                events = []
                token_path = request_path.with_suffix('.tokens.jsonl')
                if token_path.exists():
                    for line in token_path.read_text().splitlines():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            break
                durable_json(request_path, {**request, 'status': 'INTERRUPTED', 'supervisor_status': status,
                             'token_ids': [event['token_id'] for event in events],
                             'format_valid': False, 'terminal_stream_event': False})
        elapsed = time.monotonic() - start
        entry = {'status': status, 'execution_id': execution_id, 'elapsed_seconds': elapsed, 'run': str(output)}
        append_entry(ledger, entry)
        if not survivors:
            active.unlink()
            lock.unlink()
        write_json(output / 'result.json', {**entry, 'total_used_seconds': used + elapsed,
                                           'returncode': process.returncode if process else None,
                                           'cleanup_seconds': time.monotonic() - cleanup_start,
                                           'surviving_owned_pids': survivors,
                                           'minimum_device_free_bytes': minimum_free,
                                           'memory_margin_pass': cpu or minimum_free >= limits['device_margin_bytes']})
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
