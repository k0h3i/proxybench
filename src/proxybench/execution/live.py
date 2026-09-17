"""Supervise fixed phases with durable logs and bounded cooperative stops."""

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

from proxybench.execution.resources import (
    append_entry, device_memory, durable_json, group_members, host_memory,
    ledger_entries, process_memory,
)


class Console:
    """A slow terminal can lose display chunks, but never saved worker output."""

    def __init__(self, stream):
        self.pending = queue.Queue(maxsize=128)
        self.stream = stream
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        while True:
            chunk = self.pending.get()
            try:
                if chunk is None:
                    return
                self.stream.write(chunk)
                self.stream.flush()
            except (OSError, ValueError):
                return
            finally:
                self.pending.task_done()

    def put(self, text):
        try:
            self.pending.put_nowait(text)
        except queue.Full:
            pass

    def close(self):
        self.put(None)
        self.thread.join(timeout=.2)


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


class MemorySampler:
    """A slow device query cannot postpone a hard deadline or an interrupt."""

    def __init__(self, pid, cpu, interval):
        self.pending = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        def sample():
            while not self.stop.is_set():
                try:
                    value = dict(at=time.monotonic(), host=host_memory(),
                                 device={} if cpu else device_memory(), process=process_memory(pid))
                except Exception as exc:
                    value = dict(error=str(exc), at=time.monotonic())
                try:
                    self.pending.put_nowait(value)
                except queue.Full:
                    pass
                self.stop.wait(interval)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()


def supervise(command, output, limits, *, ledger, phase, phase_used=0,
              console=None, environment=None):
    """Keep phase and aggregate clocks through retries, labels, and stops.

    Workers receive a stop-file path. Only a worker's verified safe checkpoint
    can establish resume eligibility. This supervisor never infers that state.
    """
    output, ledger = Path(output), Path(ledger)
    active, lock = ledger.with_suffix('.active.json'), ledger.with_suffix('.lock')
    if active.exists() or lock.exists():
        raise ValueError('An execution ledger remains active')
    entries = ledger_entries(ledger)
    used = sum(e['elapsed_seconds'] for e in entries)
    if used >= limits['total_seconds'] or phase_used >= limits['phase_seconds']:
        raise ValueError('Execution budget is exhausted')
    cpu = limits.get('cpu_only', False)
    host = host_memory()
    device = {} if cpu else device_memory()
    if host['available_bytes'] < limits['start_host_bytes']:
        raise ValueError('Insufficient available host memory')
    if not cpu and device['free_bytes'] < limits['device_margin_bytes']:
        raise ValueError('Insufficient free device memory')
    output.mkdir(parents=True, exist_ok=False)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    display = Console(console or sys.stdout)
    start = time.monotonic()
    record = dict(execution_id=uuid.uuid4().hex, run=str(output.resolve()), phase=phase,
                  supervisor_pid=os.getpid(), started_monotonic=start, started_wall=time.time(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    durable_json(active, dict(record, status='STARTED'))
    interruptions, acknowledged = 0, 0
    requested_at = None

    def interrupt(signum, frame):
        nonlocal interruptions, requested_at
        interruptions += 1
        if requested_at is None:
            requested_at = time.monotonic()
        # Signal handlers must not acquire the terminal queue's nonreentrant lock.

    previous = signal.signal(signal.SIGINT, interrupt)
    previous_term = signal.signal(signal.SIGTERM, interrupt)
    process, status, stopped_at = None, 'SUPERVISOR_FAILED', None
    last_sample, last_beat, label = start, -float('inf'), 'loading'
    minimum_free = device.get('free_bytes')
    sampler = None
    try:
        env = dict(os.environ, **(environment or {}),
                   PROXYBENCH_PHASE_FILE=str((output / 'phase.json').resolve()),
                   PROXYBENCH_REQUEST_FILE=str((output / 'request.json').resolve()),
                   PROXYBENCH_STOP_FILE=str((output / 'stop.json').resolve()))
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        durable_json(output / 'configuration.json', dict(command=command, limits=limits,
                     phase_used=phase_used, prior_seconds=used, environment=environment))
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            wrapper = [sys.executable, '-m', 'proxybench.execution.owned',
                       str(output / 'owner.json'), str(output / 'launch.json'), *command]
            process = subprocess.Popen(wrapper, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            durable_json(active, dict(record, status='STARTED', worker_pid=process.pid))
            durable_json(output / 'launch.json', dict(execution_id=record['execution_id']))
            sampler = MemorySampler(process.pid, cpu, limits.get('sample_seconds', 1))
            with (output / 'stdout.log').open('rb') as out, (output / 'stderr.log').open('rb') as err, \
                    (output / 'memory.jsonl').open('x') as memory:
                while True:
                    tick = time.monotonic()
                    if interruptions > acknowledged:
                        display.put('Stop requested. Saving at a safe boundary, at most 30 seconds.\n'
                                    if interruptions == 1 else 'Second interrupt. Stopping the owned processes now.\n')
                        acknowledged = interruptions
                    for reader in (out, err):
                        chunk = reader.read(65536)
                        if chunk:
                            display.put(chunk.decode('utf-8', errors='replace'))
                    phase_file = output / 'phase.json'
                    if phase_file.exists():
                        label = json.loads(phase_file.read_text())['phase']
                    reason = None
                    if tick - start + used >= limits['total_seconds']:
                        reason = 'AGGREGATE_TIMEOUT'
                    if tick - start + phase_used >= limits['phase_seconds']:
                        reason = 'PHASE_TIMEOUT'
                    pending = output / 'request.json'
                    try:
                        if pending.exists() and tick >= json.loads(pending.read_text())['deadline_monotonic']:
                            reason = 'REQUEST_TIMEOUT'
                    except FileNotFoundError:
                        pass
                    try:
                        sample = sampler.pending.get_nowait()
                    except queue.Empty:
                        sample = None
                    if sample and 'error' in sample:
                        reason = 'MEMORY_MONITOR_FAILED'
                    elif sample:
                        last_sample = sample['at']
                        host, device = sample['host'], sample['device']
                        if host['available_bytes'] < limits['stop_host_bytes']:
                            reason = 'HOST_MEMORY_LIMIT'
                        if not cpu:
                            minimum_free = min(minimum_free, device['free_bytes'])
                            if device['free_bytes'] < limits['device_margin_bytes']:
                                reason = 'DEVICE_MEMORY_LIMIT'
                        memory.write(json.dumps(dict(elapsed_seconds=tick-start, phase=label, host=host,
                                                     device=device, process=sample['process'])) + '\n')
                        memory.flush()
                    if tick-last_sample > 6:
                        reason = 'MEMORY_MONITOR_STALE'
                    if interruptions and stopped_at is None:
                        stopped_at = requested_at
                        durable_json(output / 'stop.json', dict(requested_monotonic=requested_at))
                        if label not in {'training', 'compilation', 'saving'}:
                            reason = 'USER_STOP'
                    if interruptions > 1:
                        reason = 'FORCED_STOP'
                    if stopped_at is not None and tick - stopped_at >= limits.get('grace_seconds', 30):
                        reason = 'STOP_GRACE_EXHAUSTED'
                    if reason:
                        status = reason
                        kill_group(process)
                        break
                    code = process.poll()
                    if code is not None:
                        status = 'EXITED' if code == 0 else 'PROCESS_FAILED'
                        # Drain the saved logs without waiting for terminal consumption.
                        # A surviving child can still write. Bound this last display pass.
                        for reader in (out, err):
                            for _ in range(16):
                                chunk = reader.read(65536)
                                if not chunk:
                                    break
                                display.put(chunk.decode('utf-8', errors='replace'))
                        break
                    if tick - last_beat >= limits.get('heartbeat_seconds', 15):
                        last_beat = tick
                        display.put(f'[{phase}: {label}] elapsed {tick-start:.0f}s | '
                                    f'{"CPU" if cpu else "GPU"} total {used+tick-start:.0f}/{limits["total_seconds"]}s\n')
                    time.sleep(.05)
    finally:
        if process is not None:
            kill_group(process)
        if sampler is not None:
            sampler.stop.set()
        survivors = group_members(process.pid) if process else []
        if survivors:
            status = 'OWNERSHIP_UNRESOLVED'
        elapsed = time.monotonic() - start
        entry = dict(record, status=status, elapsed_seconds=elapsed)
        append_entry(ledger, entry)
        durable_json(output / 'result.json', dict(entry, returncode=process.returncode if process else None,
                     surviving_owned_pids=survivors, total_used_seconds=used+elapsed,
                     phase_used_seconds=phase_used+elapsed, minimum_device_free_bytes=minimum_free,
                     stop_requested=bool(interruptions)))
        if not survivors:
            active.unlink()
            lock.unlink()
        signal.signal(signal.SIGINT, previous)
        signal.signal(signal.SIGTERM, previous_term)
        display.close()
    return status
