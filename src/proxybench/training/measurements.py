"""Measure training work without importing a device runtime or changing its state."""

from collections import deque
from bisect import bisect_left
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import time
import uuid


EVENT_PREFIX = 'PROXYBENCH_EVENT '
EVENT_SCHEMA = 'training-event-v1'
RUNTIME_SCHEMA = 'training-runtime-v1'
STAGES = frozenset({
    'preparation', 'model_loading', 'initial_base_hashing', 'adapter_attachment',
    'restoration', 'full_loop', 'compute', 'journal', 'monitor', 'report',
    'checkpoint', 'checkpoint_serialization', 'checkpoint_readback',
    'checkpoint_comparison', 'checkpoint_hashing', 'checkpoint_synchronization',
    'checkpoint_publication', 'final_validation', 'final_publication',
    'measurement_overhead',
})
PACKAGES = ('torch', 'transformers', 'unsloth', 'unsloth-zoo', 'peft', 'accelerate',
            'triton', 'cut-cross-entropy', 'causal-conv1d', 'numpy', 'safetensors')


def _seconds(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('Measurement seconds must be finite and nonnegative')
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('Measurement counters must be nonnegative integers')
    return value


def event_line(event):
    """Encode one event for the supervisor. Events do not authorize recovery."""
    if event.get('schema_version') != EVENT_SCHEMA:
        raise ValueError('Unsupported training event schema')
    return EVENT_PREFIX + json.dumps(event, sort_keys=True, allow_nan=False)


def parse_event(line):
    """Return None for ordinary output or malformed optional telemetry."""
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        value = json.loads(line[len(EVENT_PREFIX):])
        if (not isinstance(value, dict) or value.get('schema_version') != EVENT_SCHEMA
                or value.get('kind') not in {'phase', 'update', 'measurement', 'memory', 'attempt'}):
            return None
        _seconds(value['attempt_elapsed_seconds'])
        return value
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


def epoch_progress(sample_position, epoch_boundaries):
    """Use actual sample boundaries, including unequal or partial epochs."""
    _count(sample_position)
    boundaries = list(epoch_boundaries)
    for boundary in boundaries:
        _count(boundary)
    if (not boundaries or boundaries[0] == 0
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
            or sample_position > boundaries[-1]):
        raise ValueError('Invalid epoch boundaries or sample position')
    index = bisect_left(boundaries, sample_position)
    previous = boundaries[index - 1] if index else 0
    size = boundaries[index] - previous
    completed = sample_position - previous
    return dict(epoch=index + 1, total_epochs=len(boundaries), examples_completed=completed,
                examples_per_epoch=size, fractional_epoch=index + completed / size)


class MeasurementRecorder:
    """Record inclusive wall intervals. Nested stage totals must not be added."""

    def __init__(self, *, clock=time.monotonic, emit=lambda event: None, attempt_id=None, window=12):
        if _count(window) == 0:
            raise ValueError('A measurement window must contain at least one update')
        self.clock, self.emit = clock, emit
        self.attempt_id = attempt_id or uuid.uuid4().hex
        self.started = clock()
        self.ended = None
        self.stages = {}
        self.recent = deque(maxlen=window)
        self.updates = []
        self.memory_window = None
        self.status = 'RUNNING'

    def elapsed(self):
        return _seconds((self.clock() if self.ended is None else self.ended) - self.started)

    def event(self, kind, **fields):
        event = dict(fields, schema_version=EVENT_SCHEMA, kind=kind,
                     attempt_id=self.attempt_id, attempt_elapsed_seconds=self.elapsed())
        self.emit(event)
        return event

    def phase(self, name, **fields):
        return self.event('phase', phase=name, **fields)

    @contextmanager
    def stage(self, name):
        if name not in STAGES:
            raise ValueError('Unsupported measurement stage')
        start = self.clock()
        outcome = 'FAILED'
        try:
            yield
            outcome = 'COMPLETE'
        finally:
            seconds = _seconds(self.clock() - start)
            row = self.stages.setdefault(name, dict(seconds=0.0, calls=0, failed_calls=0))
            row['seconds'] += seconds
            row['calls'] += 1
            row['failed_calls'] += outcome == 'FAILED'
            self.event('measurement', stage=name, seconds=seconds, outcome=outcome,
                       timing_method='host_wall', stage_accounting='inclusive_nested')

    @staticmethod
    def _window(rows, label):
        seconds = sum(row['full_loop_seconds'] for row in rows)
        nonpadding = sum(row['nonpadding_tokens'] for row in rows)
        supervised = sum(row['supervised_tokens'] for row in rows)
        return dict(label=label, timing='completed_update_work_excluding_event_emission', updates=len(rows), seconds=seconds,
                    first_step=rows[0]['global_step'] if rows else None,
                    last_step=rows[-1]['global_step'] if rows else None,
                    nonpadding_tokens=nonpadding, supervised_tokens=supervised,
                    nonpadding_tokens_per_second=nonpadding / seconds if seconds else None,
                    supervised_tokens_per_second=supervised / seconds if seconds else None)

    def record_update(self, *, global_step, sample_position, nonpadding_tokens, supervised_tokens,
                      full_loop_seconds, compute_seconds=None, compute_timing='host_wall', **fields):
        """Call after all update work, including reporting and any periodic save."""
        for count in (global_step, sample_position, nonpadding_tokens, supervised_tokens):
            _count(count)
        if supervised_tokens > nonpadding_tokens:
            raise ValueError('Supervised tokens exceed non-padding tokens')
        _seconds(full_loop_seconds)
        if compute_seconds is not None:
            _seconds(compute_seconds)
        if compute_timing not in {'host_wall', 'device_events', 'synchronized_wall'}:
            raise ValueError('Unsupported computation timing method')
        if self.updates and (global_step <= self.updates[-1]['global_step']
                             or sample_position <= self.updates[-1]['sample_position']):
            raise ValueError('Completed update and sample counters must increase')
        row = dict(global_step=global_step, sample_position=sample_position,
                   nonpadding_tokens=nonpadding_tokens, supervised_tokens=supervised_tokens,
                   full_loop_seconds=full_loop_seconds, compute_seconds=compute_seconds,
                   compute_timing=compute_timing,
                   update_period='first_update' if not self.updates else 'steady_state')
        self.updates.append(row)
        self.recent.append(row)
        return self.event('update', **fields, **row,
                          throughput=self._window(self.recent, 'recent_completed_updates'),
                          final_publication_in_eta=False)

    def remaining_loop_seconds(self, planned_updates):
        _count(planned_updates)
        if not self.updates:
            return None
        rows = self.updates[1:] or self.updates
        seconds = sum(row['full_loop_seconds'] for row in rows)
        if seconds == 0:
            return None
        return max(0, planned_updates - self.updates[-1]['global_step']) * seconds / len(rows)

    def begin_memory_window(self, name, *, reset, global_step):
        """Call the supplied peak reset and record its actual measurement boundary."""
        _count(global_step)
        if not isinstance(name, str) or not name:
            raise ValueError('A memory window needs a name')
        reset()
        self.memory_window = dict(label=name, peak_reset_step=global_step,
                                  peak_reset_elapsed_seconds=self.elapsed())

    def memory_sample(self, *, allocated_bytes, reserved_bytes, peak_allocated_bytes,
                      peak_reserved_bytes, free_device_bytes):
        if self.memory_window is None:
            raise ValueError('Reset and label peak counters before reporting memory')
        values = dict(allocated_bytes=allocated_bytes, reserved_bytes=reserved_bytes,
                      peak_allocated_bytes=peak_allocated_bytes, peak_reserved_bytes=peak_reserved_bytes,
                      free_device_bytes=free_device_bytes)
        for value in values.values():
            _count(value)
        if peak_allocated_bytes < allocated_bytes or peak_reserved_bytes < reserved_bytes:
            raise ValueError('A memory peak is below its current value')
        return self.event('memory', **values, peak_window=dict(self.memory_window))

    def snapshot(self):
        full_loop = self._window(self.updates, 'all_completed_updates_this_attempt')
        if 'full_loop' in self.stages:
            seconds = self.stages['full_loop']['seconds']
            full_loop.update(seconds=seconds, timing='inclusive_loop_host_wall',
                             nonpadding_tokens_per_second=full_loop['nonpadding_tokens'] / seconds if seconds else None,
                             supervised_tokens_per_second=full_loop['supervised_tokens'] / seconds if seconds else None)
        return dict(attempt_id=self.attempt_id, status=self.status,
                    attempt_elapsed_seconds=self.elapsed(),
                    stage_accounting='inclusive_nested',
                    stages={name: dict(row) for name, row in self.stages.items()},
                    full_loop=full_loop,
                    first_update=self._window(self.updates[:1], 'first_update_this_attempt'),
                    steady_state=self._window(self.updates[1:], 'later_updates_this_attempt'))

    def finish(self, status, **fields):
        if status not in {'COMPLETE', 'CLEAN_STOP', 'FAILED'}:
            raise ValueError('Unsupported attempt outcome')
        if self.ended is not None:
            raise ValueError('The measurement attempt already ended')
        self.ended = self.clock()
        self.status = status
        return self.event('attempt', **self.snapshot(), **fields)


def capture_runtime_identity(*, model, tokenizer, optimizer, hardware, kernels,
                             code_root=None, package_names=PACKAGES, package_version=metadata.version):
    """Bind observed configuration. The caller supplies device facts after loading."""
    root = Path(code_root) if code_root is not None else Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob('*.py')) if '__pycache__' not in path.parts}
    packages = {}
    for name in package_names:
        try:
            packages[name] = package_version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    value = dict(schema_version=RUNTIME_SCHEMA, code=files, packages=packages,
                 python=platform.python_version(), implementation=platform.python_implementation(),
                 platform=dict(system=platform.system(), release=platform.release(), machine=platform.machine()),
                 model=model, tokenizer=tokenizer, optimizer=optimizer, hardware=hardware, kernels=kernels)
    # Copy input mappings and reject values that cannot survive JSON storage.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def compare_runtime_identity(previous, current):
    """A matching identity alone does not establish exact device continuation."""
    required = {'code', 'packages', 'python', 'implementation', 'platform', 'model', 'tokenizer',
                'optimizer', 'hardware', 'kernels'}

    def incomplete(value, path=()):
        if value is None:
            return not (path and path[0] == 'optimizer' and path[-1] in {'fused', 'foreach'})
        if value == '' or value == 'unknown':
            return True
        if isinstance(value, dict):
            return not value or any(incomplete(item, (*path, key)) for key, item in value.items())
        if isinstance(value, list):
            return any(incomplete(item, path) for item in value)
        return False

    def valid(value):
        return (isinstance(value, dict) and value.get('schema_version') == RUNTIME_SCHEMA
                and required <= value.keys() and all(value[key] is not None for key in required)
                and isinstance(value['packages'], dict)
                and all(isinstance(version, str) and version for version in value['packages'].values())
                and not incomplete(value))

    if not valid(previous) or not valid(current):
        status = 'UNVERIFIED'
        changed = []
    else:
        changed = sorted(key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key))
        status = 'CHANGED' if changed else 'MATCH'
    return dict(status=status, changed=changed, exact_continuation_claim=False,
                requires_acceptance=status != 'MATCH')
