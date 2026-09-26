"""Render optional training measurements without controlling worker execution."""

import math
from pathlib import Path
import shutil
import time

from proxybench.training.measurements import epoch_progress, parse_event


PHASES = frozenset({'preparation', 'loading', 'training', 'saving checkpoint',
                    'validating final adapter', 'publishing final adapter'})


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _value(value, format_spec='.4g'):
    return format(value, format_spec) if _number(value) else 'unavailable'


def _percent(fraction):
    return f'{fraction * 100:.1f}'.rstrip('0').rstrip('.') + '%'


def _duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m' if hours else f'{minutes}m{seconds:02d}s'


class TrainingDisplay:
    """Keep display frequency separate from measurements and resource checks."""

    def __init__(self, write, *, tty=False, clock=time.monotonic, started=None,
                 prior_seconds=0, phase_used=0, total_seconds=None, phase_seconds=None,
                 capture_path=None):
        self.write, self.tty, self.clock = write, tty, clock
        self.started = clock() if started is None else started
        self.prior_seconds, self.phase_used = prior_seconds, phase_used
        self.total_seconds, self.phase_seconds = total_seconds, phase_seconds
        self.phase = 'preparation'
        self.data = {}
        self.memory = {}
        self.counters = None
        self.last_refresh = -float('inf')
        self.last_summary = None
        self.worker_status = None
        self.result_published = False
        self.finished = False
        self._stdout = b''
        self._frame_rows = 0
        self.capture_path = Path(capture_path) if capture_path is not None else None
        self._stderr = b''
        self._batch_shown = None
        self._last_phase_state = None
        self._warnings = set()

    def _restore_counters(self, event):
        keys = ('global_step', 'sample_position', 'planned_updates', 'epoch_boundaries')
        values = {key: event.get(key, self.data.get(key)) for key in keys}
        step, planned = values['global_step'], values['planned_updates']
        if type(step) is not int or type(planned) is not int or not 0 <= step <= planned or planned == 0:
            return False
        try:
            counters = epoch_progress(values['sample_position'], values['epoch_boundaries'])
        except (ValueError, TypeError):
            return False
        if self.counters and (step < self.data['global_step']
                              or values['sample_position'] < self.data['sample_position']):
            return False
        self.data.update(values)
        self.counters = counters
        return True

    def accept(self, event):
        """Consume a decoded event. Missing measurements remain unavailable."""
        if not isinstance(event, dict):
            return
        kind = event.get('kind')
        if not isinstance(kind, str):
            return
        if kind == 'phase' and (not isinstance(event.get('phase'), str) or event['phase'] not in PHASES):
            return
        restored = self._restore_counters(event) if kind in {'phase', 'update'} else False
        for key in ('batch_size', 'gradient_accumulation_steps', 'effective_batch_size',
                    'last_checkpoint', 'resume_eligible', 'loss', 'weighted_loss_12',
                    'learning_rate', 'gradient_norm', 'remaining_loop_seconds', 'throughput'):
            if key in event and (kind != 'update' or restored):
                self.data[key] = event[key]
        if kind == 'memory':
            self.memory.update(event)
        if kind == 'phase' and event.get('phase') in PHASES:
            self.phase = event['phase']
            state = (self.phase, self.data.get('global_step'), self.counters is not None)
            if state != self._last_phase_state:
                self._last_phase_state = state
                self.render(immediate=True)
        elif kind == 'update' and restored:
            step = self.data['global_step']
            boundary = self.counters['examples_completed'] == self.counters['examples_per_epoch']
            if not self.tty and (step % 10 == 0 or boundary) and step != self.last_summary:
                self.render(immediate=True)
            elif self.tty:
                self.tick()
        elif kind == 'attempt':
            self.worker_status = event.get('status')
            self.result_published = self.worker_status == 'COMPLETE' and event.get('result_published') is True
            if self.worker_status == 'FAILED':
                self.message('Training attempt failed. Waiting for worker exit.')
            elif self.worker_status == 'CLEAN_STOP':
                self.message('Training stopped. Waiting for worker exit.')
            elif self.result_published:
                self.message('Final result published. Waiting for worker exit.')

    def time_fields(self):
        elapsed = max(0, self.clock() - self.started)
        remaining = [ceiling - prior - elapsed for ceiling, prior in (
            (self.total_seconds, self.prior_seconds), (self.phase_seconds, self.phase_used))
            if ceiling is not None]
        budget = f'{max(0, min(remaining)):.0f}s' if remaining else 'unbounded'
        eta = self.data.get('remaining_loop_seconds')
        estimate = f'~{eta:.0f}s' if _number(eta) and eta >= 0 else 'estimating'
        return (f'attempt {elapsed:.0f}s | charged {self.prior_seconds + elapsed:.0f}s | '
                f'budget left {budget} | loop left {estimate} (final publication excluded)')

    def details(self):
        """Return all available display fields with their measurement meanings."""
        batch = ('Batch: ' + _value(self.data.get('batch_size'))
                 + ' | Accumulation: ' + _value(self.data.get('gradient_accumulation_steps'))
                 + ' | Effective batch: ' + _value(self.data.get('effective_batch_size')))
        loss = (f'loss {_value(self.data.get("loss"))} | '
                f'loss last 12 updates, response-token-weighted {_value(self.data.get("weighted_loss_12"))} | '
                f'lr {_value(self.data.get("learning_rate"))} | '
                f'gradient norm before clipping {_value(self.data.get("gradient_norm"))}')
        throughput = self.data.get('throughput')
        throughput = throughput if isinstance(throughput, dict) else {}
        timing = throughput.get('timing')
        window = ('update work window' if timing == 'completed_update_work_excluding_event_emission'
                  else 'full-loop window' if timing == 'full_loop_host_wall' else 'measurement window')
        excluded = ' (event publication excluded)' if timing == 'completed_update_work_excluding_event_emission' else ''
        rates = (f'non-padding tokens/s {_value(throughput.get("nonpadding_tokens_per_second"))} | '
                 f'response tokens/s {_value(throughput.get("supervised_tokens_per_second"))} | '
                 f'{window} {_value(throughput.get("updates"))} updates, '
                 f'{_value(throughput.get("seconds"))}s{excluded}')
        memory = ' | '.join(f'{label} '
                            + (_value(self.memory[key] / 1024**3, '.2f')
                               if _number(self.memory.get(key)) else 'unavailable') + ' GiB'
                            for label, key in (('allocated', 'allocated_bytes'),
                                               ('peak allocated', 'peak_allocated_bytes'),
                                               ('peak reserved', 'peak_reserved_bytes'),
                                               ('device free', 'free_device_bytes')))
        checkpoint = self.data.get('last_checkpoint')
        saved = (f'{checkpoint.get("path")} at update {checkpoint.get("global_step")}'
                 if isinstance(checkpoint, dict) else 'none reported')
        eligible = self.data.get('resume_eligible')
        resume = 'yes (validated clean stop)' if eligible is True else 'no' if eligible is False else 'unverified'
        return f'{batch}\n{loss}\n{self.time_fields()}\n{rates}\n{memory}\nLast save: {saved} | Resume eligible: {resume}'

    def progress(self):
        if self.counters is None:
            return f'Phase: {self.phase} | counters pending acceptance'
        step, planned = self.data['global_step'], self.data['planned_updates']
        fraction = step / planned
        counter = self.counters
        examples, size = counter['examples_completed'], counter['examples_per_epoch']
        bar = '[' + '=' * int(fraction * 16) + '.' * (16 - int(fraction * 16)) + '] '
        return (f'Phase: {self.phase} | {bar if self.tty else ""}'
                f'Optimizer updates: {step}/{planned} ({_percent(fraction)}) | '
                f'Epoch {counter["epoch"]}/{counter["total_epochs"]}: {_percent(examples / size)} complete '
                f'({examples}/{size} examples) | fractional epoch {counter["fractional_epoch"]:.2f}')

    def render(self, *, immediate=False):
        if self.finished:
            return
        tick = self.clock()
        if not immediate and tick - self.last_refresh < 1:
            return
        self.last_refresh = tick
        self.last_summary = self.data.get('global_step')
        self._last_phase_state = (self.phase, self.data.get('global_step'), self.counters is not None)
        batch = tuple(self.data.get(key) for key in
                      ('batch_size', 'gradient_accumulation_steps', 'effective_batch_size'))
        if all(type(value) is int and value > 0 for value in batch) and batch != self._batch_shown:
            self.message(f'Batch: {batch[0]} | Accumulation: {batch[1]} | Effective batch: {batch[2]}')
            self._batch_shown = batch
        rows = self.compact_rows()
        if self.tty:
            # One physical row per line. Long optional details cannot wrap and
            # leave stale frames behind, including on immediate phase changes.
            width = max(1, shutil.get_terminal_size((120, 24)).columns - 1)
            rows = [row if len(row) <= width else row[:max(0, width - 1)] + '~' for row in rows]
            self.write(self._clear_frame() + '\n'.join(rows))
            self._frame_rows = len(rows)
        else:
            self.write(' | '.join(rows) + '\n')

    def compact_rows(self):
        """Show useful values now; preserve the full contract in saved events."""
        progress = self.phase.capitalize()
        if self.counters is not None:
            step, planned = self.data['global_step'], self.data['planned_updates']
            fraction = step / planned
            counter = self.counters
            filled = int(fraction * 12)
            bar = '[' + '=' * filled + '.' * (12 - filled) + '] ' if self.tty else ''
            progress += f' {bar}{step}/{planned} updates ({_percent(fraction)})'
            progress += (f' | Epoch {counter["epoch"]}/{counter["total_epochs"]}: '
                         f'{_percent(counter["examples_completed"] / counter["examples_per_epoch"])} '
                         f'({counter["examples_completed"]}/{counter["examples_per_epoch"]})')
        rows = [progress]
        metrics = [f'{label} {_value(self.data[key])}' for key, label in
                   (('loss', 'loss'), ('weighted_loss_12', 'mean loss (12, response-weighted)'),
                    ('learning_rate', 'lr'), ('gradient_norm', 'grad (pre-clip)'))
                   if _number(self.data.get(key))]
        if metrics:
            rows.append(' | '.join(metrics))
        throughput = self.data.get('throughput')
        throughput = throughput if isinstance(throughput, dict) else {}
        rates = [f'{label} {_value(throughput[key])}' for key, label in
                 (('nonpadding_tokens_per_second', 'tokens/s'), ('supervised_tokens_per_second', 'response/s'))
                 if _number(throughput.get(key))]
        memory = [f'{label} {self.memory[key] / 1024**3:.2f} GiB' for key, label in
                  (('allocated_bytes', 'GPU'), ('peak_allocated_bytes', 'peak'),
                   ('peak_reserved_bytes', 'reserved peak'), ('free_device_bytes', 'free'))
                  if _number(self.memory.get(key))]
        if rates or memory:
            rows.append(' | '.join(rates + memory))
        elapsed = max(0, self.clock() - self.started)
        times = [f'elapsed {_duration(elapsed)}']
        remaining = [ceiling - prior - elapsed for ceiling, prior in (
            (self.total_seconds, self.prior_seconds), (self.phase_seconds, self.phase_used))
            if ceiling is not None]
        if remaining:
            times.append(f'budget {_duration(min(remaining))}')
        checkpoint = self.data.get('last_checkpoint')
        if isinstance(checkpoint, dict) and type(checkpoint.get('global_step')) is int:
            times.append(f'saved {checkpoint["global_step"]}')
        if self.data.get('resume_eligible') is True:
            times.append('resume eligible')
        eta = self.data.get('remaining_loop_seconds')
        if self.phase == 'training':
            times.append('loop ETA ' + (f'~{_duration(eta)}' if _number(eta) and eta >= 0 else 'estimating')
                         + ' (publication extra)')
        if self.prior_seconds:
            times.append(f'charged {_duration(self.prior_seconds + elapsed)}')
        rows.append(' | '.join(times))
        return rows

    def tick(self):
        if self.tty:
            self.render()

    def message(self, text):
        self.write(self._clear_frame() + text.rstrip('\n') + '\n')

    def _clear_frame(self):
        if not self.tty:
            return ''
        result = '\r\x1b[2K' + '\x1b[1A\r\x1b[2K' * max(0, self._frame_rows - 1)
        self._frame_rows = 0
        return result

    def feed_stdout(self, chunk, *, final=False):
        """Read bounded line fragments. The supervisor saves the original bytes."""
        self._stdout += chunk
        while b'\n' in self._stdout:
            line, self._stdout = self._stdout.split(b'\n', 1)
            self._line(line)
        if final or len(self._stdout) >= 65536:
            if self._stdout:
                self._line(self._stdout)
            self._stdout = b''

    def _line(self, raw):
        line = raw.decode('utf-8', errors='replace').rstrip('\r')
        event = parse_event(line)
        if event is not None:
            self.accept(event)
        elif line.startswith('Runtime identity differs or is incomplete.') and line not in self._warnings:
            self._warnings.add(line)
            self.message(line)
        # Library banners and progress bars remain in stdout.log. Only the
        # supervisor and structured events own the visible training display.

    def feed_stderr(self, chunk):
        """Keep a bounded diagnostic tail; the capture retains every byte."""
        self._stderr = (self._stderr + chunk)[-8192:]

    def finish(self, status):
        if self.finished:
            return
        self.render(immediate=True)
        if status == 'EXITED' and self.result_published:
            self.message('Training complete. Final adapter and worker result are published.')
        elif status == 'EXITED' and self.worker_status == 'CLEAN_STOP':
            self.message('Training stopped at a clean boundary. Resume eligible: '
                         + ('yes (validated clean stop).' if self.data.get('resume_eligible') is True else 'unverified.'))
        elif status == 'EXITED':
            self.message('Worker exited. Successful final publication was not confirmed.')
        else:
            self.message(f'Training did not complete. Supervisor status: {status}.')
        if status != 'EXITED' or not (self.result_published or self.worker_status == 'CLEAN_STOP'):
            if self.capture_path is not None:
                try:
                    with (self.capture_path / 'stderr.log').open('rb') as stream:
                        stream.seek(0, 2)
                        stream.seek(max(0, stream.tell() - 8192))
                        self._stderr = stream.read(8192)
                except OSError:
                    pass
            tail = '\n'.join(self._stderr.decode('utf-8', errors='replace').splitlines()[-12:])
            if tail:
                self.message(tail)
            if self.capture_path is not None:
                self.message(f'Worker diagnostics: {self.capture_path / "stderr.log"}')
        self.finished = True
