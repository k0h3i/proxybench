"""Render operation progress without changing worker execution or saved logs."""

from pathlib import Path
import re
import shutil
import time


PHASES = {
    'loading': 'Loading',
    'serialization': 'Saving merged model',
    'conversion': 'Conversion',
    'payload-inspection': 'Validating converted model',
    'evaluation': 'Evaluation',
    'cleanup': 'Cleanup',
}
_ESCAPES = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]')


def _duration(seconds):
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m' if hours else f'{minutes}m{seconds:02d}s'


def _counts(values):
    if not isinstance(values, dict):
        return None
    completed, total, failed = (values.get('completed', 0), values.get('total'), values.get('failed', 0))
    if any(type(value) is not int or value < 0 or value > 2**63 - 1
           for value in (completed, total, failed)):
        return None
    return (completed, total, failed) if failed <= completed <= total else None


def _plain(text):
    text = _ESCAPES.sub('', text)
    return ''.join(char for char in text if char == '\n' or (char.isprintable() and char != '\x1b'))


class OperationDisplay:
    """Keep optional display metadata separate from execution decisions."""

    def __init__(self, write, *, operation, capture_path, tty=False, clock=time.monotonic,
                 started=None, prior_seconds=0, phase_used=0, total_seconds=None,
                 phase_seconds=None, progress=None):
        if operation not in {'export', 'convert', 'evaluation'}:
            raise ValueError('Unsupported display operation')
        self.write, self.operation, self.tty, self.clock = write, operation, tty, clock
        self.capture_path = Path(capture_path)
        self.started = clock() if started is None else started
        self.prior_seconds, self.phase_used = prior_seconds, phase_used
        self.total_seconds, self.phase_seconds = total_seconds, phase_seconds
        self.phase = 'loading'
        self.counts = _counts(progress)
        self._initial = self.counts or (0, None, 0)
        self._session = None
        self.last_refresh = -float('inf')
        self._last_phase = None
        self._last_completed = None
        self._frame_rows = 0
        self.finished = False

    def observe(self, state):
        """Accept a known phase and valid, monotonic counts for this session."""
        if self.finished or not isinstance(state, dict):
            return
        phase = state.get('phase')
        if not isinstance(phase, str) or phase not in PHASES:
            return
        self.phase = phase
        if any(key in state for key in ('completed', 'total', 'failed')):
            remaining = (None if self._initial[1] is None
                         else self._initial[1] - self._initial[0])
            prior = self._session or (0, remaining, 0)
            values = dict(zip(('completed', 'total', 'failed'), prior))
            values.update({key: state[key] for key in values if key in state})
            counts = _counts(values)
            if counts is not None:
                completed, total, failed = counts
                initial_completed, initial_total, initial_failed = self._initial
                cumulative = (initial_completed + completed, initial_completed + total,
                              initial_failed + failed)
                if (completed >= prior[0] and failed >= prior[2]
                        and (prior[1] is None or total == prior[1])
                        and (initial_total is None or cumulative[1] == initial_total)):
                    self._session = counts
                    self.counts = cumulative
        if self.phase != self._last_phase:
            self.render(immediate=True)
        elif self.tty:
            self.tick()
        elif self.counts is not None:
            completed, total, _ = self.counts
            if (self._last_completed is None
                    or completed // 10 > self._last_completed // 10
                    or completed == total and completed != self._last_completed):
                self.render(immediate=True)

    def compact_rows(self):
        progress = PHASES[self.phase]
        if self.counts is not None:
            completed, total, failed = self.counts
            fraction = completed / total if total else 0
            percent = f'{fraction * 100:.1f}'.rstrip('0').rstrip('.')
            filled = int(fraction * 12)
            bar = '[' + '=' * filled + '.' * (12 - filled) + '] ' if self.tty else ''
            progress += f' {bar}{completed}/{total} ({percent}%) | generation failures {failed}'
        elapsed = max(0, self.clock() - self.started)
        times = [f'elapsed {_duration(elapsed)}']
        remaining = [ceiling - prior - elapsed for ceiling, prior in (
            (self.total_seconds, self.prior_seconds), (self.phase_seconds, self.phase_used))
            if ceiling is not None]
        if remaining:
            times.append(f'budget {_duration(min(remaining))}')
        if self.prior_seconds:
            times.append(f'charged {_duration(self.prior_seconds + elapsed)}')
        return [progress, ' | '.join(times)]

    def render(self, *, immediate=False):
        if self.finished:
            return
        now = self.clock()
        if not immediate and now - self.last_refresh < 1:
            return
        self.last_refresh = now
        self._last_phase = self.phase
        self._last_completed = self.counts[0] if self.counts is not None else None
        rows = self.compact_rows()
        if self.tty:
            width = max(1, shutil.get_terminal_size((120, 24)).columns - 1)
            rows = [row if len(row) <= width else row[:width - 1] + '~' for row in rows]
            self.write(self._clear_frame() + '\n'.join(rows))
            self._frame_rows = len(rows)
        else:
            self.write(' | '.join(rows) + '\n')

    def tick(self):
        if self.tty:
            self.render()

    def _clear_frame(self):
        if not self.tty or not self._frame_rows:
            return ''
        result = '\r\x1b[2K' + '\x1b[1A\r\x1b[2K' * (self._frame_rows - 1)
        self._frame_rows = 0
        return result

    def message(self, text):
        self.write(self._clear_frame() + _plain(text).rstrip('\n') + '\n')

    def feed_stdout(self, chunk, *, final=False):
        """The supervisor captures bytes; routine library output stays there."""

    def feed_stderr(self, chunk):
        """Read failure details from the durable log when the worker ends."""

    def _diagnostics(self):
        path = self.capture_path / 'stderr.log'
        try:
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 8192))
                tail = stream.read(8192)
            text = _plain(tail.decode('utf-8', errors='replace'))
            text = '\n'.join(text.splitlines()[-12:])
            if text:
                self.message(text)
        except OSError:
            pass
        self.message(f'Worker diagnostics: {path}')
        if self.operation == 'evaluation':
            self.message(f'Server diagnostics: {self.capture_path.parent / "answers/server.log"}')

    def finish(self, status):
        if self.finished:
            return
        self.render(immediate=True)
        complete = self.counts is not None and self.counts[0] == self.counts[1]
        if status != 'EXITED':
            self.message(f'{self.operation.capitalize()} did not complete. Supervisor status: {status}.')
        elif self.operation != 'evaluation':
            self.message(f'{self.operation.capitalize()} worker finished.')
        elif complete:
            self.message('Generation finished. Scoring and review are pending.')
        else:
            self.message('Generation incomplete. The worker exited before all results were confirmed.')
        if status != 'EXITED' or self.operation == 'evaluation' and not complete:
            self._diagnostics()
        self.finished = True
