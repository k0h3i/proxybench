"""Atomic run state and a process lock for one local writer."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def binding(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Run:
    def __init__(self, path, *, create=False, identity=None, config=None):
        self.path = Path(path)
        self.create = create
        self.identity = identity
        self.config = config or {}

    def __enter__(self):
        if self.create:
            self.path.mkdir(parents=True, exist_ok=False)
        self.lock = (self.path / '.writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.create:
                self.state = dict(schema='proxybench-run-v1', identity=self.identity,
                                  configuration=self.config, status='READY', consumed_seconds=0,
                                  pending_charge=None, resource_limit_seconds=self.config.get('limits', {}).get('total_seconds', self.config.get('total_seconds', 3600)))
                self.save()
            else:
                self.state = json.loads((self.path / 'run.json').read_text())
                if self.state.get('schema') != 'proxybench-run-v1':
                    raise ValueError('Unsupported run schema')
                if self.identity is not None and self.state['identity'] != self.identity:
                    raise ValueError('Run inputs changed; use a new run folder')
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *args):
        self.lock.close()

    def save(self):
        atomic_json(self.path / 'run.json', self.state)

    def reserve(self, target, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('A finite positive request limit is required')
        # An interrupted reservation remains charged. Resume cannot reset its cost.
        limit = self.state['resource_limit_seconds']
        if self.state['consumed_seconds'] + seconds > limit:
            raise ValueError('The cumulative run time limit is exhausted')
        self.state['consumed_seconds'] += seconds
        self.state['pending_charge'] = dict(target=target, seconds=seconds)
        self.save()

    def settle(self, target, seconds):
        pending = self.state['pending_charge']
        if pending is not None and pending['target'] == target:
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid saved resource usage')
            self.state['consumed_seconds'] += seconds - pending['seconds']
            self.state['pending_charge'] = None
            self.save()
