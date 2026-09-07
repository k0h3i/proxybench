"""Freeze a schedule, capture one attempt, and score after capture."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import time

from proxybench.annotation.bindings import checked_file, load_input_binding, sha256
from proxybench.evaluation.evidence import SourceContext
from proxybench.evaluation.references import admit_reference
from proxybench.evaluation.scoring import parsed_response, score_fragment
from proxybench.extraction.bundles import read_bundle
from proxybench.normalization.values import ACTIVE_GUIDE, policy_version
from proxybench.schemas.records import require, strict_json

RUNNER_VERSION = 'development-runner-v1'


def json_safe(value):
    if isinstance(value, Decimal):
        return {'json_type': 'number', 'value': str(value)}
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(json_safe(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ScheduledInput:
    input_id: str
    input_path: str
    input_sha256: str
    reference_path: str
    reference_sha256: str
    binding_sha256: str
    split: str = "development"


@dataclass(frozen=True)
class Association:
    run_id: str
    system_id: str
    input_id: str
    input_sha256: str
    attempt: int = 1


@dataclass(frozen=True)
class RecordedCapture:
    association: Association
    raw: bytes | None
    outcome: str = 'EXITED'
    returncode: int | None = 0
    truncated: bool = False
    capture_complete: bool = True


class RecordedPredictionAdapter:
    """Import operator-bound exact bytes, without editing echoed response metadata."""

    def __init__(self, captures):
        self.captures = {}
        for capture in captures:
            require(capture.association not in self.captures, 'Duplicate recorded capture association')
            require(capture.raw is None or isinstance(capture.raw, bytes), 'Recorded output must be bytes or absent')
            require(capture.outcome in ('EXITED', 'TIMEOUT', 'CRASHED', 'CAPTURE_FAILED', 'MISSING', 'TRUNCATED'),
                    'Unknown recorded process outcome')
            self.captures[capture.association] = capture

    def validate_schedule(self, associations):
        require(set(self.captures) <= set(associations), 'Recorded capture is attached to the wrong schedule')

    def capture(self, association, input_path, directory, limits):
        item = self.captures.get(association)
        if item is None:
            return {'outcome': 'MISSING', 'returncode': None, 'truncated': False, 'capture_complete': False,
                    'output_present': False, 'elapsed_seconds': None, 'measurement_reason': 'No recorded capture supplied'}
        observation = {'outcome': item.outcome, 'returncode': item.returncode, 'truncated': item.truncated,
                       'capture_complete': item.capture_complete, 'output_present': item.raw is not None,
                       'elapsed_seconds': None, 'measurement_reason': 'Replay does not measure original extraction runtime'}
        if item.raw is not None and len(item.raw) > limits['output_bytes']:
            observation.update(outcome='TRUNCATED', truncated=True,
                               limit_reason='Recorded output exceeds the declared byte limit; exact bytes retained')
        if item.raw is not None:
            with (directory / 'raw.bin').open('xb') as stream:
                written = stream.write(item.raw)
                stream.flush()
                os.fsync(stream.fileno())
                require(written == len(item.raw), 'Incomplete recorded output write')
        return observation


class SubprocessAdapter:
    """Execute inspected local code, with bounded streams and no retries."""

    def __init__(self, command, *, cwd=None, environment=None):
        require(isinstance(command, (list, tuple)) and bool(command), 'Subprocess command is required')
        self.command = list(command)
        self.cwd = cwd
        self.environment = environment

    def validate_schedule(self, associations):
        pass

    def capture(self, association, input_path, directory, limits):
        started = time.monotonic()
        observation = {'outcome': 'EXITED', 'returncode': None, 'truncated': False,
                       'capture_complete': True, 'output_present': False, 'started_at': utc()}
        process = None
        try:
            with input_path.open('rb') as source, (directory / 'raw.bin').open('xb') as output, (directory / 'stderr.bin').open('xb') as errors:
                process = subprocess.Popen(self.command, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           cwd=self.cwd, env=self.environment, start_new_session=True)
                observation['output_present'] = True
                counts = {'stdout': 0, 'stderr': 0}
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ, ('stdout', output))
                    selector.register(process.stderr, selectors.EVENT_READ, ('stderr', errors))
                    while selector.get_map():
                        remaining = limits['timeout_seconds'] - (time.monotonic() - started)
                        if remaining <= 0:
                            observation['outcome'] = 'TIMEOUT'
                            break
                        for key, _ in selector.select(min(remaining, 0.05)):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            name, stream = key.data
                            allowed = limits['output_bytes'] - counts[name]
                            kept = chunk[:allowed]
                            if stream.write(kept) != len(kept):
                                raise OSError('Incomplete output write')
                            counts[name] += len(kept)
                            if len(chunk) > allowed:
                                observation.update(outcome='TRUNCATED', truncated=True)
                                break
                        if observation['outcome'] != 'EXITED':
                            break
                    if observation['outcome'] == 'EXITED':
                        try:
                            process.wait(timeout=max(0.001, limits['timeout_seconds'] - (time.monotonic() - started)))
                        except subprocess.TimeoutExpired:
                            observation['outcome'] = 'TIMEOUT'
                    if observation['outcome'] != 'EXITED':
                        observation['capture_complete'] = False
                        self._kill(process)
                    else:
                        process.wait()
                    observation['returncode'] = process.returncode
                    if observation['outcome'] == 'EXITED' and process.returncode != 0:
                        observation['outcome'] = 'CRASHED'
                for stream in (output, errors):
                    stream.flush()
                    os.fsync(stream.fileno())
        except (OSError, ValueError) as error:
            observation.update(outcome='CAPTURE_FAILED', capture_complete=False, error=str(error))
        finally:
            if process is not None:
                if process.poll() is None:
                    self._kill(process)
                process.stdout.close()
                process.stderr.close()
            observation.update(ended_at=utc(), elapsed_seconds=time.monotonic() - started)
        return observation

    @staticmethod
    def _kill(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def preflight(root, item, guide_version):
    require(item.split == 'development', 'This runner accepts development inputs only')
    raw = checked_file(root, item.input_path, item.input_sha256)
    bundle = read_bundle(raw)
    require(bundle['input_id'] == item.input_id, 'Scheduled input identity differs')
    reference_data = strict_json(checked_file(root, item.reference_path, item.reference_sha256))
    require(reference_data['input_binding_sha256'] == item.binding_sha256, 'Scheduled binding differs')
    binding = load_input_binding(reference_data, root, input_id=item.input_id, model_input_sha256=item.input_sha256)
    manifest = strict_json(checked_file(root, binding['source_manifest_path'], binding['source_manifest_sha256']))
    original = checked_file(root, manifest['source_path'], bundle['source_sha256'])
    require(manifest.get('packet_id', item.input_id) == item.input_id, 'Manifest input identity differs')
    require(manifest['source_sha256'] == bundle['source_sha256'] and manifest['target'] == bundle['target'],
            'Source manifest differs from bundle')
    expected = {b['block_id']: (b['start_byte'], b['end_byte']) for b in manifest['blocks']}
    actual = {b['block_id']: (b['span']['start_byte'], b['span']['end_byte']) for b in bundle['blocks']}
    require(actual == expected, 'Permitted source ranges differ from binding')
    views = {}
    for b in bundle['blocks']:
        a, end = actual[b['block_id']]
        require(original[a:end] == b['original_text'].encode(bundle['encoding']), 'Bundle slice differs from original source')
        views[bundle['view_id'], b['block_id']] = {'span': b['span'], 'text': b['prepared_text']}
    doc = bundle['document_id']
    context = SourceContext({doc: original}, {doc: list(actual.values())}, {doc: bundle['encoding']}, views)
    reference = admit_reference(reference_data, root, context, input_id=item.input_id,
                                model_input_sha256=item.input_sha256, guide_version=guide_version)
    return raw, reference, context


def software_observation(root):
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git('status', '--porcelain')
    package = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(package)): sha256(p.read_bytes()) for p in sorted(package.rglob('*.py'))}
    return {'revision': git('rev-parse', 'HEAD'), 'dirty': bool(status) if status is not None else None,
            'package_hashes': files, 'python': sys.version, 'platform': platform.platform(),
            'machine': platform.machine(), 'cpu_count': os.cpu_count(), 'dependencies': 'Python standard library'}


def run_development(root, destination, schedule, adapter, *, run_id, system_id, guide_version=ACTIVE_GUIDE,
                    timeout_seconds=10, output_bytes=1048576):
    """Run only admitted development fragments. Refuse an existing destination."""
    policy_version(guide_version)
    require(isinstance(timeout_seconds, (int, float)) and 0 < timeout_seconds < float('inf'), 'Set a finite positive timeout')
    require(type(output_bytes) is int and output_bytes > 0, 'Set a positive output byte limit')
    require(bool(run_id) and bool(system_id), 'Run and system identities are required')
    schedule = tuple(schedule)
    require(bool(schedule) and len({s.input_id for s in schedule}) == len(schedule), 'Empty or duplicate input schedule')
    destination = Path(destination)
    require(not destination.exists(), 'Output destination already exists')
    associations = [Association(run_id, system_id, s.input_id, s.input_sha256) for s in schedule]
    adapter.validate_schedule(associations)
    # Admission checks happen before any extractor launch. These objects stay in the parent.
    prepared = [preflight(root, item, guide_version) for item in schedule]
    limits = {'timeout_seconds': timeout_seconds, 'output_bytes': output_bytes, 'max_attempts': 1}
    manifest = {'run_id': run_id, 'system_id': system_id, 'runner_version': RUNNER_VERSION,
                'purpose': 'DEVELOPMENT_ONLY', 'track': 'fragment', 'guide_version': guide_version,
                'parser_policy_version': policy_version(guide_version), 'schema_version': 'benchmark-v1',
                'scorer_version': 'fragment-' + policy_version(guide_version),
                'schedule': [asdict(s) for s in schedule], 'associations': [asdict(a) for a in associations],
                'limits': limits, 'software': software_observation(root), 'frozen_at': utc(),
                'command': getattr(adapter, 'command', None), 'adapter': type(adapter).__name__,
                'working_directory': str(getattr(adapter, 'cwd', None)),
                'environment_policy': 'Inherited process environment; values omitted to avoid recording credentials',
                'source_inventory': [{
                    'input_id': item.input_id, 'source_hashes': context.hashes,
                    'permitted_ranges': context.permitted_ranges}
                    for item, (_, _, context) in zip(schedule, prepared)],
                'measurement_boundary': 'Per-input process launch, extraction, stream capture, and durable flush; excludes admission and scoring',
                'peak_memory_bytes': None, 'memory_reason': 'Per-process peak memory instrumentation is not implemented',
                'token_count': None, 'token_reason': 'No model tokenizer participates in this development system',
                'isolation': 'Inspected source-only parser; no OS security sandbox; no controlled comparison claim'}
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'schedule.json', manifest)
    frozen_hash = sha256((destination / 'schedule.json').read_bytes())
    results = []
    for index, (item, association, prepared_item) in enumerate(zip(schedule, associations, prepared)):
        raw_input, reference, context = prepared_item
        directory = destination / f'{index:04d}'
        directory.mkdir()
        result = {'association': asdict(association), 'admission': 'PASS', 'execution_eligible': False,
                  'execution_success': False, 'content_passed': False}
        try:
            input_path = directory / 'input.bin'
            input_path.write_bytes(raw_input)
            # Catch changes since admission and reject them without an extractor invocation.
            preflight(root, item, guide_version)
            observation = adapter.capture(association, input_path, directory, limits)
        except Exception as error:
            observation = {'outcome': 'CAPTURE_FAILED', 'capture_complete': False, 'returncode': None,
                           'truncated': False, 'output_present': (directory / 'raw.bin').exists(), 'error': str(error)}
        result['observation'] = observation
        raw_path = directory / 'raw.bin'
        try:
            raw = raw_path.read_bytes() if raw_path.exists() else None
        except OSError as error:
            raw = None
            observation.update(outcome='CAPTURE_FAILED', capture_complete=False, error=str(error))
        if raw is not None:
            result['raw_sha256'] = sha256(raw)
        result['execution_eligible'] = (raw is not None and observation.get('outcome') == 'EXITED'
            and observation.get('returncode') == 0 and observation.get('capture_complete') is True
            and observation.get('truncated') is False)
        write_json(directory / 'observation.json', result)
        # A missing response gets failure accounting but no invented raw or parsed file.
        if raw is not None:
            parsed, _, normalized, errors, valid = parsed_response(raw, guide_version=guide_version)
            write_json(directory / 'parsed.json', {'value': parsed, 'errors': errors, 'envelope_valid': valid})
            write_json(directory / 'normalized.json', normalized)
            result['model_status'] = parsed.get('status') if isinstance(parsed, dict) else None
        else:
            result['model_status'] = None
        score = score_fragment(raw if raw is not None else b'', reference, context=context)
        write_json(directory / 'score.json', score)
        result['content_passed'] = score['passed']
        result['execution_success'] = result['execution_eligible'] and result['model_status'] == 'COMPLETE' and not score['schema_errors']
        result['primary_success'] = result['execution_eligible'] and score['passed']
        result['processing_coverage'] = {'input_bytes_supplied': len(raw_input),
                                         'input_bytes_consumed': None,
                                         'source_ranges_processed': None,
                                         'reason': 'Input supplied on stdin; consumption and source-range processing are not instrumented'}
        result['score_path'] = str((directory / 'score.json').relative_to(destination))
        results.append(result)
        write_json(directory / 'result.json', result)
    report = {'run_id': run_id, 'purpose': 'DEVELOPMENT_ONLY', 'frozen_schedule_sha256': frozen_hash,
              'scheduled': len(schedule), 'terminal_results': len(results), 'results': results,
              'execution_eligible': sum(r['execution_eligible'] for r in results),
              'primary_success': sum(r['primary_success'] for r in results)}
    write_json(destination / 'report.json', report)
    return report
