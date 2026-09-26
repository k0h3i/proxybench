"""Save exact update trajectories and restore only their current safe boundary."""

from contextlib import nullcontext
import os
import math
from pathlib import Path
import random
import time
import uuid

from proxybench.execution.resources import durable_json
from proxybench.training.checkpoints import (plain_path, require_hash, require_position,
                                           validate_checkpoint)
from proxybench.training.adapters import digest


def sample_order(count=96, epochs=2, seed=42):
    rng = random.Random(seed)
    order = []
    for _ in range(epochs):
        epoch = list(range(count))
        rng.shuffle(epoch)
        order.extend(epoch)
    return order


def weighted_loss(rows):
    return sum(r['loss'] * r['response_tokens'] for r in rows) / sum(r['response_tokens'] for r in rows)


def random_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                device=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_random(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['device']:
        torch.cuda.set_rng_state_all(state['device'])


def assert_same(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or not torch.isfinite(left).all() or not torch.isfinite(right).all()
                or not torch.equal(left.cpu(), right.cpu())):
            raise ValueError('Saved tensor differs')
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise ValueError('Saved array differs')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('Saved mapping differs')
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('Saved sequence differs')
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    elif left != right:
        raise ValueError('Saved state differs')


def parameter_map(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def publish_state(model, optimizer, tokenizer, directory, *, identity, order, completed, history,
                  get_weights=None, save_adapter=None, diagnostics=None, measurements=None):
    """Read tensors back before publishing. Keep interrupted staging intact."""
    import torch
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    claim = directory.with_name(directory.name + '.claim')
    with claim.open('x') as stream:
        stream.write(str(os.getpid()))
    stage = directory.with_name(directory.name + '.incomplete-' + uuid.uuid4().hex)
    stage.mkdir()
    state = dict(weights=(get_weights or model.state_dict)(), optimizer=optimizer.state_dict(),
                 parameter_map=parameter_map(model, optimizer), random=random_state(),
                 order=order, completed=completed, next_position=completed, history=history,
                 diagnostics=diagnostics or {})
    # A private local checkpoint contains Python and NumPy random states.
    measure = measurements.stage if measurements else lambda _: nullcontext()
    with measure('checkpoint_serialization'):
        torch.save(state, stage / 'state.pt')
    with measure('checkpoint_readback'):
        loaded = torch.load(stage / 'state.pt', map_location='cpu', weights_only=False)
    with measure('checkpoint_comparison'):
        assert_same(state, loaded)
    if save_adapter:
        save_adapter(stage / 'adapter')
    if tokenizer:
        tokenizer.save_pretrained(stage / 'tokenizer')
    with measure('checkpoint_hashing'):
        files = {str(p.relative_to(stage)): digest(p) for p in stage.rglob('*') if p.is_file()}
    durable_json(stage / 'manifest.json', dict(identity=identity, completed=completed, files=files))
    with measure('checkpoint_synchronization'):
        for path in stage.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        durable_json(stage / 'complete.json', dict(status='COMPLETE', manifest_sha256=digest(stage / 'manifest.json')))
    with measure('checkpoint_hashing'):
        validate_checkpoint(stage, identity, allow_temporary=True)
    with measure('checkpoint_publication'):
        stage.rename(directory)
        fd = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        claim.unlink()
    # Saving must not alter the random trajectory that the checkpoint records.
    restore_random(state['random'])
    return directory


def validate_trajectory(state, order, completed):
    """Inspect CPU state without changing models or the active random state."""
    import numpy as np
    import torch
    require_position(completed, len(order), 'Completed position')
    if not isinstance(state, dict):
        raise ValueError('Checkpoint state must be an object')
    required = {'weights', 'optimizer', 'parameter_map', 'random', 'order', 'completed', 'next_position', 'history'}
    if not required <= state.keys():
        raise ValueError('Checkpoint state is missing required fields')
    require_position(state['completed'], len(order), 'Checkpoint completed position')
    require_position(state['next_position'], len(order), 'Checkpoint next position')
    if (not isinstance(state['order'], list) or any(type(i) is not int for i in state['order'])
            or state['order'] != order or state['completed'] != completed or state['next_position'] != completed):
        raise ValueError('Checkpoint trajectory positions or sample order differ')
    history = state['history']
    if not isinstance(history, list) or len(history) != completed:
        raise ValueError('Checkpoint history length differs from its completed position')
    for position, row in enumerate(history):
        if (not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] != position + 1
                or type(row.get('index')) is not int or row['index'] != order[position]
                or type(row.get('response_tokens')) is not int or row['response_tokens'] <= 0
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                       for key in ('loss', 'update_seconds')) or row['update_seconds'] < 0):
            raise ValueError('Checkpoint history contains an invalid update')
    mapping = state['parameter_map']
    if (not isinstance(state['weights'], dict) or not state['weights']
            or not isinstance(state['optimizer'], dict)
            or not isinstance(state['optimizer'].get('state'), dict)
            or not isinstance(state['optimizer'].get('param_groups'), list)
            or not isinstance(mapping, list) or not mapping
            or any(not isinstance(group, list) or not group or any(not isinstance(n, str) or not n for n in group)
                   for group in mapping)):
        raise ValueError('Checkpoint weights or optimizer structure is invalid')
    groups = state['optimizer']['param_groups']
    if (len(groups) != len(mapping)
            or any(not isinstance(group, dict) or not isinstance(group.get('params'), list)
                   or len(group['params']) != len(names)
                   or any(type(index) is not int or index < 0 for index in group['params'])
                   for group, names in zip(groups, mapping))
            or ('diagnostics' in state and not isinstance(state['diagnostics'], dict))):
        raise ValueError('Checkpoint optimizer groups or diagnostics are invalid')
    parameter_ids = [index for group in groups for index in group['params']]
    if (len(set(parameter_ids)) != len(parameter_ids)
            or any(type(index) is not int or index not in parameter_ids or not isinstance(value, dict)
                   for index, value in state['optimizer']['state'].items())):
        raise ValueError('Checkpoint optimizer state mapping is invalid')
    rng = state['random']
    if not isinstance(rng, dict) or not {'python', 'numpy', 'cpu', 'device'} <= rng.keys():
        raise ValueError('Checkpoint random state is missing required fields')
    try:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['cpu'])
        if (not isinstance(rng['device'], list)
                or any(not isinstance(v, torch.Tensor) or v.device.type != 'cpu'
                       or v.dtype != torch.uint8 or v.ndim != 1 or not v.numel() for v in rng['device'])):
            raise ValueError('Invalid device random state')
    except (TypeError, ValueError, RuntimeError, IndexError) as exc:
        raise ValueError('Checkpoint random state is invalid') from exc
    return state


def load_state(directory, *, identity, order, completed):
    """Hash and load a trusted local checkpoint on CPU before GPU setup."""
    manifest = validate_checkpoint(directory, identity)
    require_position(manifest.get('completed'), len(order), 'Checkpoint completed position')
    if manifest['completed'] != completed:
        raise ValueError('Checkpoint is not the last completed update')
    if 'state.pt' not in manifest['files']:
        raise ValueError('Checkpoint inventory has no training state')
    import torch
    import pickle
    try:
        # Hashes detect changed bytes. They do not make untrusted pickle safe.
        state = torch.load(Path(directory) / 'state.pt', map_location='cpu', weights_only=False)
    except (OSError, ValueError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError('Trusted checkpoint state cannot be read') from exc
    return validate_trajectory(state, order, completed)


def restore_loaded_state(model, optimizer, state, *, order, completed, set_weights=None, get_weights=None):
    validate_trajectory(state, order, completed)
    if state['parameter_map'] != parameter_map(model, optimizer):
        raise ValueError('Checkpoint parameter mapping differs')
    current_groups = optimizer.state_dict()['param_groups']
    saved_groups = state['optimizer']['param_groups']
    if (len(saved_groups) != len(current_groups)
            or any(not isinstance(saved, dict)
                   or {key: value for key, value in saved.items() if key != 'params'}
                   != {key: value for key, value in current.items() if key != 'params'}
                   for saved, current in zip(saved_groups, current_groups))):
        raise ValueError('Checkpoint optimizer settings differ from the initialized recipe')
    (set_weights or model.load_state_dict)(state['weights'])
    assert_same(state['weights'], (get_weights or model.state_dict)())
    optimizer.load_state_dict(state['optimizer'])
    assert_same(state['optimizer'], optimizer.state_dict())
    # Initialization and restore can consume random numbers. Restore RNG last.
    restore_random(state['random'])
    return state


def restore_state(model, optimizer, directory, *, identity, order, completed,
                  set_weights=None, get_weights=None):
    state = load_state(directory, identity=identity, order=order, completed=completed)
    return restore_loaded_state(model, optimizer, state, order=order, completed=completed,
                                set_weights=set_weights, get_weights=get_weights)


def require_clean_stop(journal, checkpoint, identity, *, total=None, hash_files=True):
    if not isinstance(journal, dict) or journal.get('status') != 'CLEAN_STOP':
        raise ValueError('Resume requires a CLEAN_STOP journal. Use a new run folder for other states.')
    if not {'checkpoint', 'completed', 'pending_step', 'checkpoint_sha256'} <= journal.keys():
        raise ValueError('Clean-stop journal is missing required fields')
    if journal['pending_step'] is not None:
        raise ValueError('Clean-stop journal cannot contain a pending update')
    position = journal['completed']
    require_position(position, total if total is not None else max(position, 0) if type(position) is int else 0,
                     'Journal completed position')
    saved_path = journal['checkpoint']
    if not isinstance(saved_path, str) or not Path(saved_path).is_absolute():
        raise ValueError('Clean-stop checkpoint path must be absolute')
    path = plain_path(saved_path)
    if saved_path != str(path) or path != plain_path(checkpoint):
        raise ValueError('Clean-stop checkpoint path differs')
    require_hash(journal['checkpoint_sha256'], 'Clean-stop checkpoint hash')
    manifest = validate_checkpoint(path, identity, hash_files=hash_files)
    if 'state.pt' not in manifest['files']:
        raise ValueError('Clean-stop checkpoint inventory has no training state')
    require_position(manifest.get('completed'), total if total is not None else position,
                     'Checkpoint completed position')
    if (position != manifest['completed']
            or journal['checkpoint_sha256'] != digest(path / 'manifest.json')):
        raise ValueError('Resume requires the exact checkpoint from a clean stop')
    return position


def train_updates(model, optimizer, order, loss_fn, journal_path, *, start=0, history=None,
                  stop=lambda: False, save=None, checkpoint_interval=48, before_update=None, after_update=None,
                  memory=lambda: {}, report=print, checkpoint_steps=(), measurements=None,
                  sequence_tokens=None, epoch_boundaries=None, compute_timer=None):
    """Train deterministic updates and preserve a clean resume boundary."""
    import torch
    from proxybench.training.measurements import epoch_progress
    history = list(history or [])
    completed = start
    measure = measurements.stage if measurements else lambda _: nullcontext()
    last_checkpoint = None
    loop_started = measurements.clock() if measurements else time.monotonic()

    def journal(status, pending=None, **extra):
        with measure('journal'):
            durable_json(journal_path, dict(status=status, completed=completed, pending_step=pending, **extra))

    def checkpoint(clean):
        nonlocal last_checkpoint
        with measure('checkpoint'):
            path = save(completed, history, clean)
        if path is not None:
            last_checkpoint = dict(path=str(path), global_step=completed)
        return path

    def clean_stop():
        if save is None:
            raise ValueError('A clean stop requires a checkpoint saver')
        path = checkpoint(True)
        journal('CLEAN_STOP', checkpoint=str(path.resolve()), checkpoint_sha256=digest(path / 'manifest.json'))
        return dict(status='CLEAN_STOP', completed=completed, history=history)

    def measured_update(row, started, compute_seconds):
        if measurements is None:
            return
        elapsed = measurements.clock() - started
        row['full_loop_seconds'] = elapsed
        count = sequence_tokens(row['index']) if sequence_tokens else row['response_tokens']
        fields = dict(loss=row['loss'], weighted_loss_12=weighted_loss(history[-12:]),
                      learning_rate=optimizer.param_groups[0]['lr'], gradient_norm=row['gradient_norm'],
                      planned_updates=len(order), resume_eligible=False)
        if last_checkpoint is not None:
            fields['last_checkpoint'] = last_checkpoint
        if epoch_boundaries:
            fields.update(epoch_progress(completed, epoch_boundaries))
        # Event publication is measured separately, outside its own rate window.
        with measure('measurement_overhead'):
            measurements.record_update(global_step=completed, sample_position=completed,
                nonpadding_tokens=count, supervised_tokens=row['response_tokens'],
                full_loop_seconds=elapsed, compute_seconds=compute_seconds,
                compute_timing='device_events' if compute_timer else 'host_wall',
                remaining_loop_seconds=(None if len(measurements.updates) < 3 else
                    (measurements.clock() - loop_started)
                    / (completed - start) * (len(order) - completed)), **fields)

    journal('READY')
    for position in range(start, len(order)):
        if stop():
            return clean_stop()
        started = measurements.clock() if measurements else time.monotonic()
        journal('UPDATING', position + 1)
        if before_update:
            before_update(position + 1)
        tick = measurements.clock() if measurements else time.monotonic()
        if compute_timer:
            with measure('measurement_overhead'):
                compute_timer.start()
        with measure('compute'):
            optimizer.zero_grad(set_to_none=True)
            loss, count = loss_fn(order[position])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm')
            optimizer.step()
            # This existing scalar read waits for loss calculation, not isolated GPU computation.
            value = loss.item()
        if compute_timer:
            with measure('measurement_overhead'):
                compute_timer.stop()
        compute_seconds = (measurements.clock() if measurements else time.monotonic()) - tick
        completed = position + 1
        with measure('monitor'):
            sampled = memory()
        row = dict(step=completed, index=order[position], loss=value, response_tokens=count,
                   update_seconds=compute_seconds, gradient_norm=float(norm), **sampled)
        if compute_timer:
            with measure('measurement_overhead'):
                compute_seconds = compute_timer.seconds()
                row['device_compute_seconds'] = compute_seconds
        if epoch_boundaries:
            row.update(epoch_progress(completed, epoch_boundaries))
        history.append(row)
        journal('BOUNDARY')
        with measure('report'):
            if report is not None:
                report(f'Optimizer updates: {completed}/{len(order)} | loss {value:.5f} | '
                       f'response-token-weighted loss (last 12 updates) {weighted_loss(history[-12:]):.5f} | '
                       f'learning rate {optimizer.param_groups[0]["lr"]:g}')
        if stop():
            result = clean_stop()
            measured_update(row, started, compute_seconds)
            return result
        if after_update:
            after_update(completed, history)
        if stop():
            result = clean_stop()
            measured_update(row, started, compute_seconds)
            return result
        if save and (completed % checkpoint_interval == 0 or completed in checkpoint_steps):
            checkpoint(False)
        if stop():
            result = clean_stop()
            measured_update(row, started, compute_seconds)
            return result
        measured_update(row, started, compute_seconds)
    journal('TRAINED')
    return dict(status='TRAINED', completed=completed, history=history)
