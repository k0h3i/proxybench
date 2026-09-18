"""Save exact update trajectories and restore only their current safe boundary."""

import os
from pathlib import Path
import random
import time
import uuid

from proxybench.execution.resources import durable_json
from proxybench.training.checkpoints import validate_checkpoint
from proxybench.training.smoke import digest, read_json


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
                  get_weights=None, save_adapter=None, diagnostics=None):
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
    torch.save(state, stage / 'state.pt')
    loaded = torch.load(stage / 'state.pt', map_location='cpu', weights_only=False)
    assert_same(state, loaded)
    if save_adapter:
        save_adapter(stage / 'adapter')
    if tokenizer:
        tokenizer.save_pretrained(stage / 'tokenizer')
    files = {str(p.relative_to(stage)): digest(p) for p in stage.rglob('*') if p.is_file()}
    durable_json(stage / 'manifest.json', dict(identity=identity, completed=completed, files=files))
    for path in stage.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    durable_json(stage / 'complete.json', dict(status='COMPLETE', manifest_sha256=digest(stage / 'manifest.json')))
    validate_checkpoint(stage, identity, allow_temporary=True)
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


def restore_state(model, optimizer, directory, *, identity, order, completed,
                  set_weights=None, get_weights=None):
    import torch
    manifest = validate_checkpoint(directory, identity)
    if manifest['completed'] != completed:
        raise ValueError('Checkpoint is not the last completed update')
    state = torch.load(Path(directory) / 'state.pt', map_location='cpu', weights_only=False)
    if (state['order'] != order or state['completed'] != completed or state['next_position'] != completed
            or state['parameter_map'] != parameter_map(model, optimizer)
            or len(state['history']) != completed
            or [row['index'] for row in state['history']] != order[:completed]):
        raise ValueError('Checkpoint trajectory or parameter mapping differs')
    (set_weights or model.load_state_dict)(state['weights'])
    assert_same(state['weights'], (get_weights or model.state_dict)())
    optimizer.load_state_dict(state['optimizer'])
    assert_same(state['optimizer'], optimizer.state_dict())
    # Initialization and restore can consume random numbers. Restore RNG last.
    restore_random(state['random'])
    return state


def require_clean_stop(journal, checkpoint, identity):
    manifest = validate_checkpoint(checkpoint, identity)
    if (journal.get('status') != 'CLEAN_STOP' or journal.get('pending_step') is not None
            or journal.get('checkpoint') != str(Path(checkpoint).resolve())
            or journal['completed'] != manifest['completed']
            or journal.get('checkpoint_sha256') != digest(Path(checkpoint) / 'manifest.json')):
        raise ValueError('Resume requires the exact checkpoint from a clean stop')
    return journal['completed']


def train_updates(model, optimizer, order, loss_fn, journal_path, *, start=0, history=None,
                  stop=lambda: False, save=None, checkpoint_interval=48, before_update=None, after_update=None,
                  memory=lambda: {}, report=print):
    """Use this same loop for the pilot and the CPU resume equivalence test."""
    import torch
    history = list(history or [])
    begin = time.monotonic()
    completed = start

    def journal(status, pending=None, **extra):
        durable_json(journal_path, dict(status=status, completed=completed, pending_step=pending, **extra))

    def clean_stop():
        if save is None:
            raise ValueError('A clean stop requires a checkpoint saver')
        path = save(completed, history, True)
        journal('CLEAN_STOP', checkpoint=str(path.resolve()), checkpoint_sha256=digest(path / 'manifest.json'))
        return dict(status='CLEAN_STOP', completed=completed, history=history)

    journal('READY')
    for position in range(start, len(order)):
        if stop():
            return clean_stop()
        journal('UPDATING', position + 1)
        if before_update:
            before_update(position + 1)
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, count = loss_fn(order[position])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradient norm')
        optimizer.step()
        # Copying the loss also synchronizes the completed device update.
        value = loss.item()
        completed = position + 1
        row = dict(step=completed, index=order[position], loss=value, response_tokens=count,
                   update_seconds=time.monotonic()-tick, **memory())
        history.append(row)
        journal('BOUNDARY')
        elapsed = time.monotonic()-begin
        recent = weighted_loss(history[-12:])
        steps_here = completed-start
        eta = f'{sum(r["update_seconds"] for r in history[-12:])/min(12, len(history))*(len(order)-completed):.0f}s' if steps_here >= 4 else 'estimating'
        report(f'step {completed:03}/{len(order)} | epoch {completed/(len(order)/2):.2f}/2 | '
               f'loss {value:.5f} | recent loss {recent:.5f} | lr {optimizer.param_groups[0]["lr"]:g} | '
               f'elapsed {elapsed:.0f}s | training left ~{eta} | '
               f'allocated {row.get("allocated_bytes", 0)/1024**3:.2f} GiB | '
               f'device free {row.get("free_bytes", 0)/1024**3:.2f} GiB')
        if stop():
            return clean_stop()
        if after_update:
            after_update(completed, history)
        if save and completed % checkpoint_interval == 0:
            save(completed, history, False)
        if stop():
            return clean_stop()
    journal('TRAINED')
    return dict(status='TRAINED', completed=completed, history=history)
