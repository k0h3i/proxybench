"""Isolated Trainer experiment using the existing prepared loss and update contract."""

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import sys
import time
import uuid

from proxybench.execution.resources import durable_json
from proxybench.training.adapters import digest
from proxybench.training.trajectory import random_state, restore_random, weighted_loss


EXPERIMENT = 'plain-trainer-v1'
SCHEDULER = {'kind': 'constant-no-op-v1', 'mutable_state': False,
             'learning_rates': 'optimizer.param_groups.lr'}
EPOCH_SOURCE_SHA256 = 'c704c082dae4b742beb3787afb7636c247294aefbe5803b79f02994ab241221c'


def trainer_identity(trainer_class=None):
    """Record effective callable code, including dynamically generated patches."""
    if trainer_class is None:
        from transformers import Trainer
        trainer_class = Trainer
    methods = {}
    for name in ('__init__', 'train', '_inner_training_loop', '_run_epoch', 'training_step',
                 'get_batch_samples', 'compute_loss', 'get_train_dataloader', '_wrap_model'):
        function = getattr(trainer_class, name)
        chain = []
        while function is not None:
            code = getattr(function, '__code__', None)
            try:
                source = inspect.getsource(function)
            except (OSError, TypeError):
                source = None
            chain.append(dict(module=function.__module__, qualname=function.__qualname__,
                              code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
                              source_sha256=hashlib.sha256(source.encode()).hexdigest() if source else None,
                              source_file=Path(code.co_filename).name if code else None))
            function = getattr(function, '__wrapped__', None)
        methods[name] = chain
    return dict(experiment=EXPERIMENT, trainer=trainer_class.__module__ + '.' + trainer_class.__qualname__,
                methods=methods, scheduler=SCHEDULER)


class ConstantSchedule:
    """Trainer requires a scheduler. This one has no counter or LR mutation."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return dict(SCHEDULER)

    def load_state_dict(self, state):
        if state != SCHEDULER:
            raise ValueError('Trainer experiment scheduler state differs')


class _CleanStop(Exception):
    pass


def train_updates(model, optimizer, order, loss_fn, journal_path, *, start=0, history=None,
                  stop=lambda: False, save=None, checkpoint_interval=48, before_update=None,
                  after_update=None, memory=lambda: {}, report=print, checkpoint_steps=(),
                  measurements=None, sequence_tokens=None, epoch_boundaries=None, max_updates=None,
                  compute_timer=None):
    """Delegate backward and stepping to Trainer with one prepared example per update."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import Trainer, TrainerCallback, TrainingArguments

    if compute_timer is not None:
        raise ValueError('Device-event profiling is not supported by the Trainer experiment')
    # Installed Unsloth patches the outer loop, but this method owns update callbacks.
    # Refuse an unreviewed callback order before allowing any optimizer update.
    if hashlib.sha256(inspect.getsource(Trainer._run_epoch).encode()).hexdigest() != EPOCH_SOURCE_SHA256:
        raise ValueError('Trainer epoch implementation requires new acceptance')
    if (type(start) is not int or not 0 <= start <= len(order)
            or type(checkpoint_interval) is not int or checkpoint_interval <= 0
            or (max_updates is not None and (type(max_updates) is not int or max_updates <= 0))):
        raise ValueError('Invalid Trainer experiment update bounds')
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError('Trainer experiment requires the existing AdamW optimizer')
    history = list(history or [])
    if len(history) != start:
        raise ValueError('Trainer experiment history differs from its start position')
    device = next(model.parameters()).device
    if device.type not in {'cpu', 'cuda'} or len({p.device for p in model.parameters()}) != 1:
        raise ValueError('Trainer experiment requires one CPU or CUDA device')
    root = Path(journal_path).parent
    completed = start
    pending = {}
    initial_random = random_state()
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    clock = measurements.clock if measurements else time.monotonic
    last_checkpoint = None

    def stage(name):
        return measurements.stage(name) if measurements else nullcontext()

    def journal(status, pending_step=None, **extra):
        with stage('journal'):
            durable_json(journal_path, dict(status=status, completed=completed,
                                           pending_step=pending_step, **extra))

    def stopping():
        return stop() or (max_updates is not None and completed - start >= max_updates)

    def clean_stop():
        nonlocal last_checkpoint
        if save is None:
            raise ValueError('A clean stop requires a checkpoint saver')
        with stage('checkpoint'):
            path = save(completed, history, True)
        last_checkpoint = dict(path=str(path), global_step=completed)
        journal('CLEAN_STOP', checkpoint=str(path.resolve()), checkpoint_sha256=digest(path / 'manifest.json'))
        return dict(status='CLEAN_STOP', completed=completed, history=history)

    def record_update():
        if measurements is None or 'row' not in pending or pending.get('recorded'):
            return
        from proxybench.training.measurements import epoch_progress
        row = pending['row']
        elapsed = clock() - pending['full_begin']
        row['full_loop_seconds'] = elapsed
        fields = dict(loss=row['loss'], weighted_loss_12=weighted_loss(history[-12:]),
                      learning_rate=optimizer.param_groups[0]['lr'], gradient_norm=row['gradient_norm'],
                      planned_updates=len(order), resume_eligible=False)
        if epoch_boundaries:
            fields.update(epoch_progress(completed, epoch_boundaries))
        if last_checkpoint is not None:
            fields['last_checkpoint'] = last_checkpoint
        count = sequence_tokens(row['index']) if callable(sequence_tokens) else (
            sequence_tokens[row['index']] if sequence_tokens is not None else row['response_tokens'])
        with stage('measurement_overhead'):
            measurements.record_update(global_step=completed, sample_position=completed,
                nonpadding_tokens=count, supervised_tokens=row['response_tokens'],
                full_loop_seconds=elapsed, compute_seconds=row['update_seconds'],
                remaining_loop_seconds=(None if len(measurements.updates) < 3 else
                    (sum(r['full_loop_seconds'] for r in measurements.updates) + elapsed)
                    / (len(measurements.updates) + 1) * (len(order) - completed)), **fields)
        pending['recorded'] = True

    class Boundaries(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # Trainer construction resets seeds. Preparation can also consume RNG.
            restore_random(initial_random)

        def on_step_begin(self, args, state, control, **kwargs):
            if stopping():
                raise _CleanStop
            pending.clear()
            pending['full_begin'] = clock()
            journal('UPDATING', completed + 1)
            if before_update:
                before_update(completed + 1)
            optimizer.zero_grad(set_to_none=True)
            pending['tick'] = clock()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm')
            pending['gradient_norm'] = norm.item()

        def on_step_end(self, args, state, control, **kwargs):
            nonlocal completed, last_checkpoint
            if trainer.accelerator.optimizer_step_was_skipped:
                raise ValueError('Trainer experiment skipped an optimizer update')
            completed += 1
            compute_seconds = clock() - pending['tick']
            with stage('monitor'):
                observed = memory()
            row = dict(step=completed, index=order[completed - 1], loss=pending['loss'],
                       response_tokens=pending['count'], update_seconds=compute_seconds,
                       gradient_norm=pending['gradient_norm'], **observed)
            if epoch_boundaries:
                from proxybench.training.measurements import epoch_progress
                row.update(epoch_progress(completed, epoch_boundaries))
            pending['row'] = row
            history.append(row)
            journal('BOUNDARY')
            if report:
                with stage('report'):
                    report(f'Trainer experiment update {completed}/{len(order)} | loss {row["loss"]:.5f}')
            if stopping():
                raise _CleanStop
            if after_update:
                after_update(completed, history)
            if stopping():
                raise _CleanStop
            if save and (completed % checkpoint_interval == 0 or completed in checkpoint_steps):
                with stage('checkpoint'):
                    path = save(completed, history, False)
                if path is not None:
                    last_checkpoint = dict(path=str(path), global_step=completed)
            if stopping():
                raise _CleanStop
            record_update()

    class PreparedTrainer(Trainer):
        def get_train_dataloader(self):
            # A private generator prevents DataLoader iterator seeds from consuming model RNG.
            return DataLoader([{'index': index} for index in order[start:]], batch_size=1,
                              shuffle=False, num_workers=0, pin_memory=False,
                              generator=torch.Generator().manual_seed(0))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if return_outputs:
                raise ValueError('Trainer experiment does not produce output logits')
            index = int(inputs['index'].item())
            if index != order[completed]:
                raise ValueError('Trainer experiment sample order differs')
            loss, count = loss_fn(index)
            if loss.numel() != 1 or not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            if type(count) is not int or count <= 0:
                raise ValueError('Trainer experiment response token count is invalid')
            pending.update(loss=loss.detach().item(), count=count)
            return loss

    if start == len(order):
        journal('READY')
        if stopping():
            return clean_stop()
        journal('TRAINED')
        return dict(status='TRAINED', completed=completed, history=history)
    args = TrainingArguments(output_dir=str(root / 'trainer-internal'), use_cpu=device.type == 'cpu',
        per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=1,
        max_steps=len(order) - start, learning_rate=optimizer.param_groups[0]['lr'],
        weight_decay=optimizer.param_groups[0]['weight_decay'], lr_scheduler_type='constant',
        warmup_steps=0, optim='adamw_torch', max_grad_norm=0.0, bf16=False, fp16=False,
        tf32=False if device.type == 'cuda' else None, gradient_checkpointing=False,
        dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
        logging_strategy='no', save_strategy='no', eval_strategy='no', report_to=[],
        disable_tqdm=True, seed=42, data_seed=42, logging_nan_inf_filter=False,
        average_tokens_across_devices=False)
    try:
        trainer = PreparedTrainer(model=model, args=args, train_dataset=order[start:],
                                  optimizers=(optimizer, ConstantSchedule(optimizer)), callbacks=[Boundaries()])
        trainer.model_accepts_loss_kwargs = False
        if (trainer.accelerator.num_processes != 1 or args.n_gpu > 1
                or trainer.accelerator.mixed_precision != 'no'
                or trainer.accelerator.gradient_accumulation_steps != 1):
            raise ValueError('Trainer experiment requires one process with no added mixed precision')
        identity = trainer_identity(PreparedTrainer)
        identity_path = root / 'trainer-experiment.json'
        if start and (not identity_path.exists() or json.loads(identity_path.read_text()) != identity):
            raise ValueError('Effective Trainer implementation differs from the saved experiment')
        durable_json(identity_path, identity)
        journal('READY')
        trainer.train()
    except _CleanStop:
        result = clean_stop()
        record_update()
        return result
    finally:
        # Initialization failure must not advance the caller's restored RNG.
        if completed == start and not pending:
            restore_random(initial_random)
    if completed != len(order):
        raise ValueError('Trainer experiment ended before the requested update boundary')
    journal('TRAINED')
    return dict(status='TRAINED', completed=completed, history=history)


def worker(request):
    """Patch only this isolated worker. The ordinary training command stays unchanged."""
    from proxybench.training import runtime, trajectory
    if request['config'].get('training_loop') != EXPERIMENT:
        raise ValueError('Trainer experiment configuration marker is missing')
    original_updates, original_publish = trajectory.train_updates, trajectory.publish_state

    def candidate(*args, **kwargs):
        return train_updates(*args, **kwargs, max_updates=request['max_updates'])

    candidate._proxybench_training_loop = EXPERIMENT

    def publish(*args, **kwargs):
        diagnostics = dict(kwargs.pop('diagnostics', None) or {})
        identity_path = Path(request['run_dir']) / 'trainer-experiment.json'
        identity = json.loads(identity_path.read_text()) if identity_path.is_file() else trainer_identity()
        diagnostics['trainer_experiment'] = dict(scheduler=SCHEDULER, trainer=identity)
        return original_publish(*args, **kwargs, diagnostics=diagnostics)

    trajectory.train_updates, trajectory.publish_state = candidate, publish
    try:
        return runtime.train(request['dataset'], request['run_dir'], request['config'], request['resume'],
                             training_loop=EXPERIMENT)
    finally:
        trajectory.train_updates, trajectory.publish_state = original_updates, original_publish


def launch(operation, run_dir, config, *, max_updates, seconds, resource_ceiling=None, **arguments):
    """Retain supervisor process ownership, resource charges, and cooperative stop files."""
    from proxybench.execution.live import supervise
    from proxybench.execution.resources import ledger_entries, reconcile_captures
    if operation != 'train':
        raise ValueError('Trainer experiment supports training only')
    if config.get('training_loop') != EXPERIMENT:
        raise ValueError('Trainer experiment configuration marker is missing')
    root = Path(run_dir)
    reconcile_captures(root)
    request = root / f'trainer-request-{uuid.uuid4().hex}.json'
    durable_json(request, dict(config=config, run_dir=str(root), max_updates=max_updates, **arguments))
    used = sum(row['elapsed_seconds'] for row in ledger_entries(root / 'resources.jsonl'))
    limits = dict(config['limits'])
    limits['total_seconds'] = min(limits['total_seconds'], used + seconds)
    limits['phase_seconds'] = min(limits['phase_seconds'], used + seconds)
    if resource_ceiling is not None:
        limits['total_seconds'] = min(limits['total_seconds'], resource_ceiling)
    environment = dict(UNSLOTH_COMPILE_LOCATION=str((root/'cache'/'unsloth').resolve()),
        TRITON_CACHE_DIR=str((root/'cache'/'triton').resolve()), TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1', WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1')
    return supervise([sys.executable, '-m', 'proxybench.training.trainer_experiment', '_work', str(request)],
        root / f'execution-{uuid.uuid4().hex[:8]}', limits,
        ledger=root / 'resources.jsonl', phase='train', phase_used=used, environment=environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='operation', required=True)
    for name in ('start', 'resume'):
        command = subs.add_parser(name)
        command.add_argument('--run-dir', required=True)
        command.add_argument('--max-updates', type=int, required=True)
        command.add_argument('--seconds', type=int, required=True)
        if name == 'start':
            command.add_argument('--config', default='configs/training.json')
            command.add_argument('--dataset', default='data/training-dataset')
    command = subs.add_parser('_work')
    command.add_argument('request')
    subs.add_parser('inspect-cpu')
    args = parser.parse_args(argv)
    if args.operation == 'inspect-cpu':
        import torch
        if torch.cuda.is_initialized():
            raise ValueError('CPU inspection requires an uninitialized CUDA runtime')
        print(json.dumps(trainer_identity(), indent=2))
        return
    if args.operation == '_work':
        worker(json.loads(Path(args.request).read_text()))
        return
    if not 1 <= args.max_updates <= 660 or not 1 <= args.seconds <= 14400:
        parser.error('Use 1 through 660 updates and 1 through 14400 seconds')
    from proxybench.training import runtime
    root = runtime.require_plain_path(args.run_dir)
    if root.is_relative_to(Path.cwd().resolve()):
        raise ValueError('Use a new external folder for the Trainer experiment')
    if args.operation == 'start' and root.exists():
        raise ValueError('Use a new external folder for the Trainer experiment')
    if args.operation == 'resume' and not (root / 'trainer-experiment.json').is_file():
        raise ValueError('This folder is not a Trainer experiment')
    original_launch, original_load_config = runtime.launch, runtime.load_config
    runtime.launch = lambda *a, **kw: launch(*a, **kw, max_updates=args.max_updates, seconds=args.seconds)
    runtime.load_config = lambda path: dict(original_load_config(path), training_loop=EXPERIMENT)
    try:
        result = (runtime.cli(args, training_loop=EXPERIMENT) if args.operation == 'start'
                  else runtime.resume_training(root, training_loop=EXPERIMENT))
        print(json.dumps(result, indent=2))
    finally:
        runtime.launch, runtime.load_config = original_launch, original_load_config


if __name__ == '__main__':
    main()
