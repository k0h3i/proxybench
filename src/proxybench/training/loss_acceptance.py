"""CPU loss semantics and user-launched, bounded CUDA acceptance."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from proxybench.execution.resources import durable_json


def reference_loss(hidden, weight, labels):
    """Mean causal loss. Shift here only, never before the installed helper."""
    import torch.nn.functional as functional
    if hidden.shape[:-1] != labels.shape or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError('Loss input shapes differ')
    targets = labels[..., 1:]
    if not (targets != -100).any():
        raise ValueError('Loss requires a supervised next token')
    # Match the helper's input cast, then use FP32 arithmetic for the reference.
    logits = hidden.to(weight.dtype).float()[..., :-1, :] @ weight.float().T
    return functional.cross_entropy(logits.reshape(-1, weight.shape[0]),
                                    targets.reshape(-1), ignore_index=-100)


def loss_cases():
    """Token 9 represents termination. Ignored labels include right padding."""
    return {
        'first_response': [[-100, -100, 3, -100, -100]],
        'termination': [[-100, -100, -100, 9, -100]],
        'ignored_prompt': [[-100, -100, -100, 3, 9]],
        'right_padding': [[-100, -100, 3, 9, -100, -100]],
        'unequal_lengths': [[-100, -100, 3, 4, 9, -100],
                            [-100, 5, 9, -100, -100, -100]],
    }


def installed_helper_source():
    """Inspect installed source without importing GPU initialization code."""
    distribution = importlib.metadata.distribution('unsloth_zoo')
    path = Path(distribution.locate_file('unsloth_zoo/loss_utils.py'))
    raw = path.read_bytes()
    tree = ast.parse(raw)
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'fused_linear_cross_entropy')
    calls = [node for node in ast.walk(helper) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'linear_cross_entropy']
    if len(calls) != 1:
        raise ValueError('Installed fused helper has an unknown loss call')
    keywords = {item.arg: item.value for item in calls[0].keywords}
    if not isinstance(keywords.get('shift'), ast.Constant) or keywords['shift'].value is not True:
        raise ValueError('Installed fused helper does not apply the expected causal shift')
    if not isinstance(keywords.get('filter_eps'), ast.Name) or keywords['filter_eps'].id != 'accuracy_threshold':
        raise ValueError('Installed fused helper has an unknown gradient filter')
    return dict(version=distribution.version, sha256=hashlib.sha256(raw).hexdigest(),
                internal_shift=True, filter_argument='accuracy_threshold')


def error_metrics(actual, expected):
    import torch
    actual, expected = actual.detach().float(), expected.detach().float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('Loss comparison contains a nonfinite value')
    difference = actual - expected
    return dict(max_absolute=float(difference.abs().max()),
                relative_l2=float(difference.norm() / expected.norm().clamp_min(1e-30)))


def compare_loss_case(actual, expected, actual_inputs, reference_inputs, labels, tolerance, *, gradients):
    """Keep forward-only evidence distinct from supported backward comparisons."""
    import torch
    metrics = dict(loss=error_metrics(actual, expected))
    passed = metrics['loss']['max_absolute'] <= tolerance['loss_absolute']
    if not gradients:
        return dict(passed=passed, gradient_status='UNSUPPORTED_DTYPE', **metrics)
    actual_gradients = torch.autograd.grad(actual, actual_inputs)
    reference_gradients = torch.autograd.grad(expected, reference_inputs)
    for name, left, right in zip(('hidden_gradient', 'weight_gradient', 'adapter_a_gradient',
                                 'adapter_b_gradient'), actual_gradients, reference_gradients, strict=True):
        metrics[name] = error_metrics(left, right)
        passed = passed and metrics[name]['relative_l2'] <= tolerance['gradient_relative_l2']
    ignored = torch.ones_like(labels, dtype=torch.bool)
    ignored[..., :-1] = labels[..., 1:] == -100
    passed = passed and bool((actual_gradients[0][ignored] == 0).all())
    return dict(passed=passed, gradient_status='COMPARED', **metrics)


def run_cuda_loss(output):
    """Execute the installed helper, with no CPU substitute or manual shift."""
    import unsloth  # Import before Torch, as required by the installed stack.
    import torch
    from unsloth_zoo.loss_utils import fused_linear_cross_entropy
    del unsloth
    source = installed_helper_source()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Loss acceptance requires a CUDA GPU with BF16 support')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from proxybench.training.runtime import phase, stopping
    phase('compilation')
    report = dict(status='RUNNING', helper=source, device=torch.cuda.get_device_name(),
                  torch=torch.__version__, cases=[],
                  acceptance_scope='BF16 loss and gradients; FP32 forward loss only',
                  unsupported=[dict(dtype='float32', operation='backward',
                                    reason='Backwards requires embeddings to be bf16 or fp16')])
    # Fixed, predeclared budgets: kernel accumulation and BF16 rounding differ.
    # Filtering is tested separately, at the actual helper's default threshold.
    tolerances = {
        'float32': dict(loss_absolute=2e-5, gradient_relative_l2=2e-4),
        'bfloat16': dict(loss_absolute=5e-3, gradient_relative_l2=2e-2),
    }
    report['tolerances'] = tolerances
    for dtype_name, dtype in [('float32', torch.float32), ('bfloat16', torch.bfloat16)]:
        for threshold in (None, 'auto'):
            for name, rows in loss_cases().items():
                if stopping():
                    raise ValueError('Loss acceptance stopped before completion')
                labels = torch.tensor(rows, device='cuda', dtype=torch.long)
                generator = torch.Generator(device='cpu').manual_seed(42)
                hidden = torch.randn((*labels.shape, 64), generator=generator).to('cuda', dtype)
                weight = (torch.randn((2048, 64), generator=generator) / 8).to('cuda', dtype)
                inputs = torch.randn((*labels.shape, 16), generator=generator).to('cuda', dtype)
                adapter_a = (torch.randn((16, 8), generator=generator) / 8).to('cuda', dtype).requires_grad_()
                adapter_b = (torch.randn((8, 64), generator=generator) / 8).to('cuda', dtype).requires_grad_()
                reference_a = adapter_a.detach().clone().requires_grad_()
                reference_b = adapter_b.detach().clone().requires_grad_()
                actual_hidden = hidden + inputs @ adapter_a @ adapter_b
                actual_weight = weight.detach().requires_grad_()
                reference_hidden = hidden.detach().clone() + inputs @ reference_a @ reference_b
                reference_weight = weight.detach().clone().requires_grad_()
                actual = fused_linear_cross_entropy(actual_hidden, actual_weight, labels,
                                                    accuracy_threshold=threshold)
                expected = reference_loss(reference_hidden, reference_weight, labels)
                metrics = compare_loss_case(actual, expected,
                    (actual_hidden, actual_weight, adapter_a, adapter_b),
                    (reference_hidden, reference_weight, reference_a, reference_b), labels,
                    tolerances[dtype_name], gradients=dtype == torch.bfloat16)
                report['cases'].append(dict(name=name, dtype=dtype_name, filter=threshold,
                                           effective_filter_eps=(torch.finfo(dtype).eps / 32
                                                                 if threshold == 'auto' else None),
                                           **metrics))
                durable_json(output, report)
    report['status'] = 'PASSED' if all(row['passed'] for row in report['cases']) else 'FAILED'
    durable_json(output, report)
    if report['status'] != 'PASSED':
        raise ValueError('Installed CUDA loss differs from the reference tolerance')
    return report


def continuation_worker(request):
    """Add an acceptance stop boundary around the unchanged production worker."""
    from proxybench.training import runtime
    from proxybench.training.merge import tensor_hash
    output = Path(request['run'])
    original_stop, original_attach = runtime.stopping, runtime.attach_adapter

    def stop():
        journal = output / 'training-journal.json'
        return original_stop() or (journal.exists()
                                   and json.loads(journal.read_text())['completed'] >= request['updates'])

    def attach(model, configuration, destination):
        model = original_attach(model, configuration, destination)
        identity = {name: dict(sha256=tensor_hash(parameter), dtype=str(parameter.dtype),
                               shape=list(parameter.shape))
                    for name, parameter in model.named_parameters() if parameter.requires_grad}
        path = output / ('resume-initial-weights.json' if request['resume'] else 'initial-weights.json')
        durable_json(path, identity)
        return model

    runtime.stopping, runtime.attach_adapter = stop, attach
    try:
        result = runtime.train(request['dataset'], output, request['configuration'], resume=request['resume'])
    finally:
        runtime.stopping, runtime.attach_adapter = original_stop, original_attach
    if result['status'] != 'CLEAN_STOP' or result['completed'] != request['updates']:
        raise ValueError('Continuation acceptance did not reach its required clean boundary')


def compare_continuation(root):
    """Compare trusted acceptance checkpoints, excluding wall-clock telemetry."""
    import torch
    from proxybench.training.trajectory import assert_same, require_clean_stop
    root = Path(root)
    states, identities, initial = [], [], []
    for arm in ('uninterrupted', 'resumed'):
        folder = root / arm
        inputs = json.loads((folder / 'training-inputs.json').read_text())
        journal = json.loads((folder / 'training-journal.json').read_text())
        checkpoint = Path(journal['checkpoint'])
        if not checkpoint.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Acceptance checkpoint escaped its new run directory')
        require_clean_stop(journal, checkpoint, inputs['identity'])
        states.append(torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False))
        identities.append(inputs)
        initial.append(json.loads((folder / 'initial-weights.json').read_text()))
    assert_same(identities[0], identities[1])
    assert_same(initial[0], initial[1])
    runtime_identities = [state['diagnostics']['runtime_identity'] for state in states]
    if any(identity is None for identity in runtime_identities):
        raise ValueError('Continuation acceptance requires recorded runtime identities in both checkpoints')
    from proxybench.training.measurements import compare_runtime_identity
    runtime_comparison = compare_runtime_identity(*runtime_identities)
    if runtime_comparison['status'] != 'MATCH':
        raise ValueError('Continuation runtime identities differ')
    fields = ('weights', 'optimizer', 'parameter_map', 'random', 'order', 'completed', 'next_position')
    for field in fields:
        assert_same(states[0][field], states[1][field])
    for state in states:
        if state['completed'] != 4:
            raise ValueError('Continuation acceptance requires four completed updates')
    # Time and allocation measurements are evidence, not trajectory state.
    history_fields = ('step', 'index', 'loss', 'response_tokens')
    histories = [[{key: row[key] for key in history_fields} for row in state['history']] for state in states]
    assert_same(histories[0], histories[1])
    report = dict(status='PASSED', updates=4, interruption_after=2, comparison='exact',
                  fields=list(fields), history_fields=list(history_fields), identity=identities[0]['identity'])
    report['runtime_comparison'] = runtime_comparison
    durable_json(root / 'continuation-report.json', report)
    return report


def launch_acceptance(root, configuration, *, continuation=False):
    """Use one resource ledger for all fresh acceptance subprocesses."""
    from proxybench.execution.live import supervise
    from proxybench.extraction.runtime import load_config
    root = Path(root).expanduser().resolve()
    if root.is_relative_to(Path.cwd().resolve()):
        raise ValueError('Acceptance requires a new directory outside the repository')
    configuration = load_config(configuration)
    root.mkdir(parents=True, exist_ok=False)
    limits = dict(configuration['limits'], phase_seconds=1800 if continuation else 600,
                  total_seconds=5400 if continuation else 600, automatic_stop_margin_seconds=60)
    durable_json(root / 'acceptance-plan.json', dict(configuration=configuration, limits=limits,
                                                  continuation=continuation, helper=installed_helper_source()))
    if continuation:
        requests = [('uninterrupted', 4, False), ('resumed', 2, False), ('resumed', 4, True)]
    else:
        requests = [('loss', 0, False)]
    for position, (arm, updates, resume) in enumerate(requests):
        request = root / f'request-{position}.json'
        durable_json(request, dict(operation='continuation' if continuation else 'loss',
                                  run=str(root / arm), report=str(root / 'loss-report.json'),
                                  dataset=configuration['dataset'], configuration=configuration,
                                  updates=updates, resume=resume))
        environment = dict(UNSLOTH_COMPILE_LOCATION=str(root / 'cache' / 'unsloth'),
                           TRITON_CACHE_DIR=str(root / 'cache' / 'triton'),
                           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                           WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1')
        status = supervise([sys.executable, '-m', 'proxybench.training.loss_acceptance', '_worker', str(request)],
                           root / f'execution-{position}', limits, ledger=root / 'resources.jsonl',
                           phase=f'acceptance-{position}', environment=environment)
        if status != 'EXITED':
            raise ValueError(f'Acceptance worker did not finish: {status}')
    if continuation:
        try:
            return compare_continuation(root)
        except (ValueError, KeyError, TypeError) as exc:
            durable_json(root / 'continuation-report.json', dict(status='FAILED', reason=str(exc)))
            raise
    return json.loads((root / 'loss-report.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('inspect', 'loss', 'continuation', '_worker'))
    parser.add_argument('path', nargs='?')
    parser.add_argument('--config', default='configs/training.json')
    args = parser.parse_args()
    if args.operation == 'inspect':
        print(json.dumps(installed_helper_source(), indent=2))
    elif args.path is None:
        parser.error('A new external directory is required')
    elif args.operation == '_worker':
        if not os.environ.get('PROXYBENCH_STOP_FILE'):
            raise ValueError('Acceptance workers require the bounded supervisor')
        request = json.loads(Path(args.path).read_text())
        if request['operation'] == 'loss':
            run_cuda_loss(request['report'])
        else:
            continuation_worker(request)
    else:
        print(json.dumps(launch_acceptance(args.path, args.config,
                                          continuation=args.operation == 'continuation'), indent=2))


if __name__ == '__main__':
    main()
