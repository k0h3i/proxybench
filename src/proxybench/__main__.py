"""Commands for source preparation, training, inference, and evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

from proxybench.runstate import Run, file_hash, binding, atomic_json


def evaluation_export(run):
    """Recover completed conversion before reserving another bounded attempt."""
    from proxybench.execution.resources import reconcile_captures
    from proxybench.training.runtime import export_complete, export_model
    root = run.path / 'exports'
    config = run.state['configuration']
    used = reconcile_captures(root)
    pending = run.state.get('pending_charge')
    if 'export_resource_base_seconds' not in run.state:
        reserved = pending['seconds'] if pending and pending['target'] == 'model-export' else 0
        run.state['export_resource_base_seconds'] = max(0, run.state['consumed_seconds'] - (reserved if reserved else used))
    if pending and pending['target'] == 'model-export':
        previous = run.state.get('model_export_reservation_ledger_seconds', 0)
        run.settle('model-export', max(0, used - previous))
    run.state['consumed_seconds'] = max(run.state['consumed_seconds'],
                                      run.state['export_resource_base_seconds'] + used)
    run.save()
    complete = export_complete(run.path / 'adapter', root, config)
    if complete is not None:
        return complete
    remaining = run.state['resource_limit_seconds'] - run.state['consumed_seconds']
    if remaining <= 0:
        raise ValueError('The cumulative run time limit is exhausted before model export')
    allowance = min(config.get('export_seconds', 1800) * 2, remaining)
    run.state['model_export_reservation_ledger_seconds'] = used
    run.reserve('model-export', allowance)
    try:
        return export_model(run.path / 'adapter', root, config, remaining_seconds=allowance)
    finally:
        # An active owner prevents settlement. The reservation stays charged.
        measured = reconcile_captures(root)
        run.settle('model-export', max(0, measured - used))
        run.state['consumed_seconds'] = max(run.state['consumed_seconds'],
                                          run.state['export_resource_base_seconds'] + measured)
        run.save()


def evaluation_command(args):
    from proxybench.evaluation.workflow import (create_run, prepare_inputs, load_inputs,
                                               load_answers, generate, report, import_review, runtime_binding)
    path = Path(args.run_dir)
    if path.exists() and (path / 'run.json').exists():
        state = json.loads((path / 'run.json').read_text())
        if args.command == 'resume' and state.get('operation') == 'train':
            from proxybench.training.runtime import resume_training
            return resume_training(path)
        if args.command == 'resume' and state.get('operation') in {'infer', 'export'}:
            from proxybench.training.runtime import resume_runtime
            return resume_runtime(path)
        if args.command == 'evaluate' and state.get('operation') == 'train':
            if getattr(args, 'report_only', False):
                raise ValueError('This training run has no saved evaluation')
            from proxybench.extraction.runtime import load_config
            with Run(path) as run:
                config_path = args.config or run.state['configuration'].get('inference_config')
                if not config_path:
                    raise ValueError('Supply --config with an inference configuration for this training run')
                config = load_config(config_path)
                model = evaluation_export(run)
                config['model'] = str(model.resolve())
                dataset = run.state['dataset']
                config['dataset'] = dataset
                inputs = prepare_inputs(dataset, model, config, project_root=args.project_root)
                atomic_json(path / 'evaluation' / 'inputs.json', inputs)
                run.state['training_identity'] = run.state['identity']
                run.state['identity'] = inputs['identity']
                run.state['configuration'] = config
                run.state['inputs_sha256'] = binding(inputs)
                run.state['evaluation_resource_base_seconds'] = run.state['consumed_seconds']
                run.state['operation'] = 'evaluate'
                run.save()
    if args.command == 'evaluate' and args.model:
        if not args.dataset or not args.config:
            raise ValueError('An existing model requires --dataset and --config')
        from proxybench.extraction.runtime import load_config
        config = load_config(args.config)
        config['model'] = str(Path(args.model).resolve())
        config['dataset'] = str(Path(args.dataset).resolve())
        inputs = prepare_inputs(args.dataset, args.model, config, project_root=args.project_root)
        create_run(path, inputs, config)
    with Run(path) as run:
        inputs = load_inputs(run)
        answers = load_answers(run, inputs)
        if args.command == 'review-import':
            import_review(run, inputs, answers, args.decisions)
        elif not getattr(args, 'report_only', False):
            preliminary = report(run, inputs, answers)
            if preliminary['status'] == 'INVALID_REFERENCES':
                return preliminary
            # Validate mutable runtime files only when generation remains necessary.
            missing = any(case['id'] not in answers for case in inputs['cases'])
            if missing:
                config = run.state['configuration']
                if file_hash(config['model']) != inputs['identity']['model'] or runtime_binding(config) != inputs['identity']['runtime']:
                    raise ValueError('Model or runtime inputs changed; use a new run folder')
                from proxybench.extraction.runtime import supervised_generate_answers
                def one(messages, output, config):
                    return supervised_generate_answers([messages], output, config)[0]
                answers = generate(run, inputs, one)
        return report(run, inputs, answers)


def prepare_command(args):
    from proxybench.annotation.preparation import prepare_review
    selections = json.loads(Path(args.selections).read_text())
    return prepare_review(args.project_root, selections, args.output, prompt_path=args.prompt,
                          assignments=json.loads(Path(args.assignments).read_text()), draft_path=args.draft,
                          label_contract_path=args.label_contract)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m proxybench')
    commands = parser.add_subparsers(dest='command', required=True)
    from proxybench.training.runtime import add_cli
    add_cli(commands)
    promote = commands.add_parser('promote', help='Retain one selected adapter and GGUF')
    promote.add_argument('--run-dir', required=True)
    promote.add_argument('--models-dir', default='artifacts/models')
    promote.add_argument('--name', required=True)
    def promote_command(args):
        from proxybench.training.promotion import promote
        return promote(args.run_dir, args.models_dir, args.name)
    promote.set_defaults(handler=promote_command)
    source = commands.add_parser('import-source', help='Import a complete original filing')
    source.add_argument('--input', required=True)
    source.add_argument('--project-root', default='.')
    source.add_argument('--url', required=True)
    source.add_argument('--accession', required=True)
    source.add_argument('--filename', required=True)
    source.add_argument('--retrieval-date')
    source.add_argument('--complete', action='store_true', help='Attest that the source is complete')
    def import_command(args):
        from proxybench.sources.inventory import import_source
        return import_source(args.input, args.project_root, sec_url=args.url,
                             accession=args.accession, friendly_filename=args.filename,
                             retrieval_date=args.retrieval_date, complete=args.complete)
    source.set_defaults(handler=import_command)
    fetch = commands.add_parser('fetch-source', help='Download a bounded SEC source for inspection')
    fetch.add_argument('--url', required=True)
    fetch.add_argument('--output', required=True)
    fetch.add_argument('--ledger', required=True)
    fetch.add_argument('--kind', choices=['filing', 'index'], default='filing')
    def fetch_command(args):
        from proxybench.sources.sec import fetch
        identity = os.environ.get('PROXYBENCH_SEC_IDENTITY', '')
        if not identity.strip():
            raise ValueError('Set PROXYBENCH_SEC_IDENTITY in private runtime configuration')
        event = fetch(args.url, args.output, args.ledger, identity, kind=args.kind)
        return dict(status='COMPLETE', output=event['output'], sha256=event['sha256'])
    fetch.set_defaults(handler=fetch_command)
    evaluate = commands.add_parser('evaluate', help='Evaluate one model or regenerate its report')
    evaluate.add_argument('--run-dir', required=True)
    evaluate.add_argument('--model')
    evaluate.add_argument('--dataset')
    evaluate.add_argument('--config')
    evaluate.add_argument('--project-root')
    evaluate.add_argument('--report-only', action='store_true')
    evaluate.set_defaults(handler=evaluation_command)
    resume = commands.add_parser('resume', help='Continue an interrupted run')
    resume.add_argument('--run-dir', required=True)
    resume.set_defaults(handler=evaluation_command)
    review = commands.add_parser('review-import', help='Accept decisions for exact saved answers')
    review.add_argument('--run-dir', required=True)
    review.add_argument('--decisions', required=True)
    review.set_defaults(handler=evaluation_command)
    status = commands.add_parser('status', help='Read run progress')
    status.add_argument('--run-dir', required=True)
    def status_command(args):
        with Run(args.run_dir) as run:
            return run.state
    status.set_defaults(handler=status_command)
    prepare = commands.add_parser('prepare', help='Prepare marked source fragments for review')
    prepare.add_argument('--project-root', default='.')
    prepare.add_argument('--selections', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--prompt', default='configs/model-system-prompt.txt')
    prepare.add_argument('--assignments', required=True)
    prepare.add_argument('--draft')
    prepare.add_argument('--label-contract', default='docs/label-contract.md')
    prepare.set_defaults(handler=prepare_command)
    accept = commands.add_parser('accept', help='Export explicitly accepted labels')
    for name in ('review-dir', 'review', 'approval', 'output', 'assignments'):
        accept.add_argument('--' + name, required=True)
    accept.add_argument('--project-root', default='.')
    accept.add_argument('--prompt', default='configs/model-system-prompt.txt')
    accept.add_argument('--label-contract', default='docs/label-contract.md')
    def accept_command(args):
        from proxybench.annotation.preparation import accept_review
        return accept_review(args.project_root, args.review_dir, args.review, args.approval, args.output,
                             assignments=json.loads(Path(args.assignments).read_text()),
                             prompt_path=args.prompt, label_contract_path=args.label_contract)
    accept.set_defaults(handler=accept_command)
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        status = result.get('status') if isinstance(result, dict) else None
        if args.command == 'status':
            return 0
        return 2 if status == 'PENDING_REVIEW' else 1 if status and status not in {'COMPLETE', 'EXITED', 'TRAINED', 'READY', 'ACCEPTED'} else 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
