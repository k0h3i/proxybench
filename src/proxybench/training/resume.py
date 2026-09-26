"""Read-only training eligibility shared by commands and bounded workers."""

from pathlib import Path

from proxybench.runstate import binding
from proxybench.training.adapters import digest
from proxybench.training.checkpoints import plain_path, read_object, require_hash
from proxybench.training.trajectory import require_clean_stop, sample_order


def training_preflight(dataset, run_dir, config, *, resume=False, run_state=None):
    """Reject predictable failures without importing model packages or writing files."""
    from proxybench.training.dataset import read_release

    output = plain_path(run_dir, 'Training run path')
    if not isinstance(config, dict):
        raise ValueError('Training configuration must be an object')
    if (config.get('batch_size') != 1 or config.get('accumulation') != 1 or config.get('max_grad_norm') != 1.0):
        raise ValueError('This worker requires batch size 1, accumulation 1, and gradient norm 1')
    if type(config.get('epochs')) is not int or config['epochs'] <= 0 or type(config.get('seed')) is not int:
        raise ValueError('Training epochs and seed must be integers, with positive epochs')
    journal_path = output / 'training-journal.json'
    if not resume and any((output / name).exists() or (output / name).is_symlink() for name in
                          ('training-inputs.json', 'training-journal.json', 'training-result.json', 'checkpoints', 'adapter')):
        raise ValueError('Training exists. Use resume after a clean stop or choose a new run folder.')
    # Status comes first: ordinary journal records do not have a checkpoint key.
    previous = read_object(journal_path, 'Training journal') if resume else None
    if resume and previous.get('status') != 'CLEAN_STOP':
        raise ValueError('Resume requires a CLEAN_STOP journal. Use a new run folder for other states.')
    rows, manifest = read_release(dataset)
    if not rows['training']:
        raise ValueError('Training split is empty')
    order = sample_order(len(rows['training']), config['epochs'], config['seed'])
    identity = dict(dataset=binding(manifest), recipe=binding(config), order=binding(order))
    prompt_record = manifest.get('system_prompt')
    if not isinstance(prompt_record, dict):
        raise ValueError('Dataset prompt identity is missing')
    require_hash(prompt_record.get('sha256'), 'Dataset prompt hash')
    try:
        prompt_hash = digest(plain_path(config.get('system_prompt'), 'Training prompt path'))
    except OSError as exc:
        raise ValueError('Training prompt is missing or unreadable') from exc
    if prompt_hash != prompt_record['sha256']:
        raise ValueError('Training prompt differs from the accepted dataset')
    if run_state is None and (output / 'run.json').exists():
        run_state = read_object(output / 'run.json', 'Training run record')
    if run_state is not None:
        saved_identity = run_state.get('identity') if isinstance(run_state, dict) else None
        if (not isinstance(saved_identity, dict) or run_state.get('operation') != 'train'
                or saved_identity.get('dataset') != identity['dataset']
                or saved_identity.get('recipe') != identity['recipe']
                or saved_identity.get('prompt') != prompt_hash
                or run_state.get('configuration') != config):
            raise ValueError('Training run dataset, recipe, or prompt identity differs')
    inputs = dict(identity=identity, order=order)
    completed, checkpoint = 0, None
    if resume:
        saved_inputs = read_object(output / 'training-inputs.json', 'Training inputs')
        if (not isinstance(saved_inputs.get('order'), list)
                or any(type(index) is not int for index in saved_inputs['order'])
                or saved_inputs != inputs):
            raise ValueError('Saved training inputs or explicit sample order differ')
        completed = require_clean_stop(previous, previous.get('checkpoint'), identity,
                                       total=len(order), hash_files=False)
        checkpoint = plain_path(previous['checkpoint'])
        if not checkpoint.is_relative_to(output / 'checkpoints'):
            raise ValueError('Clean-stop checkpoint must be inside this run\'s checkpoints folder')
        # Adapter and final publication have no ordinary clean-stop recovery path.
        if (output / 'adapter').exists():
            raise ValueError('A final adapter already exists. Ordinary training resume cannot publish it again.')
    return dict(rows=rows, identity=identity, inputs=inputs, order=order,
                completed=completed, checkpoint=checkpoint)
