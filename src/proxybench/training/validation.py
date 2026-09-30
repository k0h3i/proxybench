"""Bounded, user-launched load tests for the retained model formats."""
import json
from pathlib import Path

from proxybench.execution.resources import durable_json
from proxybench.training.runtime import load_base, attach_adapter, phase, base_model_path, require_plain_path
from proxybench.training.adapters import require_same_adapter, digest


def require_project_environment(training, inference):
    """Reject disposable dependencies before either model worker starts."""
    import os
    import shlex
    import shutil
    import sys

    root = Path.cwd().resolve()
    artifacts = root / 'artifacts'

    def require_independent(value):
        if value and Path(value).expanduser().resolve().is_relative_to(artifacts):
            raise ValueError('Model validation cannot use old artifact dependencies')

    for value in (sys.prefix, sys.base_prefix, sys.executable):
        require_independent(value)
    if Path(sys.prefix).resolve() != (root / '.venv').resolve():
        raise ValueError('Model validation requires the project .venv Python environment')
    for value in sys.path:
        require_independent(value)
    for name in ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH',
                 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH'):
        for value in os.environ.get(name, '').split(os.pathsep):
            require_independent(value)
    for name in ('CC', 'CXX'):
        for value in shlex.split(os.environ.get(name, '')):
            require_independent(value)
            resolved = shutil.which(value)
            if resolved:
                require_independent(resolved)
    for name in ('PROXYBENCH_RUNTIME', 'PROXYBENCH_CUDA_LIB',
                 'PROXYBENCH_CONVERTER_SOURCE', 'CUDA_HOME', 'CUDA_PATH'):
        require_independent(os.environ.get(name, ''))
    if training.get('base_path'):
        base_model_path(training)
    if training.get('base_manifest'):
        require_independent(require_plain_path(training['base_manifest']))
    if training.get('converter_source'):
        require_plain_path(training['converter_source'])
    for value in (training.get('converter_source'),
                  inference.get('server'), inference.get('runtime_manifest')):
        require_independent(value)
    for value in inference.get('library_path', '').split(os.pathsep):
        require_independent(value)
    for value in inference.get('external_libraries', {}):
        require_independent(value)


def synthetic_fragments(directory):
    from proxybench.annotation.historical import prepare_historical
    import hashlib
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    sources = (
        ('synthetic.txt', b'Example Fund | Example Company | Elect director | FOR | Management: FOR', 'text'),
        ('synthetic.html', b'<tr><td>Example Fund</td><td>Example Company</td><td>Elect director</td><td>FOR</td><td>Management: FOR</td></tr>', 'row'),
    )
    fragments = []
    for name, raw, kind in sources:
        (root/name).write_bytes(raw)
        selection = dict(packet_id=name, accession='synthetic-validation', source_path=name,
            source_sha256=hashlib.sha256(raw).hexdigest(), encoding='utf-8', split='development',
            group_id='synthetic', spans=[[0, len(raw), kind]], target=[0, len(raw)],
            boundary_review='Synthetic fixture with one separately voted subject', reviewer='fixture-author')
        packet = prepare_historical(root, selection)
        fragments.append(packet['model_input'])
        (root/(name+'.fragment.txt')).write_text(packet['model_input'])
    return fragments


def validate_adapter(adapter, output, config):
    from proxybench.training.runtime import require_adapter_base
    require_adapter_base(adapter, config)
    from unsloth import FastLanguageModel
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoTokenizer
    from proxybench.extraction.measured_generation import generate
    from proxybench.extraction.runtime import prompt_tokens, source_messages
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    phase('loading')
    model, base_tokenizer = load_base(config)
    model = attach_adapter(model, config, output)
    saved = load_file(str(Path(adapter)/'adapter_model.safetensors'))
    set_peft_model_state_dict(model, saved)
    require_same_adapter(saved, get_peft_model_state_dict(model, save_embedding_layers=False))
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    FastLanguageModel.for_inference(model)
    model.eval()
    phase('inference')
    for index, text in enumerate(synthetic_fragments(output/'sources')):
        messages = source_messages(text, config)
        tokens, _ = prompt_tokens(tokenizer, messages, config)
        if prompt_tokens(base_tokenizer, messages, config)[0] != tokens:
            raise ValueError('Adapter and pinned base tokenizers differ')
        answer = generate(model, tokenizer, tokens, output/f'adapter-answer-{index}.json',
                          index=index, maximum=config['response_tokens'], deadline=240)
        if answer['status'] not in {'COMPLETE', 'LENGTH_STOP'}:
            raise ValueError('Adapter request did not complete')
    durable_json(output/'adapter-load.json', dict(status='COMPLETE', exact_tensors=True,
                 adapter_sha256=digest(Path(adapter)/'adapter_model.safetensors')))


def validate_runtime(args):
    from proxybench.extraction.runtime import load_config, source_messages, supervised_generate_answers
    from proxybench.training.runtime import launch, select_base, require_adapter_base
    training = load_config(args.training_config)
    if getattr(args, 'model', None):
        training = select_base(training, args.model)
    inference = load_config(args.config)
    require_project_environment(training, inference)
    require_adapter_base(args.adapter, training)
    require_adapter_base(inference['tokenizer'], training)
    from proxybench.training.checkpoints import read_object
    model_info = Path(inference['model']).parent / 'model-info.json'
    if model_info.exists():
        metadata = read_object(model_info, 'GGUF model metadata')
        if (metadata.get('base_model') != training['model_id']
                or metadata.get('base_revision') != training['model_revision']):
            raise ValueError('GGUF model metadata differs from the selected pinned base')
    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=False)
    inference['limits'] = {**inference['limits'], 'phase_seconds': 600, 'total_seconds': 600}
    # No training updates, merge, or conversion occur in this validation.
    training['limits'] = {**training['limits'], 'phase_seconds': 900, 'total_seconds': 900}
    status = launch('validate-adapter', output/'adapter', training, adapter=args.adapter)
    if status != 'EXITED':
        raise RuntimeError(f'Adapter load test stopped: {status}')
    answers = supervised_generate_answers([source_messages(text, inference) for text in synthetic_fragments(output/'sources')],
                                          output/'gguf', inference)
    if any(answer['status'] not in {'COMPLETE', 'LENGTH_STOP'} for answer in answers):
        raise ValueError('GGUF request did not complete')
    result = dict(status='COMPLETE', adapter_load=True, gguf_load=True, synthetic_requests=4,
                  accuracy_assessed=False)
    durable_json(output/'validation.json', result)
    return result
