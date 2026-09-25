"""Bounded, user-launched load tests for the retained model formats."""
import json
from pathlib import Path

from proxybench.execution.resources import durable_json
from proxybench.training.runtime import load_base, attach_adapter, phase
from proxybench.training.adapters import require_same_adapter, digest

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
    from proxybench.training.runtime import launch
    import os
    import sys
    artifacts = Path('artifacts').resolve()
    if Path(sys.prefix).resolve().is_relative_to(artifacts):
        raise ValueError('Model validation requires the external Python environment')
    for name in ('CC', 'CXX', 'LD_LIBRARY_PATH', 'PROXYBENCH_BASE_CACHE'):
        if str(artifacts/'environments') in os.environ.get(name, '') or str(artifacts/'runs') in os.environ.get(name, ''):
            raise ValueError('Model validation cannot use old experiment dependencies')
    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=False)
    training = load_config(args.training_config)
    inference = load_config(args.config)
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
