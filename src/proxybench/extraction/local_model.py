"""Text-only Qwen prompts and raw output capture; optional libraries load lazily."""

import hashlib
import json
from pathlib import Path
import time

from proxybench.extraction.bundles import read_bundle


def generate_one(model, tokenizer, prompt_path, input_path, output, config):
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = read_bundle(Path(input_path).read_bytes())
    prompt = Path(prompt_path).read_text()
    encoded = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                             add_generation_prompt=True, enable_thinking=False,
                                             return_dict=True, return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in encoded.items()}
    length = inputs['input_ids'].shape[1]
    if length >= config['context_tokens']:
        raise ValueError('Complete prompt exceeds the declared context limit')
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False,
                                   max_new_tokens=min(config['max_new_tokens'], config['context_tokens'] - length),
                                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                   use_cache=True)
    torch.cuda.synchronize()
    response_ids = generated[0, length:].tolist()
    raw = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    (output / 'raw.txt').write_text(raw)
    (output / 'token-ids.json').write_text(json.dumps(response_ids) + '\n')
    result = {'input_id': bundle['input_id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'input_tokens': length, 'output_tokens': len(response_ids), 'elapsed_seconds': time.monotonic() - start,
              'terminated': bool(response_ids and response_ids[-1] == tokenizer.eos_token_id),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    (output / 'capture.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
