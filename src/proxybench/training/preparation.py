"""Prepare three-role training messages without truncation."""
from proxybench.training.sequences import sequence

def prepare_sequences(tokenizer, rows, config):
    result = {}
    for split in ('training', 'development'):
        result[split] = []
        for row in rows[split]:
            messages = row['messages']
            if len(messages) == 3 and [m['role'] for m in messages] == ['system', 'user', 'assistant']:
                system, prompt, answer = (m['content'] for m in messages)
            else:
                raise ValueError('Historical sequence has unsupported message roles')
            for value in (prompt, answer, *([system] if system is not None else [])):
                if any(marker in value for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')):
                    raise ValueError('Reserved template marker in source or answer')
            item = sequence(tokenizer, prompt, answer, context_cap=config['context_tokens'], system=system)
            suffix = item['input_ids'][item['input_tokens']:]
            eos = tokenizer.eos_token_id
            # Supervise through the end token, excluding template whitespace after it.
            if suffix.count(eos) != 1:
                raise ValueError('Expected one answer termination token')
            end = item['input_tokens'] + suffix.index(eos) + 1
            for key in ('input_ids', 'attention_mask', 'labels'):
                item[key] = item[key][:end]
            item['response_tokens'] = end-item['input_tokens']
            item['combined_tokens'] = end
            if tokenizer.decode(item['input_ids'][item['input_tokens']:-1], skip_special_tokens=False) != answer:
                raise ValueError('Answer loss mask changes the accepted answer')
            if item['input_tokens'] > config['input_tokens'] or item['response_tokens'] > config['response_tokens']:
                raise ValueError('Accepted sequence exceeds a reviewed limit')
            result[split].append(item)
    return result
