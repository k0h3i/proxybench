"""Exact chat-template lengths and response-only loss masks without truncation."""


def sequence(tokenizer, prompt, response, *, context_cap):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
    complete = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}],
                                             tokenize=True, enable_thinking=False, return_dict=False)
    if complete[:len(prefix)] != prefix:
        raise ValueError("Chat template changes the prompt prefix at the response boundary")
    if len(complete) > context_cap:
        raise ValueError(f"Complete sequence needs {len(complete)} tokens, exceeding {context_cap}")
    labels = [-100] * len(prefix) + complete[len(prefix):]
    if not labels[len(prefix):] or tokenizer.eos_token_id not in labels[len(prefix):]:
        raise ValueError("Response has no supervised termination token")
    return {"input_ids": complete, "attention_mask": [1] * len(complete), "labels": labels,
            "input_tokens": len(prefix), "response_tokens": len(complete) - len(prefix),
            "combined_tokens": len(complete)}


def pad_sequence(item, length, pad_token_id):
    if len(item["input_ids"]) > length:
        raise ValueError("Padding cannot truncate a sequence")
    count = length - len(item["input_ids"])
    return {"input_ids": item["input_ids"] + [pad_token_id] * count,
            "attention_mask": item["attention_mask"] + [0] * count,
            "labels": item["labels"] + [-100] * count}
