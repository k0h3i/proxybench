"""Capture greedy Python generation with fresh state and durable token events."""
import json
import os
from pathlib import Path
import time

from proxybench.execution.resources import durable_json
from proxybench.training.labels import read_json, validate


def timing_summary(events, duration):
    times = [event['seconds'] for event in events]
    return dict(request_seconds=duration, output_tokens=len(times),
                first_token_seconds=times[0] if times else None,
                last_token_seconds=times[-1] if times else None,
                request_tokens_per_second=len(times) / duration if duration else None,
                decode_tokens_per_second=(len(times) - 1) / (times[-1] - times[0])
                if len(times) > 1 and times[-1] > times[0] else None,
                early_64_seconds=times[63] - times[0] if len(times) >= 64 else None,
                late_64_seconds=times[-1] - times[-64] if len(times) >= 64 else None)


def completion_status(tokens, eos, *, forced, maximum, elapsed, deadline, stream_ended):
    if elapsed > deadline:
        return 'TIMEOUT'
    if not stream_ended:
        return 'CAPTURE_INCOMPLETE'
    if forced:
        return 'PROBE_COMPLETE' if len(tokens) == maximum else 'PROBE_INCOMPLETE'
    return 'COMPLETE' if tokens and tokens[-1] == eos else 'LENGTH_STOP'


def generate(model, tokenizer, prompt, path, *, index, maximum=1792, forced=False,
             deadline=240, expected=None, streaming=True):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    path = Path(path)
    if path.exists() or path.with_suffix('.tokens.jsonl').exists():
        raise FileExistsError(path)
    begin = time.monotonic()
    request = dict(index=index, status='STARTED', prompt_token_ids=prompt, max_new_tokens=maximum,
                   force_length=forced, do_sample=False, use_cache=True, past_key_values=None,
                   fresh_request_state=True, eos_token_id=tokenizer.eos_token_id,
                   pad_token_id=tokenizer.pad_token_id, deadline_seconds=deadline,
                   streaming=streaming, started_monotonic=begin, maximum_unsaved_tokens=1 if streaming else maximum)
    durable_json(path, request)
    deadline_file = os.environ.get('PROXYBENCH_REQUEST_FILE')
    if deadline_file:
        durable_json(deadline_file, dict(request_path=str(path.resolve()), deadline_monotonic=begin + deadline))
    events = []
    class Recorder:
        first = True
        ended = False
        def put(self, value):
            if self.first:
                self.first = False
                return
            ids = value.reshape(-1).tolist()
            for token in ids:
                event = dict(token_id=token, seconds=time.monotonic() - begin,
                             text=tokenizer.decode([token], skip_special_tokens=False))
                events.append(event)
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
        def end(self):
            self.ended = True
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() - begin >= deadline
    recorder = Recorder()
    with path.with_suffix('.tokens.jsonl').open('x') as stream:
        try:
            ids = torch.tensor([prompt], dtype=torch.long, device='cuda')
            with torch.inference_mode():
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=maximum, min_new_tokens=maximum if forced else 0,
                    do_sample=False, use_cache=True, past_key_values=None,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([Deadline()]),
                    **({'streamer': recorder} if streaming else {}))
            tokens = result[0, len(prompt):].tolist()
            text = tokenizer.decode(tokens[:-1] if tokens and tokens[-1] == tokenizer.eos_token_id else tokens,
                                    skip_special_tokens=False)
            error, parsed = None, None
            try:
                parsed = read_json(text)
                validate(parsed)
            except (ValueError, TypeError) as exc:
                error = str(exc)
            if streaming and tokens != [event['token_id'] for event in events]:
                raise ValueError('Captured tokens differ from returned tokens')
            elapsed = time.monotonic() - begin
            status = completion_status(tokens, tokenizer.eos_token_id, forced=forced, maximum=maximum,
                                       elapsed=elapsed, deadline=deadline, stream_ended=recorder.ended if streaming else True)
            answer = dict(**request)
            answer.update(status=status, token_ids=tokens, text=text,
                          raw=tokenizer.decode(tokens, skip_special_tokens=False),
                          format_error=error, format_valid=status == 'COMPLETE' and error is None,
                          exact_target=status == 'COMPLETE' and error is None and parsed == expected,
                          timing=timing_summary(events, elapsed), completed_monotonic=time.monotonic(),
                          exact_token_arrivals=streaming, terminal_stream_event=recorder.ended)
            if not streaming:
                answer['timing'].update(output_tokens=len(tokens), request_tokens_per_second=len(tokens)/elapsed)
            durable_json(path, answer)
            if deadline_file:
                Path(deadline_file).unlink(missing_ok=True)
            return answer
        except BaseException as exc:
            durable_json(path, {**request, 'status': 'FAILED', 'error': str(exc),
                               'token_ids': [event['token_id'] for event in events],
                               'timing': timing_summary(events, time.monotonic() - begin)})
            if deadline_file:
                Path(deadline_file).unlink(missing_ok=True)
            raise
