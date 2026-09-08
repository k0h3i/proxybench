"""Measure a short synthetic generation with the existing local Qwen runtime."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('model_path', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repetitions', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.configuration.read_text())
    import torch
    from transformers import set_seed
    from transformers.generation.streamers import BaseStreamer
    from proxybench.training.gpu_probe import load_model, phase
    from proxybench.execution.runner import write_json

    set_seed(config['seed'])
    write_json(args.output / 'configuration.json', config)
    model, tokenizer = load_model(config, args.model_path, args.output)
    model.eval()
    prompt = 'Count from 1 to 100, separated by commas. Output only the numbers.'
    (args.output / 'prompt.txt').write_text(prompt + '\n')
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors='pt')
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    for repetition in range(args.repetitions):
        sample_output = args.output if args.repetitions == 1 else args.output / f'sample-{repetition + 1}'
        if args.repetitions > 1:
            sample_output.mkdir()
        times, ids = [], []
        with (sample_output / 'tokens.jsonl').open('x') as log:
            class TimedStreamer(BaseStreamer):
                prompt_pending = True

                def put(self, value):
                    if self.prompt_pending:
                        self.prompt_pending = False
                        return
                    token_ids = value.reshape(-1).tolist()
                    elapsed = time.perf_counter() - started
                    ids.extend(token_ids)
                    times.extend([elapsed] * len(token_ids))
                    log.write(json.dumps({'seconds': elapsed, 'token_ids': token_ids}) + '\n')
                    log.flush()

                def end(self):
                    pass

            phase('short_speed_test')
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                model.generate(**inputs, do_sample=False, max_new_tokens=128,
                               eos_token_id=tokenizer.eos_token_id,
                               pad_token_id=tokenizer.pad_token_id,
                               use_cache=True, streamer=TimedStreamer())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        (sample_output / 'raw.txt').write_text(tokenizer.decode(ids, skip_special_tokens=True))
        result = {
            'kind': 'short-synthetic-speed-test', 'device': torch.cuda.get_device_name(0),
            'input_tokens': inputs['input_ids'].shape[1], 'output_tokens': len(ids),
            'max_new_tokens': 128, 'generation_seconds': elapsed,
            'time_to_first_token_seconds': times[0] if times else None,
            'generation_tokens_per_second': len(ids) / elapsed,
            'decode_tokens_per_second': (len(ids) - 1) / (times[-1] - times[0]) if len(ids) > 1 else None,
            'ended_with_eos': bool(ids and ids[-1] == tokenizer.eos_token_id),
            'loading_excluded': True, 'warmup_runs': repetition,
        }
        write_json(sample_output / 'result.json', result)
        print(json.dumps(result), flush=True)



if __name__ == '__main__':
    main()
