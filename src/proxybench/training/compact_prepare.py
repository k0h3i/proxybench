"""Prepare compact development previews without admitting data or running a model."""

import argparse
import hashlib
import json
from pathlib import Path

from proxybench.execution.runner import write_json
from proxybench.training.compact import (
    VERSION, compact_source, decode_record, decode_response, dumps, encode_response,
)
from proxybench.training.measure import prompt_for_bundle
from proxybench.training.sequences import sequence


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(schedule, modern, tokenizer, contract, old_contract, output, *, historical_lengths, root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    inputs = []
    for item in schedule:
        bundle_path, reference_path = root / item['input_path'], root / item['reference_path']
        bundle, reference_raw = bundle_path.read_bytes(), reference_path.read_bytes()
        if digest(bundle) != item['input_sha256'] or digest(reference_raw) != item['reference_sha256']:
            raise ValueError('Frozen historical input or reference hash differs')
        reference = json.loads(reference_raw)
        expected = historical_lengths[item['input_id']]
        # Historical lengths are from a frozen prompt, not this new serialization.
        if expected.get('input_id') != item['input_id'] or not isinstance(expected.get('combined_tokens'), int):
            raise ValueError('Historical length identity is invalid')
        response = dict(schema_version='benchmark-v1', input_id=item['input_id'], status='COMPLETE',
                        records=[reference['record']], failure=None)
        inputs.append((item['input_id'], bundle, response, 'historical-accepted-development',
                       bundle_path, reference_path, historical_lengths[item['input_id']]['combined_tokens']))
    for path in sorted(Path(modern).glob('*/response.json')):
        bundle_path = path.parent / 'bundle.json'
        bundle, response = bundle_path.read_bytes(), json.loads(path.read_text())
        previous = sequence(tokenizer, prompt_for_bundle(old_contract, bundle.decode()),
                            dumps(response), context_cap=262144)['combined_tokens']
        inputs.append((response['input_id'], bundle, response, 'modern-provisional-development',
                       bundle_path, path, previous))
    for input_id, bundle, response, group, bundle_path, reference_path, previous in inputs:
        target = encode_response(response, bundle)
        # Exact reconstruction uses the original administrative ID for this audit only.
        # The operational decoder receives no reference and creates its own record ID.
        for original, compact in zip(response['records'], target['records']):
            if decode_record(dumps(compact), bundle, record_id=original['record_id']) != original:
                raise ValueError('Record conversion loses information: ' + input_id)
        expanded = decode_response(dumps(target), bundle)
        source = compact_source(bundle)
        prompt = contract + '\nSOURCE\n' + dumps(source)
        item = sequence(tokenizer, prompt, dumps(target), context_cap=262144)
        directory = output / input_id
        directory.mkdir(exist_ok=False)
        (directory / 'prompt.txt').write_text(prompt)
        (directory / 'target.json').write_text(dumps(target) + '\n')
        write_json(directory / 'source.json', source)
        write_json(directory / 'expanded.json', expanded)
        write_json(directory / 'binding.json', {
            'format': VERSION, 'input_id': input_id, 'group': group,
            'bundle_path': str(bundle_path), 'bundle_sha256': digest(bundle),
            'reference_path': str(reference_path), 'reference_sha256': digest(reference_path.read_bytes()),
            'prompt_sha256': digest(prompt.encode()), 'target_sha256': digest(dumps(target).encode()),
            'contract_sha256': digest(contract.encode()), 'training_admitted': False,
            'format_review': 'PENDING', 'record_roundtrip_exact_excluding_generated_administrative_id': True,
        })
        rows.append({'input_id': input_id, 'group': group, 'previous_combined_tokens': previous,
                     **{k: item[k] for k in ('input_tokens', 'response_tokens', 'combined_tokens')},
                     'reduction_percent': round(100 * (1 - item['combined_tokens'] / previous), 2),
                     'fits_12288': item['combined_tokens'] <= 12288,
                     'fits_8192': item['combined_tokens'] <= 8192,
                     'prompt_supervised_tokens': sum(x != -100 for x in item['labels'][:item['input_tokens']]),
                     'response_supervised_tokens': sum(x != -100 for x in item['labels'][item['input_tokens']:])})
    report = {'format': VERSION, 'examples': len(rows), 'training_admitted': 0,
              'purpose': 'representation-size-and-roundtrip-audit, not model accuracy',
              'tokenizer': getattr(tokenizer, 'name_or_path', 'caller-supplied tokenizer'),
              'comparison_limit': 'Historical prior lengths must use the same tokenizer as this run',
              'max_combined_tokens': max(x['combined_tokens'] for x in rows),
              'above_12288': sum(not x['fits_12288'] for x in rows),
              'above_8192': sum(not x['fits_8192'] for x in rows), 'rows': rows}
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--modern', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--historical-lengths', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--old-contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    lengths = {x['input_id']: x for x in json.loads(args.historical_lengths.read_text())}
    report = prepare(json.loads(args.schedule.read_text()), args.modern, tokenizer, args.contract.read_text(),
                     args.old_contract.read_text(), args.output, historical_lengths=lengths, root=Path.cwd())
    write_json(args.output / 'tokenizer-provenance.json', {
        'path': str(args.tokenizer),
        'files': {str(p.name): digest(p.read_bytes()) for p in args.tokenizer.iterdir()
                  if p.is_file() and (p.name.startswith('tokenizer') or p.name.startswith('chat_template'))},
    })
    print(dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
