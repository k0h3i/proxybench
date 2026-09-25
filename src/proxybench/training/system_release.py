"""Export source-only historical messages without changing accepted labels."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile

from proxybench.training.labels import dumps, read_json, sha
from proxybench.training.sequences import sequence


SOURCE_PREFIX = '\n\nExtract only the marked target. Source cells follow in original order.\n\n'


def source_only(packet, policy):
    prefix = policy + SOURCE_PREFIX
    value = packet['model_input']
    if not value.startswith(prefix):
        raise ValueError('Packet does not contain the frozen source prelude')
    source = value[len(prefix):]
    if 'BEGIN MARKED TARGET' not in source or 'END MARKED TARGET' not in source:
        raise ValueError('Source-only user message lost its marked target')
    return source


def export_system_release(parent, output, prompt_path, tokenizer):
    """Rebind a validated two-role release to one frozen three-role prompt."""
    from proxybench.training.dataset import read_release

    parent, output, prompt_path = map(Path, (parent, output, prompt_path))
    rows, old = read_release(parent)
    if old['schema'] != 'historical-dataset-v1' or set(rows) != {'training', 'development'}:
        raise ValueError('Expected a complete historical training and development release')
    prompt_raw = prompt_path.read_bytes()
    prompt = prompt_raw.decode('utf-8')
    if not prompt.strip() or any(marker in prompt for marker in ('<|im_start|>', '<|im_end|>')):
        raise ValueError('System prompt is empty or contains a reserved template marker')
    policy_raw = (parent/'policy.md').read_bytes()
    policy = policy_raw.decode('utf-8')
    result = {}
    examples = []
    by_split = {split: iter(value) for split, value in rows.items()}
    for meta in old['examples']:
        split = meta['split']
        prior_row = next(by_split[split])
        source = source_only(meta['packet'], policy)
        answer = prior_row['messages'][-1]['content']
        row = dict(messages=[dict(role='system', content=prompt),
                             dict(role='user', content=source),
                             dict(role='assistant', content=answer)])
        item = sequence(tokenizer, source, answer, context_cap=5120, system=prompt)
        lengths = {key: item[key] for key in ('input_tokens', 'response_tokens', 'combined_tokens')}
        if lengths['input_tokens'] > 3328 or lengths['response_tokens'] > 1792:
            raise ValueError(f'System message sequence exceeds a reviewed length: {meta["packet"]["manifest"]["packet_id"]}')
        result.setdefault(split, []).append(row)
        updated = deepcopy(meta)
        updated['row_sha256'] = sha(dumps(row).encode())
        updated['lengths'] = lengths
        examples.append(updated)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+'.incomplete-', dir=output.parent))
    (stage/'policy.md').write_bytes(policy_raw)
    (stage/'system-prompt.txt').write_bytes(prompt_raw)
    hashes = {}
    for split, items in result.items():
        raw = ''.join(dumps(row)+'\n' for row in items).encode()
        (stage/f'{split}.jsonl').write_bytes(raw)
        hashes[split] = sha(raw)
    parent_hashes = {name: sha((parent/name).read_bytes())
                     for name in ('manifest.json', 'training.jsonl', 'development.jsonl', 'policy.md')}
    manifest = dict(schema='historical-dataset-system-v1', status='COMPLETE',
                    parent_release=str(parent), parent_hashes=parent_hashes,
                    system_prompt_sha256=sha(prompt_raw), policy_sha256=sha(policy_raw),
                    dataset_hashes=hashes, examples=examples, assignments=old['assignments'])
    (stage/'manifest.json').write_text(dumps(manifest)+'\n', encoding='utf-8')
    read_release(stage)
    if output.exists():
        raise FileExistsError(output)
    os.rename(stage, output)
    return manifest


def read_system_release(path, manifest):
    """Check exact parent answers, source cells, role order, and file hashes."""
    from proxybench.training.dataset import read_release

    path = Path(path)
    if manifest.get('status') != 'COMPLETE' or manifest.get('schema') != 'historical-dataset-system-v1':
        raise ValueError('Incomplete system-message dataset')
    parent = Path(manifest['parent_release'])
    if read_json((parent/'manifest.json').read_bytes()).get('schema') != 'historical-dataset-v1':
        raise ValueError('System dataset must bind a two-role parent')
    if set(manifest['parent_hashes']) != {'manifest.json', 'training.jsonl', 'development.jsonl', 'policy.md'}:
        raise ValueError('System dataset parent file inventory differs')
    for name, expected in manifest['parent_hashes'].items():
        if sha((parent/name).read_bytes()) != expected:
            raise ValueError(f'Parent release changed: {name}')
    old_rows, old = read_release(parent)
    prompt_raw = (path/'system-prompt.txt').read_bytes()
    policy_raw = (path/'policy.md').read_bytes()
    if sha(prompt_raw) != manifest['system_prompt_sha256'] or sha(policy_raw) != manifest['policy_sha256']:
        raise ValueError('System prompt or source policy changed')
    if sha(policy_raw) != old['policy_sha256'] or manifest['assignments'] != old['assignments']:
        raise ValueError('Parent source policy or partition assignments changed')
    if len(manifest['examples']) != len(old['examples']):
        raise ValueError('System release omitted an accepted example')
    prompt, policy = prompt_raw.decode('utf-8'), policy_raw.decode('utf-8')
    rows = {}
    parent_by_split = {split: iter(value) for split, value in old_rows.items()}
    if set(manifest['dataset_hashes']) != {'training', 'development'}:
        raise ValueError('System dataset partition inventory differs')
    for split, expected_hash in manifest['dataset_hashes'].items():
        if split not in {'training', 'development'}:
            raise ValueError('Unsupported system dataset partition')
        raw = (path/f'{split}.jsonl').read_bytes()
        if sha(raw) != expected_hash:
            raise ValueError('System dataset rows changed')
        rows[split] = [read_json(line) for line in raw.splitlines()]
    by_split = {split: iter(value) for split, value in rows.items()}
    for meta, prior in zip(manifest['examples'], old['examples'], strict=True):
        split = prior['split']
        if meta['split'] != split or meta['packet'] != prior['packet'] or meta['evidence'] != prior['evidence']:
            raise ValueError('System dataset changed source or review evidence')
        row = next(by_split[split])
        old_row = next(parent_by_split[split])
        messages = row['messages']
        if ([m['role'] for m in messages] != ['system', 'user', 'assistant']
                or messages[0]['content'] != prompt
                or messages[1]['content'] != source_only(prior['packet'], policy)
                or messages[2]['content'] != old_row['messages'][-1]['content']
                or sha(dumps(row).encode()) != meta['row_sha256']):
            raise ValueError('System release changed a source cell or accepted answer')
    if any(next(iterator, None) is not None for iterator in (*by_split.values(), *parent_by_split.values())):
        raise ValueError('System dataset row count differs from parent')
    return rows, manifest
