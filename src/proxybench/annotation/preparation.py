"""Prepare source-first reviews and export explicitly accepted labels."""

from pathlib import Path
import os
import tempfile

from proxybench.annotation.bindings import review_binding
from proxybench.annotation.historical import VIEW_VERSION, literal_support, prepare_historical
from proxybench.annotation.review import write_review
from proxybench.training.dataset import COORDINATES, FILES, SCHEMA, check_assignments, preserve_exposure, read_release
from proxybench.training.labels import dumps, from_review, read_json, sha


def prepare_review(root, selections, output, *, prompt_path='configs/model-system-prompt.txt',
                   label_contract_path='docs/label-contract.md', assignments, draft_path=None):
    root, output = Path(root).resolve(), Path(output)
    if any(s['split'] == 'test' for s in selections):
        raise ValueError('Use prepare-test for protected test sources')
    assignments = preserve_exposure(root, assignments)
    from proxybench.sources.holdout import reject_protected_sources
    reject_protected_sources(root, selections, assignments)
    check_assignments(selections, assignments)
    packets = [prepare_historical(root, selection) for selection in selections]
    if len({p['manifest']['packet_id'] for p in packets}) != len(packets):
        raise ValueError('Review packet IDs must be unique')
    output.mkdir(parents=True, exist_ok=False)
    (output/'packet-set.json').write_text(dumps(packets)+'\n', encoding='utf-8')
    # Keep the exact prompt visible to the labeling tool, outside source messages.
    (output/'system-prompt.txt').write_bytes((root/prompt_path).read_bytes())
    return write_review(root, packet_directory=output.resolve(), draft_path=draft_path,
                        training_contract=(root/label_contract_path).read_text(encoding='utf-8'))


def accept_review(root, review_dir, review_path, approval_path, output, *, assignments,
                  prompt_path='configs/model-system-prompt.txt', label_contract_path='docs/label-contract.md'):
    root, output = Path(root).resolve(), Path(output).resolve()
    raw = Path(review_path).read_bytes()
    review, approval = read_json(raw), read_json(Path(approval_path).read_bytes())
    if (approval.get('decision') != 'ACCEPTED' or approval.get('export_sha256') != sha(raw)
            or not approval.get('reviewer') or not approval.get('accepted_at')):
        raise ValueError('Explicit acceptance must identify the exact review, reviewer, and date')
    ids = approval.get('accepted_packet_ids')
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError('Acceptance requires distinct packet IDs')
    if review.get('schema') != 'training-review-v1':
        raise ValueError('Unsupported label review format')
    packets = read_json((Path(review_dir)/'packet-set.json').read_bytes())
    by_id = {p['manifest']['packet_id']: p for p in packets}
    if len(by_id) != len(packets) or not set(ids) <= by_id.keys():
        raise ValueError('Duplicate or unknown packet ID')
    prompt_raw = (root/prompt_path).read_bytes()
    if prompt_raw != (Path(review_dir)/'system-prompt.txt').read_bytes():
        raise ValueError('Prompt changed after source review')
    rows = {split: [] for split in FILES}
    examples, selections = [], []
    for packet_id in ids:
        prior = by_id[packet_id]
        selection = {k: v for k, v in prior['manifest'].items()
                     if k not in {'blocks', 'packet_version', 'view_version', 'target_id'}}
        packet = prepare_historical(root, selection)
        if packet != {k: prior[k] for k in ('manifest', 'source_view', 'model_input')}:
            raise ValueError('Review source changed')
        answer = review['packets'][packet_id]
        if answer.get('reviewed') is not True or answer.get('input_binding') != review_binding(packet):
            raise ValueError('Label needs completed review against the exact source')
        label = from_review(answer['fields'])
        literal_support(label, packet)
        split = selection['split']
        if split not in FILES:
            raise ValueError('Use accept-test for test references')
        row = {'messages': [dict(role='system', content=prompt_raw.decode('utf-8')),
                            dict(role='user', content=packet['model_input']),
                            dict(role='assistant', content=dumps(label))]}
        examples.append(dict(split=split, position=len(rows[split]), row_sha256=sha(dumps(row).encode()),
                             selection=selection))
        rows[split].append(row)
        selections.append(selection)
    assignments = preserve_exposure(root, assignments)
    from proxybench.sources.holdout import reject_protected_sources
    reject_protected_sources(root, selections, assignments)
    check_assignments(selections, assignments)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+'.incomplete-', dir=output.parent))
    files = {}
    for split, name in FILES.items():
        raw = ''.join(dumps(row)+'\n' for row in rows[split]).encode()
        (stage/name).write_bytes(raw)
        files[split] = dict(path=name, sha256=sha(raw), examples=len(rows[split]))
    manifest = dict(schema=SCHEMA, status='COMPLETE', coordinate_convention=COORDINATES,
                    rendering_version=VIEW_VERSION, files=files, examples=examples, assignments=assignments,
                    system_prompt=dict(path=prompt_path, sha256=sha(prompt_raw)),
                    label_contract=dict(path=label_contract_path, sha256=sha((root/label_contract_path).read_bytes())))
    (stage/'dataset-manifest.json').write_text(dumps(manifest)+'\n', encoding='utf-8')
    read_release(stage, project_root=root)
    if output.exists():
        raise FileExistsError(output)
    os.rename(stage, output)
    return manifest
