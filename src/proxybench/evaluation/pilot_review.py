"""Bind source-review decisions before revealing paired model identities."""

import html
import json
from pathlib import Path
import random

from proxybench.evaluation.training_labels import paired_report, score
from proxybench.execution.resources import durable_json
from proxybench.training.historical import check_files, inventory
from proxybench.training.smoke import digest, read_json


def source_text(meta, *, marked=True):
    blocks = []
    for block in meta['packet']['manifest']['blocks']:
        text = '\n'.join(cell['text'] for cell in block['cells'])
        if marked and block.get('target'):
            text = 'BEGIN MARKED TARGET\n'+text+'\nEND MARKED TARGET'
        blocks.append(text)
    return '\n'.join(blocks)


def create_review(directory, cases, *, identity, kind):
    directory.mkdir(exist_ok=False)
    rng = random.Random(42)
    display, mapping, files = [], {}, []
    for case in cases:
        names = list(case['answers'])
        rng.shuffle(names)
        key = case['id']
        mapping[key] = dict(zip(('A', 'B'), names, strict=True))
        answers = {}
        for letter, name in mapping[key].items():
            path = Path(case['answers'][name])
            files.append(path)
            answer = read_json(path.read_bytes())
            answers[letter] = dict(text=answer.get('text', ''), status=answer.get('status'),
                                   token_ids=answer.get('token_ids', []))
        display.append(dict(id=key, source=case['source'], reference=case.get('reference'), answers=answers))
    packet = dict(kind=kind, identity=identity, cases=display, evidence=inventory(files))
    durable_json(directory/'packet.json', packet)
    durable_json(directory/'mapping.json', mapping)
    template = dict(packet_sha256=digest(directory/'packet.json'), reviewer='', rationale='',
                    cases={case['id']: ({'decision': 'PENDING', 'reason': ''} if kind == 'export' else
                         {letter: dict(subject_equivalent=False, quotation_errors=[], reference_problem=False, reason='')
                          for letter in ('A', 'B')}) for case in cases})
    durable_json(directory/'decision-template.json', template)
    sections = []
    for case in display:
        reference = ('<details><summary>Accepted reference</summary><pre>'+html.escape(json.dumps(case['reference'], indent=2))+'</pre></details>') if case['reference'] else ''
        paired = ''.join('<section><h3>Answer '+letter+'</h3><p>'+html.escape(answer['status'])+'</p><pre>'+html.escape(answer['text'])+'</pre></section>'
                         for letter, answer in case['answers'].items())
        sections.append('<article><h2>'+html.escape(case['id'])+'</h2><pre>'+html.escape(case['source'])+'</pre>'+reference+
                        '<div class="pair">'+paired+'</div></article>')
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
        '<title>Historical source review</title><style>body{font:16px sans-serif;margin:2em}pre{white-space:pre-wrap;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:2em}article{border-bottom:2px solid #aaa}section{min-width:0}</style>'
        '<h1>Source review</h1><p>Read each source before its answers. Save decisions before revealing the model mapping. '
        'Literal excerpts do not prove that a claim is supported. This page hides display identities, but the coordinator also knows the experiment.</p>'+''.join(sections), encoding='utf-8')
    durable_json(directory/'binding.json', dict(packet_sha256=digest(directory/'packet.json'),
                 mapping_sha256=digest(directory/'mapping.json')))
    return directory


def accept_decisions(directory, decisions, identity):
    packet = read_json((directory/'packet.json').read_bytes())
    frozen = read_json((directory/'binding.json').read_bytes())
    if (packet['identity'] != identity or decisions.get('packet_sha256') != digest(directory/'packet.json')
            or decisions['packet_sha256'] != frozen['packet_sha256']
            or digest(directory/'mapping.json') != frozen['mapping_sha256']):
        raise ValueError('Review input, output, or run identity changed')
    check_files(packet['evidence'])
    if not decisions.get('reviewer') or not decisions.get('rationale'):
        raise ValueError('Review requires a reviewer and source-based rationale')
    if set(decisions['cases']) != {c['id'] for c in packet['cases']}:
        raise ValueError('Review must resolve every case')
    for row in decisions['cases'].values():
        if packet['kind'] == 'export':
            if row.get('decision') != 'PASS' or not row.get('reason'):
                raise ValueError('A rejected or unresolved export cannot continue')
        else:
            if set(row) != {'A', 'B'}:
                raise ValueError('Both answers need source review')
            for decision in row.values():
                if (type(decision.get('subject_equivalent')) is not bool
                        or type(decision.get('reference_problem')) is not bool
                        or not isinstance(decision.get('quotation_errors'), list)
                        or any(not isinstance(v, str) for v in decision['quotation_errors'])
                        or not decision.get('reason')):
                    raise ValueError('Incomplete source-review decision')
    if (directory/'accepted.json').exists():
        raise FileExistsError(directory/'accepted.json')
    durable_json(directory/'accepted.json', dict(decisions=decisions, binding=frozen))
    # Publication of this file happens only after durable decisions.
    durable_json(directory/'revealed-mapping.json', read_json((directory/'mapping.json').read_bytes()))


def require_decisions(directory, identity):
    accepted = read_json((directory/'accepted.json').read_bytes())
    packet = read_json((directory/'packet.json').read_bytes())
    if (packet['identity'] != identity or accepted['binding']['packet_sha256'] != digest(directory/'packet.json')
            or accepted['binding']['mapping_sha256'] != digest(directory/'mapping.json')
            or accepted['decisions']['packet_sha256'] != digest(directory/'packet.json')):
        raise ValueError('Missing or stale source-review decision')
    check_files(packet['evidence'])
    if packet['kind'] == 'export' and any(v['decision'] != 'PASS' for v in accepted['decisions']['cases'].values()):
        raise ValueError('Rejected export review')
    return accepted['decisions'], read_json((directory/'mapping.json').read_bytes())


def finish_report(root, state, prepared):
    directory = root/'review-final'
    decisions, mapping = require_decisions(directory, state['identity'])
    metadata = [m for m in prepared['examples'] if m['split'] == 'development']
    cases, invalid = [], []
    for index, meta in enumerate(metadata):
        key = meta['packet']['manifest']['packet_id']
        row = dict(id=key, family=meta['packet']['manifest']['family'])
        reference = read_json(prepared['rows']['development'][index]['messages'][1]['content'])
        for letter, model in mapping[key].items():
            decision = decisions['cases'][key][letter]
            if decision['reference_problem']:
                invalid.append(key)
            phase = 'baseline' if model == 'original' else 'final'
            answer = read_json((Path(state['completed_phases'][phase]['output'])/f'answer-{index}.json').read_bytes())
            row[model] = score(reference, answer, source_text(meta, marked=False),
                               subject_equivalent=decision['subject_equivalent'], quotation_errors=decision['quotation_errors'])
        cases.append(row)
    report = paired_report(cases)
    report.update(status='INVALID_REFERENCE' if invalid else 'COMPLETE', invalid_reference_cases=sorted(set(invalid)),
                  identity=state['identity'], review_sha256=digest(directory/'accepted.json'))
    durable_json(root/'report.json', report)
    return report['status']
