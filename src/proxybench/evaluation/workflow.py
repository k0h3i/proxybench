"""Evaluate one selected model and review its exact saved answers."""
import html
import json
from pathlib import Path

from proxybench.evaluation.answers import score_system, SCORER_VERSION
from proxybench.evaluation.training_labels import parse_answer, quote_paths
from proxybench.runstate import Run, atomic_json, atomic_text, binding, file_hash
from proxybench.training.labels import read_json, validate

REPORT_SCHEMA = 'proxybench-evaluation-v1'


def runtime_binding(config):
    files = {}
    for key in ('system_prompt', 'runtime_manifest'):
        if key in config:
            files[key] = file_hash(config[key])
    if 'tokenizer' in config:
        root = Path(config['tokenizer'])
        files['tokenizer'] = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}
    return binding(dict(configuration=config, files=files))


def prepare_inputs(dataset, model, config, *, project_root=None):
    from proxybench.training.dataset import read_release
    from proxybench.annotation.testing import SCHEMA as TEST_SCHEMA, read_test_release
    schema = read_json((Path(dataset) / 'dataset-manifest.json').read_bytes()).get('schema')
    split = 'test' if schema == TEST_SCHEMA else 'development'
    reader = read_test_release if split == 'test' else read_release
    rows, manifest = reader(dataset, project_root=project_root)
    cases = []
    metadata = [item for item in manifest['examples'] if item['split'] == split]
    for index, row in enumerate(rows[split]):
        item = next(entry for entry in metadata if entry['position'] == index)
        cells = [dict(block_index=b, cell_index=c, text=cell['text'])
                 for b, block in enumerate(item['packet']['manifest']['blocks'])
                 for c, cell in enumerate(block['cells'])]
        cases.append(dict(id=str(index), reference=read_json(row['messages'][-1]['content']),
                          source_cells=cells, messages=row['messages'][:-1]))
    identity = dict(dataset=binding(manifest), split=split, model=file_hash(model), runtime=runtime_binding(config),
                    prompt=binding([case['messages'][0] for case in cases]))
    if split == 'test':
        from proxybench.training.dataset import project_file
        from proxybench.annotation.testing import acceptance_content
        preparation = read_json(acceptance_content(manifest['acceptance']['preparation']))
        protocol = read_json(project_file(Path(project_root or Path.cwd()).resolve(), preparation['protocol']))
        identity['scorer_version'] = protocol['scorer_version']
        check_test_scorer(dict(dataset_split=split, identity=identity))
    return dict(schema=REPORT_SCHEMA, identity=identity, dataset_split=split, cases=cases)


def check_test_scorer(inputs):
    if inputs.get('dataset_split') == 'test' and inputs['identity'].get('scorer_version') != SCORER_VERSION:
        raise ValueError('Frozen test scorer differs from the current scorer version')


def create_run(path, inputs, config):
    check_test_scorer(inputs)
    ids = [case['id'] for case in inputs['cases']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Evaluation requires unique nonempty targets')
    with Run(path, create=True, identity=inputs['identity'], config=config) as run:
        atomic_json(run.path / 'evaluation' / 'inputs.json', inputs)
        run.state['operation'] = 'evaluate'
        run.state['evaluation_resource_base_seconds'] = run.state['consumed_seconds']
        run.state['inputs_sha256'] = binding(inputs)
        run.save()


def load_inputs(run):
    inputs = read_json((run.path / 'evaluation' / 'inputs.json').read_text())
    if binding(inputs) != run.state['inputs_sha256'] or inputs['identity'] != run.state['identity']:
        raise ValueError('Saved evaluation inputs changed')
    check_test_scorer(inputs)
    return inputs


def answer_binding(inputs, case, answer):
    return binding(dict(identity=inputs['identity'], case=case, answer=answer))


def load_answers(run, inputs):
    from proxybench.execution.resources import reconcile_captures
    capture_seconds = reconcile_captures(run.path / 'evaluation' / 'capture')
    from proxybench.evaluation.session import recover_sessions
    recover_sessions(run, inputs)
    answers = {}
    for case in inputs['cases']:
        path = run.path / 'evaluation' / 'results' / (binding(case['id']) + '.json')
        if not path.exists():
            continue
        saved = read_json(path.read_text())
        if saved['binding'] != answer_binding(inputs, case, saved['answer']):
            raise ValueError('A cached answer has a stale input binding')
        answers[case['id']] = saved['answer']
    # Reconciled executions can exceed a reservation during interrupted cleanup.
    # Preserve the measured cost after session settlement.
    run.state['consumed_seconds'] = max(run.state['consumed_seconds'],
                                      run.state['evaluation_resource_base_seconds'] + capture_seconds)
    run.save()
    atomic_text(run.path / 'evaluation' / 'answers.jsonl', ''.join(
        json.dumps(dict(id=case['id'], binding=answer_binding(inputs, case, answers[case['id']]),
                        answer=answers[case['id']]), ensure_ascii=False) + '\n'
        for case in inputs['cases'] if case['id'] in answers))
    return answers


def review_template(inputs, case, answer):
    return dict(id=case['id'], binding=answer_binding(inputs, case, answer), reviewed=False,
                subject_equivalent=False, subject_evidence=None, quotation_errors=[], other_equivalences=[])


def validate_decision(inputs, case, answer, decision):
    expected = review_template(inputs, case, answer)
    if set(decision) != set(expected) or decision['id'] != case['id'] or decision['binding'] != expected['binding']:
        raise ValueError('Review decision does not match its exact inputs')
    if decision['reviewed'] is not True or type(decision['subject_equivalent']) is not bool:
        raise ValueError('Review needs explicit acceptance and a subject decision')
    if not isinstance(decision['quotation_errors'], list) or not all(isinstance(p, str) for p in decision['quotation_errors']):
        raise ValueError('Quotation errors must be a list of field paths')
    prediction = parse_answer(answer)
    if prediction is None:
        raise ValueError('Review requires a complete, valid answer')
    paths = {path for path, _ in quote_paths(prediction['fields'])}
    if any(path not in paths for path in decision['quotation_errors']):
        raise ValueError('Quotation decision refers to an unknown quoted field')
    if decision['subject_equivalent']:
        evidence = decision['subject_evidence']
        if not isinstance(evidence, dict) or set(evidence) != {'block_index', 'cell_index', 'passage', 'reason'}:
            raise ValueError('Subject equivalence needs source-cell evidence')
        cell = next((c for c in case['source_cells'] if c['block_index'] == evidence['block_index'] and c['cell_index'] == evidence['cell_index']), None)
        if cell is None or not evidence['passage'] or evidence['passage'] not in cell['text'] or not evidence['reason']:
            raise ValueError('Subject equivalence has invalid evidence')
    score_system(case['reference'], answer, case, subject_equivalent=decision['subject_equivalent'],
                 quotation_errors=decision['quotation_errors'], other_equivalences=decision['other_equivalences'])


def import_review(run, inputs, answers, path):
    incoming = read_json(Path(path).read_text())
    destination = run.path / 'evaluation' / 'review' / 'decisions.json'
    committed = read_json(destination.read_text()) if destination.exists() else {}
    if not isinstance(incoming, list):
        raise ValueError('Review import must contain a list of decisions')
    for decision in incoming:
        case = next((c for c in inputs['cases'] if c['id'] == decision.get('id')), None)
        if case is None or case['id'] not in answers:
            raise ValueError('Review refers to an unknown or incomplete target')
        validate_decision(inputs, case, answers[case['id']], decision)
        previous = committed.get(case['id'])
        if previous is not None and previous != decision:
            raise ValueError('Review conflicts with a committed decision')
        committed[case['id']] = decision
    atomic_json(destination, committed)


def write_review(run, inputs, pending, answers):
    folder = run.path / 'evaluation' / 'review'
    decisions = [review_template(inputs, case, answers[case['id']]) for case in pending]
    atomic_json(folder / 'template.json', decisions)
    content = '<!doctype html><meta charset="utf-8"><title>Source review</title><h1>Source review</h1>'
    for case in pending:
        content += '<h2>Target ' + html.escape(case['id']) + '</h2><h3>Source cells</h3><pre>'
        content += html.escape(json.dumps(case['source_cells'], ensure_ascii=False, indent=2)) + '</pre>'
        content += '<h3>Reference</h3><pre>' + html.escape(json.dumps(case['reference'], ensure_ascii=False, indent=2)) + '</pre>'
        content += '<h3>Answer</h3><pre>' + html.escape(answers[case['id']]['text']) + '</pre>'
    payload = []
    for case, decision in zip(pending, decisions):
        prediction = parse_answer(answers[case['id']])
        payload.append(dict(decision=decision, cells=case['source_cells'],
                            quotes=[path for path, _ in quote_paths(prediction['fields'])],
                            reference_sha256=binding(case['reference']), answer_sha256=binding(answers[case['id']])))
    content += '<div id="controls"></div><button id="save">Export decisions</button>'
    content += '<script type="application/json" id="data">' + json.dumps(payload).replace('<', '\\u003c') + '</script>'
    content += r"""<script>
    const items=JSON.parse(document.getElementById('data').textContent), root=document.getElementById('controls');
    const forms=[];
    function element(tag,parent,text){const n=document.createElement(tag); if(text)n.textContent=text;parent.append(n);return n;}
    function input(parent,label,type='text'){const l=element('label',parent,label+' '), n=element('input',l);n.type=type;element('br',parent);return n;}
    function evidence(parent,item){const sel=element('select',parent);item.cells.forEach((c,i)=>{const o=element('option',sel,`Block ${c.block_index}, cell ${c.cell_index}: ${c.text.slice(0,100)}`);o.value=i;});
      return {sel, passage:input(parent,'Exact source passage'),reason:input(parent,'Reason')};}
    function getEvidence(ui,item){const c=item.cells[Number(ui.sel.value)];return {block_index:c.block_index,cell_index:c.cell_index,passage:ui.passage.value,reason:ui.reason.value};}
    items.forEach(item=>{const f=element('fieldset',root);element('legend',f,'Target '+item.decision.id);
      const reviewed=input(f,'I reviewed the source, reference, and answer','checkbox');
      const subject=input(f,'The subject wording has the same meaning','checkbox'), se=evidence(f,item);
      element('p',f,'Select quotations that do not support their fields.');
      const quotes=item.quotes.map(path=>[path,input(f,path,'checkbox')]);
      element('p',f,'Optional OTHER equivalence. Both fields must use OTHER and quote the same source cell.');
      const other=input(f,'Accept one OTHER wording equivalence','checkbox');
      const ref=input(f,'Reference field path (start with fields.)'), pred=input(f,'Answer field path (start with fields.)');
      const column=input(f,'Source column'), oe=evidence(f,item);
      forms.push(()=>{const d={...item.decision,reviewed:reviewed.checked,subject_equivalent:subject.checked,
        subject_evidence:subject.checked?getEvidence(se,item):null,quotation_errors:quotes.filter(x=>x[1].checked).map(x=>x[0]),other_equivalences:[]};
        if(other.checked)d.other_equivalences.push({...getEvidence(oe,item),reference_field_path:ref.value,prediction_field_path:pred.value,
          column:column.value,reference_sha256:item.reference_sha256,answer_sha256:item.answer_sha256});return d;});
    });
    document.getElementById('save').onclick=()=>{const values=forms.map(f=>f());if(values.some(d=>!d.reviewed)){alert('Review every target before export.');return;}
      const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(values,null,2)],{type:'application/json'}));
      a.download='review-decisions.json';a.click();URL.revokeObjectURL(a.href);};
    </script>"""
    atomic_text(folder / 'index.html', content)


def report(run, inputs, answers):
    check_test_scorer(inputs)
    destination = run.path / 'evaluation' / 'review' / 'decisions.json'
    decisions = read_json(destination.read_text()) if destination.exists() else {}
    invalid, missing, pending, scores = [], [], [], []
    for case in inputs['cases']:
        try:
            validate(case['reference'])
        except (ValueError, TypeError, KeyError) as error:
            invalid.append(dict(id=case['id'], error=str(error)))
            continue
        answer = answers.get(case['id'])
        if answer is None:
            missing.append(case['id'])
            continue
        decision = decisions.get(case['id'])
        if parse_answer(answer) is not None and decision is None:
            pending.append(case)
            continue
        options = {}
        if decision is not None:
            validate_decision(inputs, case, answer, decision)
            options = {key: decision[key] for key in ('subject_equivalent', 'quotation_errors', 'other_equivalences')}
        scores.append(dict(id=case['id'], **score_system(case['reference'], answer, case, **options)))
    status = 'INVALID_REFERENCES' if invalid else 'INCOMPLETE' if missing else 'PENDING_REVIEW' if pending else 'COMPLETE'
    result = dict(schema=REPORT_SCHEMA, scorer=SCORER_VERSION, identity=inputs['identity'], status=status,
                  valid_accuracy=status == 'COMPLETE', targets=len(inputs['cases']), invalid_references=invalid,
                  missing_answers=missing, pending_review=[case['id'] for case in pending], cases=scores,
                  aggregate=None, dataset_split=inputs.get('dataset_split', 'development'),
                  interpretation=('Frozen test data; independence is limited to audited project exposure'
                                  if inputs.get('dataset_split') == 'test'
                                  else 'Development data; not an untouched test set'))
    if status == 'COMPLETE':
        result['aggregate'] = {key: sum(bool(row[key]) for row in scores)
                               for key in ('format_valid', 'exact', 'source_value_correct')}
        result['aggregate']['fields'] = {key: sum(row['field_correct'][key] for row in scores)
                                         for key in scores[0]['field_correct']} if scores else {}
        for key in ('origin_errors', 'derivation_errors', 'unsupported_quotes', 'quotation_errors'):
            result['aggregate'][key] = sum(len(row[key] or []) for row in scores)
    if pending:
        write_review(run, inputs, pending, answers)
        result['review_path'] = str(run.path / 'evaluation' / 'review' / 'index.html')
    atomic_json(run.path / 'evaluation' / 'report.json', result)
    run.state['status'] = status
    run.save()
    return result
