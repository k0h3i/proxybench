"""Source-first review pages and explicit, non-admitting draft ledgers."""

import html
import json
from pathlib import Path

from proxybench.execution.runner import write_json


def render(output, targets, report, *, review_budget):
    """Publish escaped sources before suggestions; export review notes separately."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = {r['input_id']: r for r in report['results']}
    ledger, sections = [], []
    for i, target in enumerate(targets):
        input_id = target['bundle']['input_id']
        result = results[input_id]
        raw_path = Path(target['attempt_directory']) / 'raw.bin'
        raw = raw_path.read_bytes().decode('utf-8', errors='replace') if raw_path.exists() else 'No native final response was captured.'
        source = '\n\n'.join(f'Block {j}\n{b["original_text"]}' for j, b in enumerate(target['bundle']['blocks']))
        ledger.append(dict(input_id=input_id, generated=raw_path.exists(),
            mechanically_valid=result.get('mechanically_valid'),
            execution_eligible=result.get('execution_eligible', False), source_reviewed=False,
            unresolved=True, rejected=False, training_admitted=False, active_seconds=None,
            boundary_review='PENDING', record_multiplicity=None, scope_review='PENDING',
            source_group=target['group_id'], label_revision=None, label_sha256=None,
            reference_role=target.get('reference_role'), corrections=[],
            generation_status=result['status'], mechanical_error=result.get('mechanical_error')))
        def escaped(value):
            return html.escape(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
        sections.append(f'''<section data-id="{html.escape(input_id, quote=True)}">
<h2>{i + 1}. {escaped(input_id)}</h2>
<p>{escaped(target['selection_reason'])}</p>
<p>Target: {escaped(target['bundle']['target'])}. All positions use UTF-8 bytes, with the end excluded.</p>
<p>{escaped(target.get('boundary_finding', 'The target boundary needs source review.'))}</p>
<button class="start">Start or resume source review</button>
<button class="pause">Pause review</button><span class="timer">0 seconds</span>
<h3>Source</h3><pre>{escaped(source)}</pre>
<button class="reveal" disabled>Reveal suggestions after reading the source</button>
<div class="suggestions" hidden>
<h3>Execution and mechanical findings</h3><pre>{escaped(result)}</pre>
<h3>Original Sol draft</h3><pre>{escaped(raw)}</pre>
<h3>Independent mapper observations</h3><pre>{escaped(target.get('mapper', 'No modern mapper applies to this historical input.'))}</pre>
</div>
<label>Review finding <select><option>UNRESOLVED</option><option>SOURCE_REVIEWED</option><option>REJECTED</option></select></label>
<p>Record the subject, reporting scope, record count, unsupported values, evidence meaning, corrections, and disagreements.</p>
<textarea rows="7" aria-label="Review notes"></textarea>
</section>''')
    budget = html.escape(json.dumps(review_budget, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>First-wave source review</title>
<style>body{max-width:1050px;margin:2em auto;font:18px sans-serif;padding:1em}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;background:#f5f5f5;padding:1em}section{border-top:2px solid #777;margin-top:3em}button,select{font:inherit;margin:.4em;padding:.4em}textarea{width:100%;font:inherit} [hidden]{display:none}</style>
<h1>First-wave source review</h1>
<p>Read each source before revealing suggestions. A mechanically valid response still needs review of its meaning.</p>
<p>These are development drafts. This page does not admit labels for training or change accepted historical references.</p>
<p>The timer records active page time. Include discussion, corrections, and repeated review in the same review allowance.</p>
<p>Download notes before closing this page. The page does not save notes to the repository.</p>
<h2>Review allowance</h2><pre>''' + budget + '''</pre>
<button id="download">Pause all timers and download review notes</button>
''' + ''.join(sections) + '''
<script>
const records = new Map();
function pause(s) {const r=records.get(s); if(r.start!==null){r.seconds+=(performance.now()-r.start)/1000;r.start=null;}}
document.querySelectorAll('section').forEach(s=>{
 records.set(s,{seconds:0,start:null,revealed:false});
 s.querySelector('.start').onclick=()=>{records.forEach((r,other)=>pause(other));records.get(s).start=performance.now();s.querySelector('.reveal').disabled=false;};
 s.querySelector('.pause').onclick=()=>pause(s);
 s.querySelector('.reveal').onclick=()=>{records.get(s).revealed=true;s.querySelector('.suggestions').hidden=false;};
});
setInterval(()=>records.forEach((r,s)=>{s.querySelector('.timer').textContent=Math.floor(r.seconds+(r.start===null?0:(performance.now()-r.start)/1000))+' seconds';}),500);
document.addEventListener('visibilitychange',()=>{if(document.hidden)records.forEach((r,s)=>pause(s));});
document.querySelector('#download').onclick=()=>{
 const rows=[]; records.forEach((r,s)=>{pause(s);rows.push({input_id:s.dataset.id,active_seconds:r.seconds,suggestions_revealed:r.revealed,status:s.querySelector('select').value,notes:s.querySelector('textarea').value,training_admitted:false});});
 const blob=new Blob([JSON.stringify({version:'first-wave-review-v1',exported_at:new Date().toISOString(),rows},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='first-wave-review-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
</script></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    write_json(output / 'admission-ledger.json', ledger)
    write_json(output / 'review-budget.json', review_budget)
    return ledger
