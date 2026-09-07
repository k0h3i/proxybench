"""Write a local review page with optional hidden assistant drafts."""

import argparse
import hashlib
import json
from pathlib import Path

from proxybench.annotation.packets import FIELDS


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ProxyBench · First calibration review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#edf1f3;color:#192b37;font:16px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:24px auto;padding:0 22px}h1{font-size:28px;margin-bottom:8px}h2{font-size:22px}
p{max-width:85ch}button,input,textarea,select{font:inherit}button{padding:8px 15px;border:1px solid #8498a5;border-radius:5px;background:white;cursor:pointer}
button.primary{background:#164c68;color:white;border-color:#164c68}button:disabled{opacity:.55;cursor:default}
input,textarea,select{width:100%;border:1px solid #8e9fab;border-radius:4px;padding:6px;background:white;color:#192b37}
textarea{min-height:68px;resize:vertical}label{display:block;margin:10px 0}input[type=checkbox]{width:auto}
.card{background:white;border:1px solid #c9d3da;border-radius:8px;padding:20px;margin:16px 0}.bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.timer{font-variant-numeric:tabular-nums;margin-left:auto}.small{font-size:14px;color:#4c626f}.notice{border-left:4px solid #bb841f;padding-left:12px}
iframe{width:100%;height:650px;border:1px solid #b5c2cb;background:white}nav{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}
nav button[aria-current=true]{background:#d7e8f0;border:2px solid #164c68}table.fields{width:100%;border-collapse:collapse;font-size:14px}
.fields th{text-align:left}.fields td,.fields th{padding:9px 6px;vertical-align:top;border-bottom:1px solid #d4dde3}
.fields td:first-child{width:16%}.fields td:nth-child(2){width:31%}.fields td:last-child{width:22%}fieldset{border:0;padding:0;margin:0}
summary{cursor:pointer;color:#164c68}a{color:#164c68}#status{min-height:24px}#prior{max-width:110px}
@media(max-width:750px){main{padding:0 10px}.card{padding:12px}.fields{display:block;overflow:auto}iframe{height:550px}}
</style></head><body><main>
<h1>First calibration review</h1>
<p>Six source packets are ready. Read the yellow target and form your answer first. Then reveal the draft and revise it as needed.</p>
<div class="card">
<div class="bar"><button id="start" class="primary">Start review</button><button id="pause" disabled>Pause</button><button id="save">Download review draft</button><span id="timer" class="timer">00:00 / 30:00</span></div>
<p class="small">Start the timer before reading the sources or guide. Pause for interruptions. This page pauses when you leave its browser tab.</p>
<label>Earlier active review minutes, including guide reading and corrections <input id="prior" type="number" min="0" max="240" step="0.1" value="0"></label>
<p class="small">This session stops at 30 active minutes or 240 total minutes. Reserve at least 30 total minutes for corrections.</p>
<p id="status" role="status">No review time is recorded until you start. Complete fewer packets if time runs out.</p>
<label class="small">Resume a downloaded draft <input id="load" type="file" accept="application/json,.json"></label>
</div>
<div id="work" hidden>
<nav id="nav" aria-label="Calibration packets"></nav>
<section class="card"><h2 id="title"></h2><p id="admin" class="small"></p>
<p class="notice">Label only the yellow target. Adjacent rows supply context. One target can continue across two rows.</p>
<iframe id="source" title="Original source passages with one marked target" sandbox=""></iframe>
<p id="locations" class="small"></p>
<details><summary>Guide and field meanings</summary>
<p>Reporting fund or fund group identifies whose voting record this is. Issuer name identifies the organization whose security is held.</p>
<p>The issuer can also be a fund. Use the reporting heading to distinguish these roles.</p>
<p>Evidence means the source location that supports a value.</p>
<p>Preserve the original meeting date. For this pilot, interpret numeric dates as month/day/year or year-month-day when the year comes first.</p>
<p>If both date orders remain possible, record the month/day interpretation separately as INFERRED. This pilot assumption is not an SEC-wide rule.</p>
<p>Read the packet and form your answer before revealing the draft. You do not need to type your answer first.</p>
<p>Compare the draft with the source. Agreement alone does not prove that either answer is correct.</p>
<p>Keep identifiers and proposal numbers as strings. Keep unresolved evidence unresolved.</p>
<p>Record source wording in the value box. For each vote component, include its direction, disclosed quantity, and disclosed management alignment.</p>
<p>A vote component holds one direction and its disclosed details. Preserve multiple directions without inventing quantities.</p>
<p>A blank cell does not establish an explicit absence. Keep packet absence separate from findings elsewhere in the original filing.</p>
<p>Use PRESENT for a supported value, ABSENT_IN_CONTEXT for missing packet evidence, and AMBIGUOUS for multiple possible interpretations.</p>
<p>Use UNREADABLE for illegible evidence, CONFLICTING for disagreeing disclosures, and NOT_APPLICABLE when the field does not apply.</p>
<p>Use EXTRACTED for disclosed values, DERIVED for calculations, and INFERRED for added semantic interpretations. Leave origin blank for unresolved fields.</p>
<p>Do not reconstruct a recommendation from management alignment. Do not expand a collective vote or fund group into individual records.</p>
<p>Use block numbers and exact quotations for evidence. A source location alone does not prove that a value is supported.</p>
<p>Leave untouched fields blank. Unreviewed fields are not missing-value labels. Keep difficult cases in the feedback box.</p>
<p>These drafts inform the initial guide. They do not freeze the benchmark schema or become accepted reference labels automatically.</p>
</details>
</section>
<section class="card"><h2>Compare and revise</h2>
<div id="suggestion-controls" hidden>
<label><input id="source-read" type="checkbox"> I read the source and formed my own answer.</label>
<button id="reveal" class="primary" disabled>Reveal draft labels</button>
<p class="small">Drafts are suggestions. Revise any answer that the source does not support, even when it matches your first answer.</p>
<p class="small">Revealing fills untouched fields. Your existing answers stay in place, with the assistant draft shown beside them.</p>
</div>
<p id="draft-state" class="small">Fields remain blank until you reveal a draft or enter your own answers.</p>
<fieldset id="form"><table class="fields"><thead><tr><th>Field</th><th>Value in source wording</th><th>Availability</th><th>Origin</th><th>Evidence: block and quotation</th></tr></thead><tbody id="fields"></tbody></table>
<label>Feedback on the display, target boundary, or unresolved fields<textarea id="feedback"></textarea></label>
<label><input id="reviewed" type="checkbox"> I finished reviewing this packet's fields.</label></fieldset>
<p class="small">Download a draft when you stop. The download preserves fields, packet versions, and active review time.</p></section>
</div><p class="small">All six examples remain development data. Review this first set before any further pilot packets.</p>
</main>
<script id="packet-data" type="application/json">__PACKETS__</script>
<script id="assistant-data" type="application/json">__DRAFTS__</script>
<script>
'use strict';
const packets=JSON.parse(document.getElementById('packet-data').textContent);
const suggestions=JSON.parse(document.getElementById('assistant-data').textContent);
const fieldNames=__FIELDS__;
const fieldLabels={reporting_scope:'Reporting fund or fund group'};
const availability=['','PRESENT','ABSENT_IN_CONTEXT','AMBIGUOUS','UNREADABLE','CONFLICTING','NOT_APPLICABLE'];
const origins=['','EXTRACTED','DERIVED','INFERRED'];
const $=id=>document.getElementById(id);
const key='proxybench-calibration-first-v1';
let state={set_id:'calibration-first-v1',schema:'calibration-draft-v1',session_id:crypto.randomUUID(),revision:0,active_seconds:0,prior_review_minutes:0,session_started_at:null,events:[],packets:{}};
let current=0,running=false,last=0;
for(const p of packets)state.packets[p.manifest.packet_id]={packet_id:p.manifest.packet_id,packet_version:p.manifest.packet_version,source_sha256:p.manifest.source_sha256,status:'unreviewed',fields:Object.fromEntries(fieldNames.map(f=>[f,{value:'',raw_text:'',availability:'',origin:'',evidence_input:''}])),feedback:'',reviewed:false};
function message(text){$('status').textContent=text;}
function persist(){try{localStorage.setItem(key,JSON.stringify(state));}catch(e){message('Browser draft storage is unavailable. Download your draft before closing this page.');}}
function allowed(){return Math.max(0,Math.min(1800,(240-state.prior_review_minutes)*60));}
function updateTime(){const s=Math.floor(state.active_seconds),m=Math.floor(s/60);$('timer').textContent=String(m).padStart(2,'0')+':'+String(s%60).padStart(2,'0')+' / '+Math.floor(allowed()/60)+':00';}
function tick(){if(!running)return;const now=performance.now();state.active_seconds=Math.min(allowed(),state.active_seconds+(now-last)/1000);last=now;updateTime();if(state.active_seconds>=allowed()){pause('The review limit is reached. Download the completed work and leave the rest unresolved.');}else persist();}
function pause(reason='Review paused. Resume when you are ready.'){if(running){const now=performance.now();state.active_seconds=Math.min(allowed(),state.active_seconds+(now-last)/1000);}running=false;state.events.push({type:'pause',at:new Date().toISOString(),active_seconds:state.active_seconds});$('work').hidden=true;$('pause').disabled=true;$('prior').disabled=false;$('start').disabled=state.active_seconds>=allowed();$('start').textContent='Resume review';$('load').disabled=false;updateTime();persist();message(reason);}
function start(){state.prior_review_minutes=Number($('prior').value);if(!Number.isFinite(state.prior_review_minutes)||state.prior_review_minutes<0||state.prior_review_minutes>240){message('Enter earlier review minutes between 0 and 240.');return;}if(state.active_seconds>=allowed()){message('The review limit is reached. Download the completed work.');return;}state.session_started_at ||= new Date().toISOString();state.events.push({type:'start',at:new Date().toISOString(),active_seconds:state.active_seconds});running=true;last=performance.now();$('prior').disabled=true;$('start').disabled=true;$('pause').disabled=false;$('load').disabled=true;$('work').hidden=false;show(current);updateTime();message('Review is running. Aim for about four minutes per packet, with time for feedback.');persist();}
function revealDraft(){
 const id=packets[current].manifest.packet_id,a=state.packets[id],draft=suggestions.packets?.[id];
 if(!running||!a.source_read_at||a.revealed_at||!draft)return;
 a.before_reveal=structuredClone(a.fields);
 a.before_reveal_status={reviewed:a.reviewed,status:a.status};
 a.assistant_draft=structuredClone(draft);
 a.suggestion_set_id=suggestions.draft_set_id;a.suggestion_file_sha256=suggestions.file_sha256;
 a.revealed_at=new Date().toISOString();a.revealed_at_active_seconds=state.active_seconds;
 a.prefilled_fields=[];
 for(const f of fieldNames){const v=a.fields[f];if(['value','raw_text','availability','origin','evidence_input'].every(k=>!v[k])){a.fields[f]=Object.fromEntries(['value','raw_text','availability','origin','evidence_input'].map(k=>[k,draft.fields[f][k]]));a.prefilled_fields.push(f);}}
 a.reviewed=false;a.status='assistant_assisted_user_draft';
 state.events.push({type:'reveal',packet_id:id,at:a.revealed_at,active_seconds:state.active_seconds,suggestion_set_id:a.suggestion_set_id});
 persist();show(current);message('Draft revealed. Compare each answer with the source, then revise any disagreement.');
}
function show(index){current=index;const p=packets[index],id=p.manifest.packet_id,answer=state.packets[id];$('title').textContent=id+' · Source passage';$('admin').textContent='Accession '+p.manifest.accession+' · Packet version '+p.manifest.packet_version+' · Development data';const frame=document.createElement('iframe');frame.id='source';frame.title='Original source passages with one marked target';frame.setAttribute('sandbox','');frame.srcdoc=p.source_view;$('source').replaceWith(frame);$('locations').textContent=p.manifest.blocks.map(b=>b.block_id+': original lines '+b.line_start+'–'+b.line_end).join(' · ');$('fields').replaceChildren();
 $('suggestion-controls').hidden=!suggestions.packets?.[id];$('source-read').checked=!!answer.source_read_at;$('source-read').disabled=!!answer.revealed_at;
 $('reveal').disabled=!answer.source_read_at||!!answer.revealed_at;$('reveal').textContent=answer.revealed_at?'Draft revealed':'Reveal draft labels';
 $('draft-state').textContent=answer.revealed_at?'The boxes contain your answers or the filled draft. Differences from the draft appear below the boxes.':'Draft labels are hidden. Read the source before revealing them.';
 for(const f of fieldNames){const row=document.createElement('tr');const name=document.createElement('td');name.textContent=fieldLabels[f]||f.replaceAll('_',' ');row.append(name);
 for(const [prop,options] of [['value',null],['availability',availability],['origin',origins],['evidence_input',null]]){const cell=document.createElement('td'),control=document.createElement(options?'select':'textarea');control.setAttribute('aria-label',f+' '+prop);if(options){for(const v of options){const o=document.createElement('option');o.value=v;o.textContent=v||(prop==='origin'?'No origin':'Unreviewed');control.append(o);}}control.value=answer.fields[f][prop];control.oninput=()=>{answer.fields[f][prop]=control.value;if(prop==='value')answer.fields[f].raw_text=control.value;answer.status=answer.revealed_at?'assistant_assisted_user_draft':'user_draft';answer.reviewed=false;$('reviewed').checked=false;persist();};cell.append(control);
 if(answer.revealed_at&&answer.assistant_draft){const draft=answer.assistant_draft.fields[f];const hint=document.createElement('p');hint.className='small';hint.style.whiteSpace='pre-wrap';hint.textContent='Draft: '+(draft[prop]||'(blank)');hint.hidden=answer.fields[f][prop]===draft[prop];cell.append(hint);const saveInput=control.oninput;control.oninput=()=>{saveInput();hint.hidden=control.value===draft[prop];};if(prop==='value'){if(draft.raw_text&&draft.raw_text!==draft.value){const raw=document.createElement('p');raw.className='small';raw.textContent='Source wording: '+draft.raw_text;cell.append(raw);}if(draft.note){const note=document.createElement('p');note.className='small';note.textContent='Draft note: '+draft.note;cell.append(note);}}}
 row.append(cell);}$('fields').append(row);}
 $('feedback').value=answer.feedback;$('reviewed').checked=answer.reviewed;for(const [i,b]of [...$('nav').children].entries())b.setAttribute('aria-current',String(i===index));}
function exportDraft(){if(running)pause('Review paused and draft downloaded. Resume only if review time remains.');state.revision++;state.saved_at=new Date().toISOString();state.total_review_minutes=state.prior_review_minutes+state.active_seconds/60;state.status='user_draft_not_accepted_reference';state.review_workflow=Object.values(state.packets).some(a=>a.revealed_at)?'source_then_assistant_reveal':'source_first_manual';for(const a of Object.values(state.packets)){if(a.revealed_at&&a.assistant_draft){a.fields_changed_from_assistant=fieldNames.filter(f=>['value','availability','origin','evidence_input'].some(k=>a.fields[f][k]!==a.assistant_draft.fields[f][k]));}}state.completed_packets=Object.values(state.packets).filter(p=>p.reviewed).length;persist();const blob=new Blob([JSON.stringify(state,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='proxybench-calibration-'+state.session_id+'-v'+state.revision+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
function validateDraft(x){if(x.set_id!==state.set_id||x.schema!==state.schema||typeof x.session_id!=='string'||!Number.isFinite(x.active_seconds)||x.active_seconds<0||x.active_seconds>1800||!Number.isFinite(x.prior_review_minutes)||x.prior_review_minutes<0||x.prior_review_minutes>240||!Number.isInteger(x.revision)||!Array.isArray(x.events))throw Error('The draft has invalid session metadata.');for(const p of packets){const a=x.packets?.[p.manifest.packet_id];if(!a||a.source_sha256!==p.manifest.source_sha256||a.packet_version!==p.manifest.packet_version||typeof a.feedback!=='string')throw Error('The draft does not match this packet set.');for(const f of fieldNames){const v=a.fields?.[f];if(!v||typeof v.value!=='string'||typeof v.evidence_input!=='string'||!availability.includes(v.availability)||!origins.includes(v.origin))throw Error('The draft has an invalid field.');}
 if(a.revealed_at){const d=a.assistant_draft;if(!d||d.source_sha256!==p.manifest.source_sha256||d.packet_version!==p.manifest.packet_version||!a.before_reveal||typeof a.suggestion_set_id!=='string')throw Error('The saved reveal history does not match this packet.');for(const f of fieldNames){if(!d.fields?.[f]||typeof d.fields[f].value!=='string'||typeof d.fields[f].note!=='string'||!a.before_reveal[f])throw Error('The saved reveal history is incomplete.');}}
 }return x;}
for(const [i,p]of packets.entries()){const b=document.createElement('button');b.textContent=p.manifest.packet_id;b.onclick=()=>show(i);$('nav').append(b);}
$('feedback').oninput=()=>{state.packets[packets[current].manifest.packet_id].feedback=$('feedback').value;persist();};
$('reviewed').onchange=()=>{const a=state.packets[packets[current].manifest.packet_id];if($('reviewed').checked&&fieldNames.some(f=>!a.fields[f].availability)){message('Some fields remain unreviewed. Record their availability or leave this packet unfinished.');$('reviewed').checked=false;return;}a.reviewed=$('reviewed').checked;a.status='user_draft';persist();};
$('start').onclick=start;$('pause').onclick=()=>pause();$('save').onclick=exportDraft;
$('source-read').onchange=()=>{const a=state.packets[packets[current].manifest.packet_id];if(a.revealed_at)return;a.source_read_at=$('source-read').checked?new Date().toISOString():null;$('reveal').disabled=!a.source_read_at;persist();};
$('reveal').onclick=revealDraft;
$('load').onchange=async()=>{try{const file=$('load').files[0];if(!file)return;state=validateDraft(JSON.parse(await file.text()));$('prior').value=state.prior_review_minutes;updateTime();message('Draft restored. Resume the timer before reading or editing.');persist();}catch(e){message(e.message);}};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)pause('Review paused because this tab is hidden.');});
window.addEventListener('beforeunload',()=>{if(running)tick();persist();});
try{const saved=localStorage.getItem(key);if(saved){state=validateDraft(JSON.parse(saved));$('prior').value=state.prior_review_minutes;message('Browser draft restored. Resume the timer before reading or editing.');}}catch(e){message('No usable browser draft was restored. You can load a downloaded draft.');}
updateTime();setInterval(tick,1000);
</script></body></html>'''


def load_suggestions(path, packets):
    """Match suggestions to the exact packet input before adding them to the page."""
    raw = Path(path).read_bytes()
    drafts = json.loads(raw)
    expected = {p['manifest']['packet_id'] for p in packets}
    if set(drafts.get('packets', {})) != expected:
        raise ValueError('Assistant drafts do not match this packet set.')
    for p in packets:
        m = p['manifest']
        d = drafts['packets'][m['packet_id']]
        fingerprint = hashlib.sha256(json.dumps(m, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (d.get('source_sha256') != m['source_sha256']
                or d.get('packet_version') != m['packet_version']
                or d.get('packet_fingerprint') != fingerprint):
            raise ValueError('An assistant draft belongs to a different source or packet version.')
        if set(d.get('fields', {})) != set(FIELDS):
            raise ValueError('An assistant draft has an incomplete field checklist.')
        for field in d['fields'].values():
            if any(not isinstance(field.get(key), str) for key in ('value', 'raw_text', 'availability', 'origin', 'evidence_input', 'note')):
                raise ValueError('Assistant draft fields must contain review text.')
            if field['availability'] not in ('PRESENT', 'ABSENT_IN_CONTEXT', 'AMBIGUOUS', 'UNREADABLE', 'CONFLICTING', 'NOT_APPLICABLE'):
                raise ValueError('Unknown availability state in an assistant draft.')
            if field['origin'] not in ('', 'EXTRACTED', 'DERIVED', 'INFERRED'):
                raise ValueError('Unknown origin in an assistant draft.')
    drafts['file_sha256'] = hashlib.sha256(raw).hexdigest()
    return drafts


def write_review(root, *, draft_path=None):
    root = Path(root)
    directory = root / 'data/packets/calibration'
    packets = json.loads((directory / 'packet-set.json').read_text(encoding='utf-8'))
    payload = json.dumps(packets, ensure_ascii=False).replace('<', '\\u003c')
    drafts = load_suggestions(draft_path, packets) if draft_path else {'packets': {}}
    draft_payload = json.dumps(drafts, ensure_ascii=False).replace('<', '\\u003c')
    page = TEMPLATE.replace('__PACKETS__', payload).replace('__FIELDS__', json.dumps(FIELDS)).replace('__DRAFTS__', draft_payload)
    (directory / 'index.html').write_text(page, encoding='utf-8')
    return directory / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drafts', type=Path, help='Preserved assistant draft labels for this packet set.')
    args = parser.parse_args()
    print(write_review(Path.cwd(), draft_path=args.drafts))
