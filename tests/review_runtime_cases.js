// Execute with the actual helper and validation functions from review.TEMPLATE.
// The host supplies JavaScript evaluation; no browser or network access is used.
function runReviewCases(helpers, validation) {
  const binding = {binding_version: 'review-binding-v1', manifest_sha256: 'manifest',
    review_view_sha256: 'view', model_input_sha256: 'input'};
  const packets = [{manifest: {packet_id: 'synthetic-1', packet_version: 1, source_sha256: 'source'}, review_binding: binding}];
  const fields = {issuer_name: {value: 'Original value', raw_text: 'Original source wording',
    availability: 'PRESENT', origin: 'EXTRACTED', evidence_input: 'B1: source'}};
  const state = {set_id: 'synthetic', schema: 'calibration-draft-v2'};
  const draft = {...state, session_id: 'session', active_seconds: 5, prior_review_minutes: 0,
    revision: 1, events: [], packets: {'synthetic-1': {packet_version: 1, source_sha256: 'source',
      input_binding: binding, fields, feedback: '', reviewed: false, correction_history: []}}};
  const api = new Function('packets', 'state', 'fieldNames', 'availability', 'origins',
    helpers + validation + 'return {validateDraft, updateAnswer};')(packets, state, ['issuer_name'],
      ['', 'PRESENT', 'ABSENT_IN_CONTEXT'], ['', 'EXTRACTED', 'DERIVED', 'INFERRED']);
  const clone = value => JSON.parse(JSON.stringify(value));
  const require = (condition, message) => {if (!condition) throw Error(message);};
  const reject = value => {
    let rejected = false;
    try {api.validateDraft(value);} catch (error) {rejected = true;}
    require(rejected, 'A stale or malformed review draft was accepted.');
  };
  api.validateDraft(clone(draft));
  for (const key of ['manifest_sha256', 'review_view_sha256', 'model_input_sha256']) {
    const changed = clone(draft);
    changed.packets['synthetic-1'].input_binding[key] = 'changed';
    reject(changed);
  }
  for (const mutate of [
    x => {x.packets.foreign = {reviewed: true};},
    x => {delete x.packets['synthetic-1'];},
    x => {x.packets = [];},
    x => {x.packets['synthetic-1'].reviewed = 'true';},
    x => {const a=x.packets['synthetic-1']; a.reviewed=true; a.fields.issuer_name.availability='';},
    x => {const a=x.packets['synthetic-1']; a.reviewed=true; a.correction_history=[{reason:null}];},
    x => {const a=x.packets['synthetic-1']; a.reviewed=true; a.correction_history=[{reason:'  '}];}
  ]) {
    const changed = clone(draft);
    mutate(changed);
    reject(changed);
  }
  const completed = clone(draft);
  completed.packets['synthetic-1'].reviewed = true;
  completed.packets['synthetic-1'].correction_history = [{reason:'Corrected from the source.'}];
  api.validateDraft(completed);
  const legacy = clone(draft);
  delete legacy.packets['synthetic-1'].input_binding;
  reject(legacy);
  const edited = clone(draft);
  const answer = edited.packets['synthetic-1'];
  api.updateAnswer(answer, 'issuer_name', 'value', 'Corrected normalized value');
  require(answer.fields.issuer_name.raw_text === 'Original source wording', 'Value editing overwrote source wording.');
  require(answer.correction_history.length === 1, 'Correction history is missing.');
  require(answer.correction_history[0].old_value === 'Original value', 'Original answer was lost.');
  require(draft.packets['synthetic-1'].fields.issuer_name.value === 'Original value', 'Earlier draft was modified.');
  api.validateDraft(edited);
  const revealed = clone(draft);
  const a = revealed.packets['synthetic-1'];
  a.revealed_at = '2026-09-07';
  a.suggestion_set_id = 'synthetic-suggestions';
  a.before_reveal = clone(fields);
  a.assistant_draft = {packet_version: 1, source_sha256: 'source', input_binding: clone(binding),
    fields: {issuer_name: {...fields.issuer_name, note: ''}}};
  api.validateDraft(revealed);
  a.assistant_draft.input_binding.model_input_sha256 = 'stale';
  reject(revealed);
  return {status: 'PASS', cases: ['valid resume', 'exact packet inventory', 'boolean review state', 'completion availability', 'correction reasons', 'valid completion', 'changed manifest', 'changed view', 'changed model input',
    'legacy binding refusal', 'raw wording preservation', 'correction history', 'prior version preservation',
    'valid reveal history', 'stale revealed suggestion refusal'], browser_ui_exercised: false};
}
