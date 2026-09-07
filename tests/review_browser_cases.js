// Run against a newly rendered synthetic review page in a same-origin frame.
async function runReviewBrowserCases(w) {
  const require = (condition, message) => {if (!condition) throw Error(message);};
  const clone = value => JSON.parse(JSON.stringify(value));
  const current = () => JSON.parse(w.eval('JSON.stringify(state)'));
  const base = current();
  const id = Object.keys(base.packets)[0];
  const input = w.document.getElementById('load');
  async function load(draft) {
    const transfer = new w.DataTransfer();
    transfer.items.add(new w.File([JSON.stringify(draft)], 'synthetic.json', {type:'application/json'}));
    input.files = transfer.files;
    await input.onchange();
  }
  const cases = [];
  for (const [name, mutate] of [
    ['foreign packet', x => {x.packets.foreign = {reviewed:true};}],
    ['missing packet', x => {delete x.packets[id];}],
    ['incomplete reviewed packet', x => {x.packets[id].reviewed = true;}],
    ['nonboolean completion', x => {x.packets[id].reviewed = 'true';}],
    ['missing correction reason', x => {
      const a=x.packets[id]; a.reviewed=true; a.correction_history=[{reason:null}];
      Object.values(a.fields).forEach(f=>f.availability='ABSENT_IN_CONTEXT');
    }]
  ]) {
    const invalid = clone(base);
    mutate(invalid);
    const before = JSON.stringify(current());
    await load(invalid);
    require(JSON.stringify(current()) === before, name + ': state changed');
    require(!w.document.getElementById('import-warning').hidden, name + ': warning missing');
    cases.push(name);
  }
  const valid = clone(base);
  valid.session_id = 'synthetic-completed-resume';
  const answer = valid.packets[id];
  Object.values(answer.fields).forEach(f=>f.availability='ABSENT_IN_CONTEXT');
  answer.reviewed = true;
  answer.correction_history = [{reason:'Synthetic source correction.'}];
  await load(valid);
  require(current().session_id === valid.session_id, 'Valid draft was not restored');
  require(w.document.getElementById('import-warning').hidden, 'Valid import did not clear warning');
  w.document.getElementById('start').click();
  require(w.document.getElementById('reviewed').checked, 'Completion did not render after resume');
  let exported;
  w.URL.createObjectURL = blob => {exported=blob; return 'blob:synthetic-test';};
  w.URL.revokeObjectURL = () => {};
  w.HTMLAnchorElement.prototype.click = function() {};
  w.document.getElementById('save').click();
  const saved = JSON.parse(await exported.text());
  require(saved.completed_packets === 1, 'Export completion count differs');
  require(Object.keys(saved.packets).length === Object.keys(base.packets).length, 'Export packet inventory differs');
  require(saved.status === 'user_draft_not_accepted_reference', 'Export acceptance state changed');
  require(w.eval('running') === false, 'Export did not pause review');
  // Exercise the separate browser-storage restore path with a foreign packet.
  const invalidStored = clone(valid);
  invalidStored.packets.foreign = {reviewed:true};
  w.localStorage.setItem(w.eval('key'), JSON.stringify(invalidStored));
  w.eval("state.packets.foreign={reviewed:true};");
  await new Promise(resolve => {w.frameElement.addEventListener('load', resolve, {once:true}); w.location.reload();});
  require(Object.keys(current().packets).length === Object.keys(base.packets).length, 'Storage imported foreign packet');
  require(current().session_id !== valid.session_id, 'Invalid browser draft replaced initial state');
  cases.push('valid resume', 'rendered completion', 'export count and draft status', 'save pauses timer', 'invalid browser storage');
  return {status:'PASS', cases, native_file_dialog_exercised:false, native_download_exercised:false};
}
