"""Prepare and accept a source-bound test-only dataset."""

from datetime import date
from pathlib import Path
import re

from proxybench.annotation.bindings import checked_file, review_binding
from proxybench.annotation.historical import VIEW_VERSION, literal_support, prepare_historical
from proxybench.annotation.review import write_review
from proxybench.runstate import atomic_json, atomic_text
from proxybench.sources.holdout import load_ledger, protect_sources, source_cik
from proxybench.training.dataset import COORDINATES, _read_release, project_file
from proxybench.training.labels import dumps, from_review, read_json, sha

SCHEMA = 'test-dataset-v1'
FILES = {'test': 'test-examples.jsonl'}


def snapshot_test_baseline(root, output):
    """Retain the exact existing source inventory before test acquisition."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, original in (('prior-dataset-manifest.json', 'data/training-dataset/dataset-manifest.json'),
                           ('prior-source-manifest.json', 'data/source-manifest.json')):
        destination = output / name
        if destination.exists():
            raise FileExistsError('Test baseline already exists')
        atomic_text(destination, (root / original).read_text(encoding='utf-8'))
    return dict(schema='test-selection-audit-v1',
                prior_dataset=file_binding(output / 'prior-dataset-manifest.json', root),
                prior_inventory=file_binding(output / 'prior-source-manifest.json', root), candidates=[])


def file_binding(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path.read_bytes()))


def validate_protocol(protocol):
    years = protocol.get('filing_years')
    if (protocol.get('schema') != 'test-protocol-v1' or not isinstance(years, list) or not years
            or any(type(y) is not int for y in years) or len(set(years)) != len(years)
            or protocol.get('examples') != len(years)
            or not set(protocol.get('standardized_years', [])) <= set(years)
            or not isinstance(protocol.get('scorer_version'), str) or not protocol['scorer_version'].strip()):
        raise ValueError('Invalid test selection protocol')


def validate_selections(root, selections, protocol, audit, *, retained=None):
    """Enforce reviewed year slots and known project exposure before labeling."""
    validate_protocol(protocol)
    if len(selections) != protocol['examples']:
        raise ValueError('Test count differs from the reviewed protocol')
    baseline = read_json(test_metadata(root, audit['prior_dataset'], retained))
    inventory = read_json(test_metadata(root, audit['prior_inventory'], retained))
    validate_family_audit(root, selections, protocol, audit, baseline, inventory, retained=retained)
    comparisons = None
    if 'source_comparisons' in audit:
        comparisons = read_json(test_metadata(root, audit['source_comparisons'], retained))
        if (comparisons.get('schema') != 'test-source-comparison-v1'
                or comparisons.get('baseline_inventory') != audit['prior_inventory']
                or comparisons.get('prior_dataset') != audit['prior_dataset']
                or comparisons.get('family_audit') != audit.get('family_audit')
                or set(comparisons.get('records', {})) != {s['packet_id'] for s in selections}):
            raise ValueError('Source comparisons differ from the test audit and packet set')
    for key in ('agent_review', 'token_audit'):
        if key in audit:
            test_metadata(root, audit[key], retained)
    # The retained dataset cannot change between eligibility audit and acceptance.
    if retained is None and sha((Path(root) / 'data/training-dataset/dataset-manifest.json').read_bytes()) != audit['prior_dataset']['sha256']:
        raise ValueError('Prior dataset changed after the test-source audit')
    prior_accessions = set(baseline['assignments'])
    prior_groups = {a['group_id'] for a in baseline['assignments'].values()}
    prior_hashes = {s['sha256'] for s in inventory['sources']}
    prior_ciks = set()
    if comparisons is not None:
        for recovered in comparisons.get('recovered_prior_sources', []):
            if retained is None:
                raw = project_file(root, recovered['source'])
                header = raw[:10000].decode('latin-1')
                accession = re.search(r'ACCESSION NUMBER:\s*(\S+)', header)
                cik = re.search(r'CENTRAL INDEX KEY:\s*(\d+)', header)
                if not accession or not cik or accession[1] != recovered['accession'] or str(int(cik[1])) != recovered['cik']:
                    raise ValueError('Recovered prior filing differs from its audited identity')
            if recovered['accession'] not in baseline['assignments']:
                raise ValueError('Recovered prior filing differs from its audited identity')
            prior_accessions.add(recovered['accession'])
            prior_hashes.add(recovered['source']['sha256'])
            prior_ciks.add(recovered['cik'])
    for source in inventory['sources']:
        for location in source['locations']:
            prior_accessions.add(location['accession'])
            cik = source_cik(dict(cik=location.get('cik'), source_url=location.get('sec_url', '')))
            if cik:
                prior_ciks.add(cik)
    years, accessions, groups, hashes, ciks = [], set(), set(), set(), set()
    current_inventory = test_inventory(root, retained)
    for selection in selections:
        if comparisons is not None:
            comparison = comparisons['records'][selection['packet_id']]
            if any(comparison.get(key) != selection[key] for key in ('accession', 'source_path', 'source_sha256')) or (
                    comparison.get('target_bytes') != selection['target']):
                raise ValueError('Source comparison describes another selected source or target')
        if selection.get('split') != 'test' or not Path(selection['source_path']).is_relative_to('data/raw/test'):
            raise ValueError('Test selections require isolated test source storage')
        accession, group, digest = selection['accession'], selection['group_id'], selection['source_sha256']
        cik = source_cik(selection)
        admitted = next((s for s in current_inventory['sources'] if s['path'] == selection['source_path']
                         and s['sha256'] == digest), None)
        if admitted is None or not any(
                loc['accession'] == accession and loc.get('filing_date') == selection.get('filing_date')
                and loc['sec_url'] == selection.get('source_url')
                and source_cik(dict(cik=loc.get('cik'), source_url=loc['sec_url'])) == cik
                for loc in admitted['locations']):
            raise ValueError('Test selection differs from its admitted filing identity')
        for context in selection.get('context_sources', []):
            entry = next((s for s in current_inventory['sources'] if s['path'] == context['source_path']
                          and s['sha256'] == context['source_sha256']), None)
            if (not Path(context['source_path']).is_relative_to('data/raw/test')
                    or context['source_sha256'] in prior_hashes or entry is None
                    or not any(loc['accession'] == accession and loc['sec_url'] == context.get('source_url')
                               and source_cik(dict(cik=loc.get('cik'), source_url=loc['sec_url'])) == cik
                               for loc in entry['locations'])):
                raise ValueError('Test context differs from its admitted filing or prior-source audit')
        if (accession in prior_accessions or group in prior_groups or digest in prior_hashes or cik in prior_ciks):
            raise ValueError('Test source overlaps known project sources or exposure')
        if accession in accessions or group in groups or digest in hashes or cik in ciks:
            raise ValueError('Test records must use distinct filings and source families')
        if not cik or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
            raise ValueError('Test filing needs a registrant and accession identity')
        year = date.fromisoformat(selection['filing_date']).year
        if int(accession[11:13]) != year % 100:
            raise ValueError('Filing year differs from accession year')
        if year in protocol.get('standardized_years', []):
            evidence = selection.get('standardization_evidence')
            if not selection.get('standardized_format') or not isinstance(evidence, list) or not evidence:
                raise ValueError('Standardized filing year requires reviewed source evidence')
            packet = prepare_historical(root, selection)
            cells = [re.sub(r'\s+', ' ', c['text']).strip().casefold()
                     for b in packet['manifest']['blocks'] for c in b['cells']]
            if any(not isinstance(p, str) or not p.strip() or
                   not any(re.sub(r'\s+', ' ', p).strip().casefold() in c for c in cells) for p in evidence):
                raise ValueError('Standardization evidence is absent from supplied source cells')
        if not selection.get('independence_review'):
            raise ValueError('Test source needs a reviewed amendment and copied-content audit')
        years.append(year)
        accessions.add(accession)
        groups.add(group)
        hashes.add(digest)
        ciks.add(cik)
    if sorted(years) != sorted(protocol['filing_years']):
        raise ValueError('Test filing years differ from the reviewed protocol')


def validate_family_audit(root, selections, protocol, audit, baseline, inventory, *, retained=None):
    """Compare reviewed provider identities, not only registrant identities."""
    if 'family_audit' not in audit:
        if protocol.get('family_audit_required'):
            raise ValueError('Test preparation requires a reviewed provider-family audit')
        return
    families = read_json(test_metadata(root, audit['family_audit'], retained))
    if (families.get('schema') != 'test-family-audit-v1'
            or families.get('prior_dataset') != audit['prior_dataset']
            or families.get('prior_inventory') != audit['prior_inventory']):
        raise ValueError('Provider-family audit differs from the test baseline')
    sources = families.get('baseline_sources', [])
    expected = {(s['path'], s['sha256']) for s in inventory['sources']}
    if len(sources) != len(expected) or {(s['path'], s['sha256']) for s in sources} != expected:
        raise ValueError('Provider-family audit does not cover all prior source files')
    assignments = families.get('baseline_assignments', {})
    if set(assignments) != set(baseline['assignments']):
        raise ValueError('Provider-family audit does not cover all prior source assignments')
    known = set()
    for record in sources + list(assignments.values()):
        ids = record.get('family_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(f, str) or not re.fullmatch(r'family-[a-z0-9]+(?:-[a-z0-9]+)*', f) for f in ids):
            raise ValueError('Prior provider-family identity is incomplete')
        # Prior source evidence was checked before acceptance. Its exact audit is
        # retained, so finalized loading does not require historical raw files.
        check_family_evidence(root, record.get('evidence', []), frozen_prior=retained is not None)
        known.update(ids)
    admitted = test_inventory(root, retained)
    seen = set()
    for selection in selections:
        record = families.get('candidates', {}).get(selection['accession'], {})
        family = selection.get('family_id')
        if (not isinstance(family, str) or not re.fullmatch(r'family-[a-z0-9]+(?:-[a-z0-9]+)*', family)
                or record.get('family_id') != family or record.get('cik') != source_cik(selection)
                or selection['source_sha256'] not in record.get('source_sha256', [])):
            raise ValueError('Test selection differs from its reviewed provider family')
        if family in known or family in seen:
            raise ValueError('Test provider family overlaps prior sources or another test record')
        evidence = record.get('evidence', [])
        for item in evidence:
            if not any(entry['path'] == item['source']['path'] and entry['sha256'] == item['source']['sha256']
                       and any(loc['accession'] == selection['accession']
                               and source_cik(dict(cik=loc.get('cik'), source_url=loc['sec_url'])) == source_cik(selection)
                               for loc in entry['locations']) for entry in admitted['sources']):
                raise ValueError('Provider-family evidence belongs to another filing')
        check_family_evidence(root, evidence)
        seen.add(family)


def check_family_evidence(root, evidence, *, frozen_prior=False):
    if not evidence:
        raise ValueError('Provider family requires original source evidence')
    for item in evidence:
        a, b = item['bytes']
        if type(a) is not int or type(b) is not int or not 0 <= a < b:
            raise ValueError('Provider-family evidence differs from its original source range')
        if frozen_prior:
            if any(not re.fullmatch(r'[a-f0-9]{64}', digest) for digest in
                   (item['source']['sha256'], item['slice_sha256'])):
                raise ValueError('Prior provider-family evidence has an invalid hash')
            continue
        raw = project_file(root, item['source'])
        if b > len(raw) or sha(raw[a:b]) != item['slice_sha256']:
            raise ValueError('Provider-family evidence differs from its original source range')


def prepare_test_review(root, selections, output, *, protocol_path, audit_path,
                        prompt_path='configs/model-system-prompt.txt',
                        label_contract_path='docs/label-contract.md', draft_path=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = read_json((root / protocol_path).read_bytes())
    audit = read_json((root / audit_path).read_bytes())
    validate_selections(root, selections, protocol, audit)
    packets = [prepare_historical(root, s) for s in selections]
    ids = [p['manifest']['packet_id'] for p in packets]
    if len(set(ids)) != len(ids):
        raise ValueError('Test packet IDs must be unique')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'packet-set.json', packets)
    atomic_text(output / 'system-prompt.txt', (root / prompt_path).read_text(encoding='utf-8'))
    protect_sources(root, selections)
    ledger = load_ledger(root)
    accessions = {s['accession'] for s in selections}
    atomic_json(output / 'protection.json', dict(schema=ledger['schema'], sources=[
        s for s in ledger['sources'] if s['accession'] in accessions]))
    protection = file_binding(output / 'protection.json', root)
    preparation = dict(schema='test-preparation-v1', status='PENDING_ACCEPTANCE', packet_ids=ids,
                       system_prompt=file_binding(root / prompt_path, root),
                       label_contract=file_binding(root / label_contract_path, root),
                       protocol=file_binding(root / protocol_path, root),
                       audit=file_binding(root / audit_path, root), protection=protection,
                       packets=file_binding(output / 'packet-set.json', root))
    atomic_json(output / 'preparation.json', preparation)
    return write_review(root, packet_directory=output, draft_path=draft_path,
                        training_contract=(root / label_contract_path).read_text(encoding='utf-8'))


def checked_preparation(root, path):
    preparation = read_json(Path(path).read_bytes())
    return check_preparation_content(root, preparation)


def check_preparation_content(root, preparation, *, retained=None):
    """Check a preparation document, including one retained in the manifest."""
    if preparation.get('schema') != 'test-preparation-v1':
        raise ValueError('Unsupported test preparation')
    for key in ('system_prompt', 'label_contract', 'protocol'):
        project_file(root, preparation[key])
    for key in ('audit', 'protection', 'packets'):
        test_metadata(root, preparation[key], retained)
    protected = read_json(test_metadata(root, preparation['protection'], retained))
    ledger = load_ledger(root)
    for source in protected['sources']:
        present = next((s for s in ledger['sources'] if s['accession'] == source['accession']), None)
        if (present is None or present['group_id'] != source['group_id'] or present.get('cik') != source.get('cik')
                or (source.get('family_id') and present.get('family_id') != source['family_id'])
                or not set(source['sha256']) <= set(present['sha256'])
                or not set(source.get('aliases', [])) <= set(present.get('aliases', []))):
            raise ValueError('Protected test reservations changed or disappeared')
    packets = read_json(test_metadata(root, preparation['packets'], retained))
    for packet in packets:
        selection = packet['manifest']
        if not any(s['accession'] == selection['accession'] and s['group_id'] == selection['group_id']
                   and s.get('cik') == source_cik(selection) and selection['source_sha256'] in s['sha256']
                   for s in protected['sources']):
            raise ValueError('Test packet lacks its protected source reservation')
    return preparation


def accepted_rows(root, preparation, review, approval, review_raw, preparation_raw, *, retained=None):
    packets = read_json(test_metadata(root, preparation['packets'], retained))
    ids = preparation['packet_ids']
    if (approval.get('decision') != 'ACCEPTED' or not approval.get('reviewer') or not approval.get('accepted_at')
            or approval.get('export_sha256') != sha(review_raw)
            or approval.get('preparation_sha256') != sha(preparation_raw)
            or approval.get('accepted_packet_ids') != ids):
        raise ValueError('Explicit test acceptance must bind the exact review, preparation, and ordered packet IDs')
    if review.get('schema') != 'training-review-v1' or set(review.get('packets', {})) != set(ids):
        raise ValueError('Test review does not match its ordered packet set')
    if [p['manifest']['packet_id'] for p in packets] != ids or len(set(ids)) != len(ids):
        raise ValueError('Test packet order or identities changed')
    rows, metadata, selections = [], [], []
    prompt = project_file(root, preparation['system_prompt']).decode('utf-8')
    for packet in packets:
        selection = {k: v for k, v in packet['manifest'].items()
                     if k not in {'blocks', 'packet_version', 'view_version', 'target_id'}}
        rebuilt = prepare_historical(root, selection)
        if rebuilt != {k: packet[k] for k in ('manifest', 'source_view', 'model_input')}:
            raise ValueError('Test source packet changed after review')
        answer = review['packets'][selection['packet_id']]
        if answer.get('reviewed') is not True or answer.get('input_binding') != review_binding(rebuilt):
            raise ValueError('Test reference needs completed review against its exact source')
        label = from_review(answer['fields'])
        literal_support(label, rebuilt)
        row = dict(messages=[dict(role='system', content=prompt), dict(role='user', content=rebuilt['model_input']),
                             dict(role='assistant', content=dumps(label))])
        metadata.append(dict(split='test', position=len(rows), row_sha256=sha(dumps(row).encode()), selection=selection))
        rows.append(row)
        selections.append(selection)
    protocol = read_json(project_file(root, preparation['protocol']))
    audit = read_json(test_metadata(root, preparation['audit'], retained))
    validate_selections(root, selections, protocol, audit, retained=retained)
    return rows, metadata


def accept_test_review(root, review_dir, review_path, approval_path, output):
    """Write accepted references only after exact user approval."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if (output / 'dataset-manifest.json').exists() or (output / FILES['test']).exists():
        raise FileExistsError('An accepted test dataset already exists')
    preparation_path = Path(review_dir) / 'preparation.json'
    preparation_raw = preparation_path.read_bytes()
    preparation = checked_preparation(root, preparation_path)
    review_raw, approval_raw = Path(review_path).read_bytes(), Path(approval_path).read_bytes()
    rows, examples = accepted_rows(root, preparation, read_json(review_raw), read_json(approval_raw),
                                  review_raw, preparation_raw)
    output.mkdir(parents=True, exist_ok=True)
    raw = ''.join(dumps(row) + '\n' for row in rows)
    atomic_text(output / FILES['test'], raw)
    assignments = {e['selection']['accession']: dict(split='test', group_id=e['selection']['group_id'],
                                                   development_exposed=False) for e in examples}
    manifest = dict(schema=SCHEMA, status='COMPLETE', coordinate_convention=COORDINATES,
                    rendering_version=VIEW_VERSION, files={'test': dict(path=FILES['test'], sha256=sha(raw.encode()), examples=len(rows))},
                    examples=examples, assignments=assignments, system_prompt=preparation['system_prompt'],
                    label_contract=preparation['label_contract'], evidence=freeze_test_evidence(root, preparation),
                    acceptance={key: dict(content=raw.decode('utf-8'), sha256=sha(raw))
                                for key, raw in (('review', review_raw), ('approval', approval_raw),
                                                 ('preparation', preparation_raw))})
    atomic_json(output / 'dataset-manifest.json', manifest)
    read_test_release(output, project_root=root)
    return manifest


def read_test_release(path, *, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    path = Path(path).resolve()
    rows, manifest = _read_release(path, project_root=root, schema=SCHEMA, files=FILES)
    receipt = manifest['acceptance']
    review_raw = acceptance_content(receipt['review'])
    approval_raw = acceptance_content(receipt['approval'])
    preparation_raw = acceptance_content(receipt['preparation'])
    retained = manifest['evidence']
    check_retained_evidence(retained)
    preparation = check_preparation_content(root, read_json(preparation_raw), retained=retained)
    if any(manifest[key] != preparation[key] for key in ('system_prompt', 'label_contract')):
        raise ValueError('Test manifest policy differs from accepted preparation')
    expected_rows, expected_examples = accepted_rows(root, preparation, read_json(review_raw), read_json(approval_raw),
                                                   review_raw, preparation_raw, retained=retained)
    if (rows['test'] != expected_rows or len(manifest['examples']) != len(expected_examples) or any(
            {k: v for k, v in item.items() if k != 'packet'} != expected
            for item, expected in zip(manifest['examples'], expected_examples))):
        raise ValueError('Test dataset differs from its accepted review')
    return rows, manifest


def acceptance_content(receipt):
    """Retain exact acceptance bytes without extra files in the dataset folder."""
    content = receipt.get('content')
    if not isinstance(content, str) or sha(content.encode('utf-8')) != receipt.get('sha256'):
        raise ValueError('Retained test acceptance content has a hash mismatch')
    return content.encode('utf-8')


def test_metadata(root, binding, retained=None):
    """Resolve accepted metadata only, without a preparation-file fallback."""
    if retained is None:
        return project_file(root, binding)
    path = Path(binding['path'])
    if path.is_absolute() or '..' in path.parts or path.is_relative_to('data/raw'):
        raise ValueError('Retained test metadata cannot replace raw sources or leave the project')
    receipt = retained['documents'].get(binding['path'])
    if receipt is None or receipt.get('sha256') != binding['sha256']:
        raise ValueError('Required accepted test metadata is missing or has a hash mismatch')
    return acceptance_content(receipt)


def test_inventory(root, retained=None):
    if retained is None:
        return read_json((Path(root) / 'data/source-manifest.json').read_bytes())
    return read_json(acceptance_content(retained['admitted_inventory']))


def freeze_test_evidence(root, preparation):
    """Retain accepted metadata, not temporary files or duplicate raw sources."""
    documents = {}

    def keep(binding):
        raw = project_file(root, binding)
        documents[binding['path']] = dict(content=raw.decode('utf-8'), sha256=sha(raw))
        return read_json(raw)

    audit = keep(preparation['audit'])
    for key in ('protection', 'packets'):
        keep(preparation[key])
    for key in ('prior_dataset', 'prior_inventory', 'family_audit', 'source_comparisons', 'agent_review', 'token_audit'):
        if key in audit:
            keep(audit[key])
    raw = (Path(root) / 'data/source-manifest.json').read_bytes()
    return dict(schema='test-evidence-v1', documents=documents,
                admitted_inventory=dict(content=raw.decode('utf-8'), sha256=sha(raw)))


def check_retained_evidence(retained):
    if retained.get('schema') != 'test-evidence-v1' or not isinstance(retained.get('documents'), dict):
        raise ValueError('Unsupported retained test evidence')
    for path, receipt in retained['documents'].items():
        candidate = Path(path)
        if candidate.is_absolute() or '..' in candidate.parts or candidate.is_relative_to('data/raw'):
            raise ValueError('Retained test metadata cannot replace raw sources or leave the project')
        acceptance_content(receipt)
    acceptance_content(retained['admitted_inventory'])
