"""Test-only review, acceptance, source protection, and inference boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from proxybench.annotation.bindings import review_binding
from proxybench.annotation.historical import prepare_historical
from proxybench.annotation.preparation import prepare_review
from proxybench.annotation.testing import (accept_test_review, prepare_test_review, read_test_release,
                                          snapshot_test_baseline, validate_selections, file_binding)
from proxybench.evaluation.workflow import prepare_inputs, check_test_scorer, report
from proxybench.runstate import atomic_json
from proxybench.sources.holdout import load_ledger, reject_protected_sources
from proxybench.sources.inventory import import_source
from proxybench.training.dataset import read_release
from proxybench.training.labels import TYPES, dumps, sha, to_review


class TestingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'configs').mkdir()
        (self.root / 'configs/model-system-prompt.txt').write_text('Exact policy\n')
        (self.root / 'docs').mkdir()
        (self.root / 'docs/label-contract.md').write_text('Exact contract\n')
        atomic_json(self.root / 'data/training-dataset/dataset-manifest.json', dict(assignments={
            '0000000999-21-000001': dict(split='development', group_id='old', development_exposed=False)}))
        atomic_json(self.root / 'data/source-manifest.json', dict(schema='source-inventory-v1', sources=[]))
        self.audit = snapshot_test_baseline(self.root, self.root / 'audit')
        atomic_json(self.root / 'audit.json', self.audit)
        self.protocol = dict(schema='test-protocol-v1', examples=2, filing_years=[2023, 2024],
                             standardized_years=[2024], scorer_version='source-cells-v1')
        atomic_json(self.root / 'protocol.json', self.protocol)
        self.selections = []
        for year, cik in ((2023, 100), (2024, 101)):
            accession = f'{cik:010}-{year % 100:02}-000001'
            raw = b'<p>Shares on loan</p><tr><td>01</td><td>For</td></tr>'
            source = self.root / f'{year}.htm'
            source.write_bytes(raw + f'<!--{year}-->'.encode())
            url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace("-", "")}/source.htm'
            entry = import_source(source, self.root, sec_url=url, accession=accession,
                                  friendly_filename=f'{year}.htm', storage='test', complete=True,
                                  filing_date=f'{year}-08-20', cik=str(cik), form='N-PX')
            start = raw.index(b'<tr>')
            self.selections.append(dict(packet_id=str(year), accession=accession, source_path=entry['path'],
                source_sha256=entry['sha256'], source_url=url, cik=str(cik), filing_date=f'{year}-08-20',
                group_id=f'new-{cik}', split='test', encoding='utf-8', spans=[[0,start,'block'],[start,len(raw),'row']],
                target=[start,len(raw)], boundary_review='One complete subject', reviewer='Synthetic reviewer',
                standardized_format=year == 2024, standardization_evidence=['Shares on loan'] if year == 2024 else [],
                independence_review='Synthetic distinct sources and amendment families'))
        self.review = self.root / 'custom-review'
        self.output = self.root / 'test-dataset'

    def prepare(self):
        return prepare_test_review(self.root, self.selections, self.review, protocol_path='protocol.json', audit_path='audit.json')

    def receipt(self):
        self.prepare()
        packets = json.loads((self.review / 'packet-set.json').read_text())
        label = {'fields': {k:dict(value=None,availability='ABSENT_IN_CONTEXT',origin=None,raw_text=None) for k in TYPES}}
        review = dict(schema='training-review-v1', packets={p['manifest']['packet_id']:dict(
            reviewed=True, input_binding=review_binding(p), fields=to_review(label)) for p in packets})
        path = self.root / 'review.json'
        atomic_json(path, review)
        approval = dict(decision='ACCEPTED', reviewer='Synthetic user', accepted_at='2026-09-29',
                        accepted_packet_ids=[s['packet_id'] for s in self.selections],
                        export_sha256=sha(path.read_bytes()), preparation_sha256=sha((self.review / 'preparation.json').read_bytes()))
        receipt = self.root / 'approval.json'
        atomic_json(receipt, approval)
        return path, receipt

    def test_changed_contract_or_review_cannot_be_accepted(self):
        path, receipt = self.receipt()
        (self.root / 'docs/label-contract.md').write_text('Changed contract')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            accept_test_review(self.root, self.review, path, receipt, self.output)
        (self.root / 'docs/label-contract.md').write_text('Exact contract\n')
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Explicit test acceptance'):
            accept_test_review(self.root, self.review, path, receipt, self.output)

    def test_changed_order_and_stale_source_binding_are_refused(self):
        path, receipt = self.receipt()
        approval = json.loads(receipt.read_text())
        approval['accepted_packet_ids'].reverse()
        atomic_json(receipt, approval)
        with self.assertRaisesRegex(ValueError, 'ordered packet IDs'):
            accept_test_review(self.root, self.review, path, receipt, self.output)
        approval['accepted_packet_ids'].reverse()
        review = json.loads(path.read_text())
        review['packets']['2023']['input_binding']['model_input_sha256'] = 'stale'
        atomic_json(path, review)
        approval['export_sha256'] = sha(path.read_bytes())
        atomic_json(receipt, approval)
        with self.assertRaisesRegex(ValueError, 'exact source'):
            accept_test_review(self.root, self.review, path, receipt, self.output)

    def test_distinct_years_sources_and_standardization_evidence(self):
        bad = deepcopy(self.selections)
        bad[1] = deepcopy(bad[0])
        with self.assertRaisesRegex(ValueError, 'distinct filings'):
            validate_selections(self.root, bad, self.protocol, self.audit)
        bad = deepcopy(self.selections)
        bad[1]['standardization_evidence'] = ['Imaginary standardized field']
        with self.assertRaisesRegex(ValueError, 'absent from supplied source'):
            validate_selections(self.root, bad, self.protocol, self.audit)
        bad[1]['cik'] = '100'
        with self.assertRaisesRegex(ValueError, 'source URL'):
            validate_selections(self.root, bad, self.protocol, self.audit)
        bad = deepcopy(self.selections)
        bad[0]['filing_date'] = '2024-08-20'
        with self.assertRaisesRegex(ValueError, 'admitted filing identity'):
            validate_selections(self.root, bad, self.protocol, self.audit)

    def test_known_development_family_without_exposure_flag_is_excluded(self):
        bad = deepcopy(self.selections)
        bad[0]['group_id'] = 'old'
        with self.assertRaisesRegex(ValueError, 'known project'):
            validate_selections(self.root, bad, self.protocol, self.audit)

    def test_alias_hash_and_registrant_cannot_bypass_protection(self):
        self.prepare()
        candidate = dict(self.selections[0], split='training', accession='alias', group_id='alias')
        assignments = {'alias': dict(split='training', group_id='alias', development_exposed=False)}
        with self.assertRaisesRegex(ValueError, 'Protected test'):
            reject_protected_sources(self.root, [candidate], assignments)
        with self.assertRaisesRegex(ValueError, 'Protected test'):
            prepare_review(self.root, [candidate], self.root / 'ordinary-review', assignments=assignments)
        candidate['source_sha256'] = 'changed'
        candidate['source_path'] = 'elsewhere.txt'
        with self.assertRaisesRegex(ValueError, 'Protected test'):
            reject_protected_sources(self.root, [candidate], assignments)

    def test_missing_ledger_fails_closed_for_custom_review_directory(self):
        self.prepare()
        (self.root / 'data/test-source-ledger.json').unlink()
        with self.assertRaisesRegex(ValueError, 'ledger is missing'):
            load_ledger(self.root)

    def test_cross_storage_duplicate_is_refused(self):
        source = self.root / 'ordinary.txt'
        source.write_bytes(b'Previously retained ordinary source')
        options = dict(sec_url='https://www.sec.gov/source.txt', accession='0000000999-22-000001',
                       friendly_filename='ordinary.txt', complete=True)
        import_source(source, self.root, **options)
        with self.assertRaisesRegex(ValueError, 'duplicates an existing non-test source'):
            import_source(source, self.root, storage='test', **options)

    def test_manifest_cannot_claim_another_accepted_policy(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        manifest_path = self.output / 'dataset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        (self.root / 'docs/other-contract.md').write_text('Different contract')
        manifest['label_contract'] = dict(path='docs/other-contract.md', sha256=sha(b'Different contract'))
        atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'policy differs'):
            read_test_release(self.output, project_root=self.root)

    def context(self, year=2024):
        selection = next(s for s in self.selections if s['packet_id'] == str(year))
        raw = b'<p>Omitted cover</p><tr><td colspan="2" rowspan="3">Fund &amp; Series</td></tr><p>Omitted end</p>'
        original = self.root / 'cover.htm'
        original.write_bytes(raw)
        url = selection['source_url'].replace('source.htm', 'cover.htm')
        entry = import_source(original, self.root, storage='test', sec_url=url,
                              accession=selection['accession'], friendly_filename=f'{year}-cover.htm',
                              complete=True, filing_date=selection['filing_date'], cik=selection['cik'], form='N-PX')
        start, end = raw.index(b'<tr>'), raw.index(b'</tr>') + len(b'</tr>')
        context = dict(accession=selection['accession'], source_path=entry['path'], source_sha256=entry['sha256'],
                       source_url=url, cik=selection['cik'], encoding='utf-8', spans=[[start, end, 'row']],
                       association_review='Matching synthetic series identifier')
        selection['context_sources'] = [context]
        return context

    def test_context_display_and_original_hash_are_preserved(self):
        context = self.context()
        packet = prepare_historical(self.root, self.selections[1])
        cell = packet['manifest']['blocks'][0]['cells'][0]
        self.assertEqual((cell['text'], cell['colspan'], cell['rowspan']), ('Fund & Series', '2', '3'))
        self.assertIn('colspan="2" rowspan="3"', packet['source_view'])
        self.assertIn('[OMITTED D1 SOURCE BYTES', packet['model_input'])
        self.assertNotIn('Omitted end', packet['model_input'])
        self.assertTrue(packet['manifest']['blocks'][-1]['block_id'].startswith('D2:'))
        (self.root / context['source_path']).write_bytes(b'Changed cover')
        with self.assertRaisesRegex(ValueError, 'Context source hash changed'):
            prepare_historical(self.root, self.selections[1])

    def test_context_must_match_the_admitted_filing(self):
        context = self.context()
        context['source_url'] = context['source_url'].replace('cover.htm', 'unadmitted.htm')
        with self.assertRaisesRegex(ValueError, 'Test context differs'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)

    def test_context_cannot_bypass_training_source_protection(self):
        context = self.context()
        context['split'] = 'test'
        self.prepare()
        candidate = dict(self.selections[0], split='training', accession='other', group_id='other',
                         source_path='ordinary.txt', source_sha256='ordinary', cik='999',
                         source_url='https://www.sec.gov/Archives/edgar/data/999/ordinary.txt',
                         context_sources=[context])
        assignments = {'other': dict(split='training', group_id='other', development_exposed=False)}
        with self.assertRaisesRegex(ValueError, 'Protected test'):
            reject_protected_sources(self.root, [candidate], assignments)

    def test_every_admitted_attachment_is_reserved(self):
        context = self.context()
        self.selections[1].pop('context_sources')
        self.prepare()
        source = next(s for s in load_ledger(self.root)['sources'] if s['accession'] == context['accession'])
        self.assertIn(context['source_sha256'], source['sha256'])

    def test_reader_rejects_changed_retained_acceptance(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        manifest_path = self.output / 'dataset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['acceptance']['approval']['content'] = '{}'
        atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'hash'):
            read_test_release(self.output, project_root=self.root)

    def test_rehashed_acceptance_still_needs_the_exact_review(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        manifest_path = self.output / 'dataset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['acceptance']['review']['content'] += ' '
        manifest['acceptance']['review']['sha256'] = sha(manifest['acceptance']['review']['content'].encode())
        atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'Explicit test acceptance'):
            read_test_release(self.output, project_root=self.root)

    def test_embedded_preparation_keeps_source_protection_checks(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        ledger_path = self.root / 'data/test-source-ledger.json'
        ledger = json.loads(ledger_path.read_text())
        ledger['sources'] = []
        atomic_json(ledger_path, ledger)
        with self.assertRaisesRegex(ValueError, 'reservations changed'):
            read_test_release(self.output, project_root=self.root)

    def test_final_release_needs_no_preparation_or_training_dataset(self):
        self.families()
        self.context()
        path, receipt = self.receipt()
        manifest = accept_test_review(self.root, self.review, path, receipt, self.output)
        self.assertEqual(set(manifest['files']), {'test'})
        self.assertEqual({p.name for p in self.output.iterdir()}, {'test-examples.jsonl', 'dataset-manifest.json'})
        self.assertEqual(manifest['acceptance']['review']['content'].encode(), path.read_bytes())
        self.assertEqual(manifest['acceptance']['approval']['content'].encode(), receipt.read_bytes())
        self.assertEqual(manifest['acceptance']['preparation']['content'].encode(),
                         (self.review / 'preparation.json').read_bytes())
        # Retain only policies, canonical test originals, the ledger, and the
        # final two-file dataset. Even prior raw evidence is no longer needed.
        for item in self.root.iterdir():
            if item.name not in {'configs', 'docs', 'data', 'protocol.json', self.output.name}:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        for item in (self.root / 'data').iterdir():
            if item.name not in {'raw', 'test-source-ledger.json'}:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        rows, manifest = read_test_release(self.output, project_root=self.root)
        self.assertEqual(len(rows['test']), 2)
        self.assertEqual([m['role'] for m in rows['test'][0]['messages']], ['system','user','assistant'])
        with self.assertRaisesRegex(ValueError, 'unsupported dataset'):
            read_release(self.output, project_root=self.root)
        model = self.root / 'synthetic-model.gguf'
        model.write_bytes(b'CPU input checks only')
        inputs = prepare_inputs(self.output, model, {}, project_root=self.root)
        self.assertEqual(inputs['dataset_split'], 'test')
        self.assertEqual(inputs['identity']['split'], 'test')
        self.assertEqual(inputs['identity']['scorer_version'], 'source-cells-v1')
        self.assertTrue(all([m['role'] for m in case['messages']] == ['system', 'user'] for case in inputs['cases']))

    def test_missing_retained_metadata_cannot_fall_back_to_preparation_files(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        manifest_path = self.output / 'dataset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        preparation = json.loads(manifest['acceptance']['preparation']['content'])
        del manifest['evidence']['documents'][preparation['packets']['path']]
        atomic_json(manifest_path, manifest)
        self.assertTrue((self.review / 'packet-set.json').exists())
        with self.assertRaisesRegex(ValueError, 'metadata is missing'):
            read_test_release(self.output, project_root=self.root)

    def test_retained_metadata_cannot_replace_live_raw_sources(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        manifest_path = self.output / 'dataset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        source_path = self.selections[0]['source_path']
        manifest['evidence']['documents'][source_path] = dict(content='fake raw', sha256=sha(b'fake raw'))
        atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'cannot replace raw sources'):
            read_test_release(self.output, project_root=self.root)

    def test_retained_audits_keep_their_accepted_hashes(self):
        self.families()
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        manifest_path = self.output / 'dataset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        frozen = manifest['evidence']['documents']['family-audit.json']
        frozen['content'] += ' '
        frozen['sha256'] = sha(frozen['content'].encode())
        atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'metadata is missing or has a hash mismatch'):
            read_test_release(self.output, project_root=self.root)

    def test_changed_live_test_source_still_fails_after_acceptance(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        (self.root / self.selections[0]['source_path']).write_bytes(b'Changed selected source')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            read_test_release(self.output, project_root=self.root)

    def test_acceptance_still_refuses_a_changed_live_training_baseline(self):
        path, receipt = self.receipt()
        atomic_json(self.root / 'data/training-dataset/dataset-manifest.json', dict(assignments={}))
        with self.assertRaisesRegex(ValueError, 'Prior dataset changed'):
            accept_test_review(self.root, self.review, path, receipt, self.output)

    def test_frozen_scorer_change_prevents_test_preparation_and_reporting(self):
        path, receipt = self.receipt()
        accept_test_review(self.root, self.review, path, receipt, self.output)
        model = self.root / 'model.gguf'
        model.write_bytes(b'Synthetic model')
        inputs = prepare_inputs(self.output, model, {}, project_root=self.root)
        with patch('proxybench.evaluation.workflow.SCORER_VERSION', 'changed-scoring-rules'):
            with self.assertRaisesRegex(ValueError, 'Frozen test scorer'):
                prepare_inputs(self.output, model, {}, project_root=self.root)
            with self.assertRaisesRegex(ValueError, 'Frozen test scorer'):
                check_test_scorer(inputs)
            with self.assertRaisesRegex(ValueError, 'Frozen test scorer'):
                report(None, inputs, {})

    def test_source_comparison_hash_and_target_binding_are_checked(self):
        comparison_path = self.root / 'source-comparisons.json'
        comparisons = dict(schema='test-source-comparison-v1', baseline_inventory=self.audit['prior_inventory'],
                           prior_dataset=self.audit['prior_dataset'], records={s['packet_id']:dict(
                               accession=s['accession'], source_path=s['source_path'], source_sha256=s['source_sha256'],
                               target_bytes=s['target']) for s in self.selections})
        atomic_json(comparison_path, comparisons)
        self.audit['source_comparisons'] = file_binding(comparison_path, self.root)
        validate_selections(self.root, self.selections, self.protocol, self.audit)
        comparisons['records']['2024']['target_bytes'] = [0, 1]
        atomic_json(comparison_path, comparisons)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)
        self.audit['source_comparisons'] = file_binding(comparison_path, self.root)
        with self.assertRaisesRegex(ValueError, 'another selected source or target'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)

    def families(self):
        legacy = self.root / 'old-provider.txt'
        legacy.write_bytes(b'Prior exposed provider')
        evidence = dict(source=file_binding(legacy, self.root), bytes=[0, 5], slice_sha256=sha(b'Prior'))
        audit = dict(schema='test-family-audit-v1', prior_inventory=self.audit['prior_inventory'],
                     prior_dataset=self.audit['prior_dataset'], baseline_sources=[],
                     baseline_assignments={'0000000999-21-000001': dict(family_ids=['family-old'], evidence=[evidence])},
                     candidates={})
        for selection in self.selections:
            family = f'family-new-{selection["cik"]}'
            selection['family_id'] = family
            path = self.root / selection['source_path']
            audit['candidates'][selection['accession']] = dict(family_id=family, cik=selection['cik'],
                source_sha256=[selection['source_sha256']], evidence=[dict(source=file_binding(path, self.root),
                bytes=[0, 3], slice_sha256=sha(path.read_bytes()[:3]))])
        self.family_path = self.root / 'family-audit.json'
        self.save_families(audit)
        return audit

    def save_families(self, audit):
        atomic_json(self.family_path, audit)
        self.audit['family_audit'] = file_binding(self.family_path, self.root)
        atomic_json(self.root / 'audit.json', self.audit)

    def test_provider_audit_is_required_by_the_real_protocol_gate(self):
        self.protocol['family_audit_required'] = True
        with self.assertRaisesRegex(ValueError, 'requires a reviewed provider-family audit'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)
        self.families()
        validate_selections(self.root, self.selections, self.protocol, self.audit)

    def test_provider_family_overlap_and_noncanonical_names_are_rejected(self):
        audit = self.families()
        first, second = self.selections
        for family, error in [('family-old','overlaps prior'), (second['family_id'],'another test'), ('Family-New-100','reviewed provider')]:
            first['family_id'] = family
            audit['candidates'][first['accession']]['family_id'] = family
            self.save_families(audit)
            with self.assertRaisesRegex(ValueError, error):
                validate_selections(self.root, self.selections, self.protocol, self.audit)

    def test_provider_evidence_must_belong_to_the_selected_filing(self):
        audit = self.families()
        first, second = self.selections
        audit['candidates'][first['accession']]['evidence'] = audit['candidates'][second['accession']]['evidence']
        self.save_families(audit)
        with self.assertRaisesRegex(ValueError, 'belongs to another filing'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)

    def test_provider_evidence_keeps_original_bytes_and_ranges(self):
        audit = self.families()
        audit['candidates'][self.selections[0]['accession']]['evidence'][0]['bytes'] = [0, 4]
        self.save_families(audit)
        with self.assertRaisesRegex(ValueError, 'original source range'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)
        (self.root / 'old-provider.txt').write_bytes(b'Changed prior provider')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)

    def test_provider_audit_must_cover_unselected_prior_files_and_assignments(self):
        audit = self.families()
        audit['baseline_assignments'] = {}
        self.save_families(audit)
        with self.assertRaisesRegex(ValueError, 'all prior source assignments'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)
        audit['baseline_assignments'] = {'0000000999-21-000001':dict(family_ids=['family-old'],
            evidence=[dict(source=file_binding(self.root / 'old-provider.txt', self.root), bytes=[0,5],slice_sha256=sha(b'Prior'))])}
        audit['baseline_sources'] = [dict(path='invented.txt',sha256='invented',family_ids=['family-old'])]
        self.save_families(audit)
        with self.assertRaisesRegex(ValueError, 'all prior source files'):
            validate_selections(self.root, self.selections, self.protocol, self.audit)

    def test_protected_provider_family_blocks_another_registrant(self):
        self.families()
        self.prepare()
        candidate = dict(self.selections[0], split='training',accession='other',group_id='other',
                         source_sha256='other',source_path='other.txt',cik='999',
                         source_url='https://www.sec.gov/Archives/edgar/data/999/other.txt')
        assignments = {'other':dict(split='training',group_id='other',development_exposed=False)}
        with self.assertRaisesRegex(ValueError, 'Protected test'):
            reject_protected_sources(self.root,[candidate],assignments)
        candidate.pop('family_id')
        candidate['group_id'] = self.selections[0]['family_id']
        assignments['other']['group_id'] = candidate['group_id']
        with self.assertRaisesRegex(ValueError, 'Protected test'):
            reject_protected_sources(self.root,[candidate],assignments)

    def test_source_comparisons_cannot_reuse_a_stale_provider_audit(self):
        self.families()
        comparisons = dict(schema='test-source-comparison-v1', baseline_inventory=self.audit['prior_inventory'],
                           prior_dataset=self.audit['prior_dataset'], records={s['packet_id']:dict(
                               accession=s['accession'], source_path=s['source_path'], source_sha256=s['source_sha256'],
                               target_bytes=s['target']) for s in self.selections})
        path = self.root / 'comparisons.json'
        atomic_json(path, comparisons)
        self.audit['source_comparisons'] = file_binding(path, self.root)
        with self.assertRaisesRegex(ValueError, 'Source comparisons differ'):
            validate_selections(self.root,self.selections,self.protocol,self.audit)
        comparisons['family_audit'] = self.audit['family_audit']
        atomic_json(path, comparisons)
        self.audit['source_comparisons'] = file_binding(path, self.root)
        validate_selections(self.root,self.selections,self.protocol,self.audit)


if __name__ == '__main__':
    unittest.main()
