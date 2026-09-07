"""Synthetic source boundaries, date policies, capture, and subprocess failures."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from proxybench.annotation.bindings import sha256
from proxybench.evaluation.projection import primary_paths
from proxybench.evaluation.references import synthetic_reference
from proxybench.evaluation.scoring import score_fragment
from proxybench.execution.runner import (Association, RecordedCapture, RecordedPredictionAdapter,
    ScheduledInput, SubprocessAdapter, run_development)
from proxybench.extraction.html_table import extract
from proxybench.normalization.values import ACTIVE_GUIDE, LEGACY_GUIDE, calendar_date
from proxybench.schemas.records import ValidationError, normalize_record


def bundle(*, issuer='Different Issuer', fund='Fund: Cedar Income', vote='For',
           date='6/10/2014', date_heading='Meeting Date', before='', after='', number='001',
           description='Elect the collective slate', reverse=False):
    columns = [('Proposal Number', number), ('Description', description), ('Vote Cast', vote), ('Mgmt Rec', 'For')]
    if reverse:
        columns.reverse()
    heading = '<tr>' + ''.join('<th>' + a + '</th>' for a, _ in columns) + '</tr>'
    target = '<tr>' + ''.join('<td>' + b + '</td>' for _, b in columns) + '</tr>'
    source = (f'<p>{fund}</p>{before}<table><tr><td>Issuer:</td><td>{issuer}</td></tr>'
              f'<tr><td>{date_heading}</td><td>{date}</td></tr>' + heading + target + after + '</table>')
    raw = source.encode()
    start = raw.index(target.encode())
    span = {'document_id': 'synthetic', 'source_sha256': sha256(raw), 'start_byte': 0, 'end_byte': len(raw)}
    return {'input_version': 'fragment-input-v1', 'input_id': 'example', 'track': 'fragment', 'encoding': 'utf-8',
            'view_id': 'view', 'document_id': 'synthetic', 'source_sha256': sha256(raw),
            'target': {'start_byte': start, 'end_byte': start + len(target.encode()), 'sha256': sha256(target.encode())},
            'blocks': [{'block_id': 'B1', 'span': span, 'original_text': source, 'prepared_text': 'unused'}]}


def replace_source(b, source):
    """Rebuild a synthetic source mapping after a deliberate layout mutation."""
    target_start = source.rfind('<tr>')
    target_end = source.find('</tr>', target_start) + len('</tr>')
    raw = source.encode()
    target = source[target_start:target_end].encode()
    b['source_sha256'] = sha256(raw)
    b['blocks'][0].update(original_text=source, span={**b['blocks'][0]['span'],
        'source_sha256': sha256(raw), 'end_byte': len(raw)})
    b['target'] = {'start_byte': len(source[:target_start].encode()),
                   'end_byte': len(source[:target_end].encode()), 'sha256': sha256(target)}
    return b


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode()


def fixture(root, source=None):
    b = source or bundle()
    raw = encoded(b)
    def save(name, content):
        data = content if isinstance(content, bytes) else encoded(content)
        (root / name).write_bytes(data)
        return sha256(data)
    save('source.html', b['blocks'][0]['original_text'].encode())
    input_hash = save('input.json', raw)
    manifest = {'source_path': 'source.html', 'source_sha256': b['source_sha256'], 'target': b['target'],
                'blocks': [{'block_id': 'B1', 'start_byte': 0, 'end_byte': len((root / 'source.html').read_bytes())}]}
    mh = save('manifest.json', manifest)
    vh = save('view.html', b'Preserved synthetic review view')
    lh = save('original.json', b'Preserved synthetic accepted label')
    binding = {'binding_version': 'input-binding-v1', 'input_id': b['input_id'], 'review_tool_revision': 'fixture',
               'representation_decision': 'SAME_EVIDENCE', 'reviewer': 'synthetic reviewer', 'reviewed_at': '2026-09-07',
               'reason': 'Synthetic behavior fixture', 'source_manifest_path': 'manifest.json', 'source_manifest_sha256': mh,
               'review_view_path': 'view.html', 'review_view_sha256': vh, 'model_input_path': 'input.json', 'model_input_sha256': input_hash}
    bh = save('binding.json', binding)
    r = extract(raw)['records'][0]
    reference = {'reference_id': 'synthetic-reference', 'input_id': b['input_id'], 'label_version': 'fixture',
                 'label_status': 'HUMAN_ACCEPTED', 'reviewer': 'synthetic reviewer', 'reviewed_at': '2026-09-07',
                 'guide_version': ACTIVE_GUIDE, 'original_label_path': 'original.json', 'original_label_sha256': lh,
                 'input_binding_path': 'binding.json', 'input_binding_sha256': bh,
                 'record': r, 'primary_paths': primary_paths(normalize_record(r, guide_version=ACTIVE_GUIDE)), 'conversion_log': []}
    rh = save('reference.json', reference)
    return ScheduledInput(b['input_id'], 'input.json', input_hash, 'reference.json', rh, bh), encoded(extract(raw))


class DateTests(unittest.TestCase):
    def test_active_and_original_policy(self):
        for value, expected in [('09/01/2006', '2006-09-01'), ('6/10/2014', '2014-06-10'),
                                ('12/3/2014', '2014-12-03'), ('2/29/2024', '2024-02-29'),
                                (' APR 30, 2021 ', '2021-04-30'), ('29-Feb-2024', '2024-02-29')]:
            self.assertEqual(calendar_date(value, guide_version=ACTIVE_GUIDE), expected)
        for value in ('15/05/2007', '2/29/2023', '2024-2-01', '2024/01/01', 'May 1 2007', '1/1/07'):
            with self.assertRaises(ValueError):
                calendar_date(value, guide_version=ACTIVE_GUIDE)
        self.assertEqual(calendar_date('15/05/2007', guide_version=LEGACY_GUIDE), '2007-05-15')
        with self.assertRaises(ValueError):
            calendar_date('6/10/2014', guide_version=LEGACY_GUIDE)
        with self.assertRaises(ValueError):
            calendar_date('2024-01-01', guide_version='future')

    def test_scorer_uses_reference_policy(self):
        response = extract(encoded(bundle()))
        reference = synthetic_reference(response['records'][0], input_id='example', guide_version=ACTIVE_GUIDE)
        response['records'][0]['fields']['meeting_date']['value'] = '6/10/2014'
        self.assertTrue(score_fragment(encoded(response), reference)['passed'])
        old = synthetic_reference(reference.record, input_id='example', guide_version=LEGACY_GUIDE)
        self.assertFalse(score_fragment(encoded(response), old)['passed'])
        response['records'][0]['fields']['meeting_date']['value'] = '15/05/2007'
        self.assertIn('MALFORMED_RESPONSE', score_fragment(encoded(response), reference)['reasons'])


class ParserTests(unittest.TestCase):
    def fields(self, **kwargs):
        result = extract(encoded(bundle(**kwargs)))
        self.assertEqual(result['status'], 'COMPLETE', result)
        normalize_record(result['records'][0], guide_version=ACTIVE_GUIDE)
        return result['records'][0]['fields']

    def test_names_reordered_columns_collective_and_strings(self):
        fields = self.fields(issuer='Élan &amp; Sons', fund='Fund Group: Birch and Cedar', reverse=True,
                             after='<tr><td>002</td><td>Other subject</td><td>Against</td><td>Against</td></tr>')
        self.assertEqual(fields['issuer_name']['value'], 'Élan & Sons')
        self.assertEqual(fields['reporting_scope']['value']['name']['value'], 'Birch and Cedar')
        self.assertEqual(fields['reporting_scope']['value']['scope_type']['value'], 'FUND_GROUP')
        self.assertEqual(fields['raw_description']['value'], 'Elect the collective slate')
        self.assertEqual(fields['proposal_number']['value'], '001')
        self.assertIsNone(fields['separate_subject']['value'])
        # Reorder neighboring physical rows while keeping the same marked subject.
        for neighbor_first in (True, False):
            b = bundle()
            source = b['blocks'][0]['original_text']
            start, end = b['target']['start_byte'], b['target']['end_byte']
            target_text = source.encode()[start:end].decode()
            neighbor = target_text.replace('collective slate', 'other slate').replace('001', '009')
            source = source.replace(target_text, neighbor + target_text if neighbor_first else target_text + neighbor)
            raw = source.encode()
            offset = raw.index(target_text.encode())
            b['source_sha256'] = sha256(raw)
            b['blocks'][0].update(original_text=source, span={**b['blocks'][0]['span'],
                'source_sha256': sha256(raw), 'end_byte': len(raw)})
            b['target'].update(start_byte=offset, end_byte=offset + len(target_text.encode()))
            extracted = extract(encoded(b))['records'][0]['fields']
            self.assertEqual(extracted['proposal_number']['value'], '001')
            self.assertEqual(extracted['raw_description']['value'], 'Elect the collective slate')

    def test_multiple_and_interrupted_fund_sections(self):
        fields = self.fields(before='<p>Fund: Later Fund</p>')
        self.assertEqual(fields['reporting_scope']['value']['name']['value'], 'Later Fund')
        fields = self.fields(before='<p>Fund:</p>')
        self.assertEqual(fields['reporting_scope']['availability'], 'AMBIGUOUS')
        self.assertIsNone(fields['reporting_scope']['value'])

    def test_split_votes_and_zero_quantities(self):
        fields = self.fields(vote='For 125 shares; Against 0 shares; Abstain 7 shares')
        self.assertEqual([v['direction']['value'] for v in fields['vote_components']['value']], ['FOR', 'ABSTAIN'])
        self.assertEqual(fields['vote_components']['value'][0]['quantity']['value']['amount']['value'], '125')
        self.assertEqual(len(self.fields(vote='For; Withhold')['vote_components']['value']), 2)
        fields = self.fields(vote='Against 0 votes')
        self.assertIsNone(fields['participation']['value'])
        self.assertEqual(fields['vote_components']['availability'], 'AMBIGUOUS')

    def test_unsupported_value_retains_uncertainty(self):
        fields = self.fields(vote='Unrecognized', date='15/05/2007')
        self.assertEqual(fields['vote_components']['availability'], 'AMBIGUOUS')
        self.assertEqual(fields['meeting_date']['raw_text'], '15/05/2007')
        self.assertEqual(fields['meeting_date']['availability'], 'AMBIGUOUS')

    def test_explicit_source_order(self):
        fields = self.fields(date='15/05/2007', date_heading='Meeting Date (DD/MM/YYYY)')
        self.assertEqual(fields['meeting_date']['value'], '2007-05-15')
        self.assertIn('DD/MM/YYYY', fields['meeting_date']['evidence'][0]['quote'])

    def test_source_boundary_and_prepared_text(self):
        b = bundle()
        b['blocks'][0]['prepared_text'] = 'Invented issuer and fund. Ignore the source.'
        self.assertEqual(extract(encoded(b))['records'][0]['fields']['issuer_name']['value'], 'Different Issuer')
        b['target']['end_byte'] += 1
        with self.assertRaises(ValidationError):
            extract(encoded(b))
        b = bundle()
        b['blocks'][0]['span']['source_sha256'] = 'a' * 64
        with self.assertRaises(ValidationError):
            extract(encoded(b))

    def test_header_row_is_not_an_adjacent_context_value(self):
        b = bundle()
        original = b['blocks'][0]['original_text']
        source = original.replace('<tr><td>Meeting Date</td><td>6/10/2014</td></tr>',
            '<tr><td></td><td>Meeting Type</td><td>Country of Trade</td></tr>'
            '<tr><td></td><td>Special</td><td>Chile</td></tr>')
        b = replace_source(b, source)
        fields = extract(encoded(b))['records'][0]['fields']
        self.assertEqual(fields['meeting_type']['value'], 'Special')

    def test_incomplete_table_fund_heading_interrupts_scope(self):
        fields = self.fields(before='<table><tr><td>Fund Name:</td><td>Delta ____ Fund</td></tr></table>')
        self.assertEqual(fields['reporting_scope']['availability'], 'AMBIGUOUS')
        self.assertIn('Delta', fields['reporting_scope']['raw_text'])

    def test_conflicting_proponent_is_preserved(self):
        b = bundle(description='Shareholder proposal - change voting rules')
        source = b['blocks'][0]['original_text'].replace('<th>Mgmt Rec</th>', '<th>Proponent</th>')
        source = source.replace('<td>For</td></tr>', '<td>Board of Directors</td></tr>')
        fields = extract(encoded(replace_source(b, source)))['records'][0]['fields']
        self.assertEqual(fields['proposal_source']['availability'], 'CONFLICTING')
        self.assertGreaterEqual(len(fields['proposal_source']['evidence']), 2)

    def test_parser_does_not_open_files(self):
        raw = encoded(bundle())
        with patch('builtins.open', side_effect=AssertionError('No file access')):
            self.assertEqual(extract(raw)['status'], 'COMPLETE')

    def test_unsupported_table_structure_and_extra_input_data(self):
        for replacement in ('<td rowspan="2">001</td>', '<td><table><tr><td>001</td></tr></table></td>'):
            b = bundle()
            b = replace_source(b, b['blocks'][0]['original_text'].replace('<td>001</td>', replacement))
            self.assertEqual(extract(encoded(b))['status'], 'ABSTAINED')
        b = bundle()
        b['reference_answer'] = 'Outside source-only contract'
        with self.assertRaises(ValidationError):
            extract(encoded(b))

    def test_split_alignment_is_not_invented_per_component(self):
        b = bundle(vote='For; Against')
        source = b['blocks'][0]['original_text'].replace('<th>Mgmt Rec</th>', '<th>For/Agnst Mgmt</th>')
        fields = extract(encoded(replace_source(b, source)))['records'][0]['fields']
        self.assertTrue(all(c['disclosed_management_alignment']['availability'] == 'AMBIGUOUS'
                            for c in fields['vote_components']['value']))

    def test_non_row_target_abstains(self):
        b = bundle()
        original = b['blocks'][0]['original_text'].encode()
        b['target']['start_byte'] += 4
        b['target']['sha256'] = sha256(original[b['target']['start_byte']:b['target']['end_byte']])
        self.assertEqual(extract(encoded(b))['status'], 'ABSTAINED')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.item, self.answer = fixture(self.root)
        self.association = Association('run', 'system', 'example', self.item.input_sha256)

    def run_adapter(self, adapter, **kwargs):
        return run_development(self.root, self.root / 'out', [self.item], adapter, run_id='run', system_id='system', **kwargs)

    def test_replay_round_trip_and_no_overwrite(self):
        adapter = RecordedPredictionAdapter([RecordedCapture(self.association, self.answer)])
        report = self.run_adapter(adapter)
        self.assertTrue(report['results'][0]['primary_success'])
        self.assertEqual((self.root / 'out/0000/raw.bin').read_bytes(), self.answer)
        with self.assertRaises(ValidationError):
            self.run_adapter(adapter)
        json.dumps(report, allow_nan=False)

    def test_swapped_capture_and_duplicate_schedule_refused(self):
        wrong = Association('run', 'system', 'different', self.item.input_sha256)
        with self.assertRaises(ValidationError):
            self.run_adapter(RecordedPredictionAdapter([RecordedCapture(wrong, self.answer)]))
        with self.assertRaises(ValidationError):
            run_development(self.root, self.root / 'out', [self.item, self.item], RecordedPredictionAdapter([]), run_id='run', system_id='system')
        self.assertFalse((self.root / 'out').exists())

    def test_test_split_is_refused(self):
        self.item = replace(self.item, split='test')
        with self.assertRaises(ValidationError):
            self.run_adapter(RecordedPredictionAdapter([]))

    def test_changed_original_source_prevents_launch(self):
        with (self.root / 'source.html').open('ab') as stream:
            stream.write(b'changed source')
        with self.assertRaises(ValidationError):
            self.run_adapter(RecordedPredictionAdapter([]))

    def test_recorded_limit_keeps_bytes_and_fails_eligibility(self):
        result = self.run_adapter(RecordedPredictionAdapter([RecordedCapture(self.association, self.answer)]), output_bytes=16)['results'][0]
        self.assertFalse(result['execution_eligible'])
        self.assertEqual((self.root / 'out/0000/raw.bin').read_bytes(), self.answer)

    def test_every_scheduled_input_survives_capture_failure(self):
        second_bundle = bundle(issuer='Second Issuer')
        second_bundle['input_id'] = 'second'
        second_dir = self.root / 'second'
        second_dir.mkdir()
        second, answer = fixture(second_dir, second_bundle)
        binding = json.loads((second_dir / 'binding.json').read_text())
        for name in ('source_manifest', 'review_view', 'model_input'):
            binding[name + '_path'] = 'second/' + binding[name + '_path']
        manifest = json.loads((second_dir / 'manifest.json').read_text())
        manifest['source_path'] = 'second/source.html'
        (second_dir / 'manifest.json').write_bytes(encoded(manifest))
        binding['source_manifest_sha256'] = sha256(encoded(manifest))
        (second_dir / 'binding.json').write_bytes(encoded(binding))
        ref = json.loads((second_dir / 'reference.json').read_text())
        ref.update(original_label_path='second/original.json', input_binding_path='second/binding.json',
                   input_binding_sha256=sha256(encoded(binding)))
        (second_dir / 'reference.json').write_bytes(encoded(ref))
        second = replace(second, input_path='second/input.json', reference_path='second/reference.json',
                         reference_sha256=sha256(encoded(ref)), binding_sha256=sha256(encoded(binding)))
        association = Association('run', 'system', 'second', second.input_sha256)
        report = run_development(self.root, self.root / 'out', [self.item, second],
            RecordedPredictionAdapter([RecordedCapture(association, answer)]), run_id='run', system_id='system')
        self.assertEqual(report['scheduled'], 2)
        self.assertEqual(report['terminal_results'], 2)
        self.assertEqual(report['results'][0]['observation']['outcome'], 'MISSING')
        self.assertTrue(report['results'][1]['primary_success'])

    def test_changed_binding_prevents_launch(self):
        (self.root / 'binding.json').write_text('{}')
        adapter = SubprocessAdapter([sys.executable, '-c', 'raise Exception("must not run")'])
        with patch.object(adapter, 'capture') as capture, self.assertRaises(ValidationError):
            self.run_adapter(adapter)
        capture.assert_not_called()

    def test_timeout_complete_content_is_ineligible(self):
        capture = RecordedCapture(self.association, self.answer, outcome='TIMEOUT')
        result = self.run_adapter(RecordedPredictionAdapter([capture]))['results'][0]
        self.assertTrue(result['content_passed'])
        self.assertFalse(result['primary_success'])
        self.assertFalse(result['execution_success'])
        self.assertEqual(result['model_status'], 'COMPLETE')

    def test_missing_output_has_terminal_result(self):
        report = self.run_adapter(RecordedPredictionAdapter([]))
        self.assertEqual(report['scheduled'], report['terminal_results'])
        self.assertEqual(report['results'][0]['observation']['outcome'], 'MISSING')
        self.assertFalse((self.root / 'out/0000/raw.bin').exists())

    def test_malformed_non_utf8_bytes_preserved(self):
        raw = b'\xff{"status":'
        report = self.run_adapter(RecordedPredictionAdapter([RecordedCapture(self.association, raw)]))
        self.assertFalse(report['results'][0]['primary_success'])
        self.assertEqual((self.root / 'out/0000/raw.bin').read_bytes(), raw)

    def test_partial_capture_cannot_pass(self):
        result = self.run_adapter(RecordedPredictionAdapter([
            RecordedCapture(self.association, self.answer, capture_complete=False)]))['results'][0]
        self.assertTrue(result['content_passed'])
        self.assertFalse(result['primary_success'])

    def test_wrong_echo_is_only_metadata(self):
        value = json.loads(self.answer)
        value['input_id'] = 'wrong-echo'
        result = self.run_adapter(RecordedPredictionAdapter([RecordedCapture(self.association, encoded(value))]))['results'][0]
        self.assertTrue(result['primary_success'])
        score = json.loads((self.root / 'out/0000/score.json').read_text())
        self.assertEqual(score['metadata_errors'], ['INPUT_ID_MISMATCH'])

    def test_real_timeout_and_no_second_invocation(self):
        marker = self.root / 'calls'
        code = 'from pathlib import Path; import sys,time; p=Path(sys.argv[1]); p.open("a").write("call\\n"); sys.stdout.buffer.write(Path(sys.argv[2]).read_bytes()); sys.stdout.flush(); time.sleep(2)'
        (self.root / 'answer').write_bytes(self.answer)
        result = self.run_adapter(SubprocessAdapter([sys.executable, '-c', code, str(marker), str(self.root / 'answer')]), timeout_seconds=0.2)['results'][0]
        self.assertEqual(marker.read_text(), 'call\n')
        self.assertEqual(result['observation']['outcome'], 'TIMEOUT')
        self.assertEqual(result['model_status'], 'COMPLETE')
        self.assertFalse(result['primary_success'])

    def test_real_output_limit(self):
        result = self.run_adapter(SubprocessAdapter([sys.executable, '-c', 'import sys; sys.stdout.buffer.write(b"x"*10000)']), output_bytes=64)['results'][0]
        self.assertEqual(result['observation']['outcome'], 'TRUNCATED')
        self.assertEqual(len((self.root / 'out/0000/raw.bin').read_bytes()), 64)

    def test_real_crash_and_numeric_malformed_json_serialization(self):
        result = self.run_adapter(SubprocessAdapter([sys.executable, '-c', 'print("{\\"number\\":1.5}"); raise SystemExit(7)']))['results'][0]
        self.assertEqual(result['observation']['outcome'], 'CRASHED')
        parsed = json.loads((self.root / 'out/0000/parsed.json').read_text())
        self.assertEqual(parsed['value']['number']['json_type'], 'number')

    def test_partial_write_exception_has_terminal_failure(self):
        class BrokenAdapter(RecordedPredictionAdapter):
            def capture(self, association, input_path, directory, limits):
                (directory / 'raw.bin').write_bytes(b'{"status":')
                raise OSError('Synthetic disk failure')
        result = self.run_adapter(BrokenAdapter([]))['results'][0]
        self.assertEqual(result['observation']['outcome'], 'CAPTURE_FAILED')
        self.assertFalse(result['primary_success'])
        self.assertEqual((self.root / 'out/0000/raw.bin').read_bytes(), b'{"status":')


if __name__ == '__main__':
    unittest.main()
