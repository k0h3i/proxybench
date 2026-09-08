"""Synthetic source assertions, independent of mapped values and rendered targets."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import sys

from proxybench.sources.npx_xml import FORM_NS, VOTE_NS, XMLRejected, parse_xml, discover_attachments
from proxybench.normalization.npx import map_filing
from proxybench.training.rendering import render_record
from proxybench.evaluation.evidence import SourceContext
from proxybench.training.sequences import sequence, pad_sequence
from proxybench.training.preparation import prepare, record_review
from proxybench.execution.resources import limit_reason, supervise
from proxybench.execution.probe_report import score_probe
from proxybench.evaluation.references import synthetic_reference
from proxybench.normalization.values import ACTIVE_GUIDE


PRIMARY = f'''<edgarSubmission xmlns="{FORM_NS}"><headerData><submissionType>N-PX</submissionType>
<filerInfo><registrantType>RMIC</registrantType></filerInfo></headerData><formData>
<coverPage><reportInfo><reportType>FUND VOTING REPORT</reportType></reportInfo></coverPage>
<seriesPage><seriesDetails><seriesReports><idOfSeries>S000000001</idOfSeries>
<nameOfSeries>Fonds Élan &amp; Co</nameOfSeries></seriesReports>
<seriesReports><idOfSeries>S000000002</idOfSeries><nameOfSeries>Next Fund</nameOfSeries>
</seriesReports></seriesDetails></seriesPage></formData></edgarSubmission>'''.encode()
ROW = '''<proxyTable><issuerName>Élan &amp; Co</issuerName><cusip>001234567</cusip>
<meetingDate>06/01/2025</meetingDate><voteDescription>Approve A &amp; B &lt;C&gt;</voteDescription>
<voteCategories><voteCategory><categoryType>AUDIT-RELATED</categoryType></voteCategory></voteCategories>
<voteSource>SECURITY HOLDER</voteSource><sharesVoted>12</sharesVoted><sharesOnLoan>7</sharesOnLoan>
<vote><voteRecord><howVoted>ABSTAIN</howVoted><sharesVoted>2</sharesVoted>
<managementRecommendation>AGAINST</managementRecommendation></voteRecord>
<voteRecord><howVoted>FOR</howVoted><sharesVoted>10</sharesVoted>
<managementRecommendation>FOR</managementRecommendation></voteRecord></vote><voteSeries>S000000001</voteSeries></proxyTable>'''


def votes(row=ROW):
    return f'<proxyVoteTable xmlns="{VOTE_NS}">{row}</proxyVoteTable>'.encode()


def mapped(row=ROW, primary=PRIMARY):
    return map_filing(primary, votes(row), primary_id="p", votes_id="v")


class XMLTests(unittest.TestCase):
    def test_dtd_entities_encodings_and_limits(self):
        cases = [b'<!DOCTYPE x [<!ENTITY x "boom">]><x/>',
                 b'<!DOCTYPE x SYSTEM "https://invalid.example/a"><x/>',
                 b'<?xml version="1.0" encoding="utf-16"?><x/>', b'<x>',
                 b'<proxyVoteTable xmlns="unknown"/>']
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(XMLRejected):
                parse_xml(raw, namespace=VOTE_NS, root_name="proxyVoteTable")
        for limit in ({"max_bytes": 8}, {"max_depth": 2}, {"max_nodes": 3}, {"max_records": 1}):
            with self.subTest(limit=limit), self.assertRaises(XMLRejected):
                parse_xml(votes(ROW + ROW), namespace=VOTE_NS, root_name="proxyVoteTable", **limit)

    def test_namespace_and_byte_spans(self):
        raw = votes()
        root = parse_xml(raw, namespace=VOTE_NS, root_name="proxyVoteTable")
        node = root.children[0].one("issuerName")
        self.assertEqual(raw[node.start:node.end].decode(), '<issuerName>Élan &amp; Co</issuerName>')
        self.assertEqual(node.text, "Élan & Co")
        with self.assertRaises(XMLRejected):
            parse_xml(votes(ROW.replace('<cusip>', '<cusip xmlns="wrong">')), namespace=VOTE_NS, root_name="proxyVoteTable")

    def test_roles_not_names(self):
        base = "https://www.sec.gov/Archives/edgar/data/1/000000000125000001/index.html"
        raw = '<tr><td>2</td><td>DATA</td><td><a href="unusual.xml">unusual.xml</a></td><td>PROXY VOTING RECORD</td><td>55</td></tr>'
        self.assertEqual(discover_attachments(raw, base)[0]["name"], "unusual.xml")
        self.assertEqual(discover_attachments(raw.replace('href="unusual.xml"', 'href="https://bad.example/unusual.xml"'), base), [])

    def test_independent_values_split_and_loan(self):
        result = mapped()
        self.assertEqual((result['discovered'], result['provisional'], result['rejected'], result['admitted']), (1, 1, 0, 0))
        record = result['records'][0]
        actual = {c['key']: c['text'] for c in record['cells']}
        self.assertEqual(actual['scope'], 'Fonds Élan & Co')
        self.assertEqual(actual['cusip'], '001234567')
        self.assertEqual(actual['quantity:0'], '2')
        self.assertEqual(actual['quantity:1'], '10')
        self.assertEqual(actual['loan'], '7')
        self.assertEqual(record['components'], 2)

    def test_rejections_preserve_denominator_and_location(self):
        changes = [('<voteSeries>S000000001', '<voteSeries>S000009999'),
                   ('<sharesVoted>2</sharesVoted>', '<sharesVoted>0</sharesVoted>'),
                   ('<sharesVoted>12</sharesVoted>', '<sharesVoted>14</sharesVoted>'),
                   ('AUDIT-RELATED', 'DIRECTOR ELECTIONS'),
                   ('<cusip>001234567</cusip>', '<cusip>1</cusip><cusip>2</cusip>'),
                   ('<issuerName>', '<mystery>test</mystery><issuerName>'),
                   ('<voteSeries>', '<voteOtherInfo>Fund group</voteOtherInfo><voteSeries>')]
        for before, after in changes:
            with self.subTest(after=after):
                result = mapped(ROW + ROW.replace(before, after))
                self.assertEqual((result['selected'], result['provisional'], result['rejected']), (2, 1, 1))
                self.assertIn('start_byte', result['records'][1]['source'])
                self.assertTrue(result['records'][1]['reason'])
        with self.assertRaises(XMLRejected):
            mapped(primary=PRIMARY.replace(b'RMIC', b'IM'))
        with self.assertRaises(XMLRejected):
            mapped(primary=PRIMARY.replace(b'FUND VOTING REPORT', b'NOTICE REPORT'))

    def test_amendment_and_neighbor_scopes(self):
        result = mapped(ROW + ROW.replace('S000000001', 'S000000002'), PRIMARY.replace(b'>N-PX<', b'>N-PX/A<'))
        self.assertEqual(result['submission_type'], 'N-PX/A')
        self.assertEqual(result['records'][0]['cells'][0]['text'], 'Fonds Élan & Co')
        self.assertEqual(result['records'][1]['cells'][0]['text'], 'Next Fund')


class RendererTests(unittest.TestCase):
    def test_both_styles_match_independent_expectations(self):
        for style in ('text', 'table'):
            with self.subTest(style=style):
                output = render_record(mapped()['records'][0], input_id='example', group_id='dev', style=style)
                f = output['response']['records'][0]['fields']
                self.assertEqual(f['security_identifiers']['value'][0]['value']['value'], '001234567')
                self.assertEqual(f['proposal_source']['value'], 'SECURITY HOLDER')
                self.assertEqual(f['management_recommendation']['availability'], 'ABSENT_IN_CONTEXT')
                self.assertEqual(f['vote_components']['value'][0]['disclosed_management_alignment']['value'], 'AGAINST_MANAGEMENT')
                self.assertEqual(f['vote_components']['value'][0]['quantity']['value']['unit']['value'], 'shares')
                self.assertEqual(f['vote_components']['value'][0]['quantity']['value']['unit']['raw_text'], 'Shares voted')
                self.assertEqual(f['security_identifiers']['value'][0]['source_label']['raw_text'], 'CUSIP')
                self.assertEqual(f['reporting_scope']['value']['members']['availability'], 'NOT_APPLICABLE')
                self.assertEqual(f['raw_description']['value'], 'Approve A & B <C>')
                self.assertNotIn('AGAINST_MANAGEMENT', output['source_bytes'].decode())
                self.assertFalse(output['lineage']['training_admitted'])
                bundle = output['bundle']
                context = SourceContext({bundle['document_id']: output['source_bytes']},
                    {bundle['document_id']: [(0, len(output['source_bytes']))]},
                    views={(bundle['view_id'], b['block_id']): {'span': b['span'], 'text': b['prepared_text']} for b in bundle['blocks']})
                def walk(value):
                    if isinstance(value, dict):
                        for e in value.get('evidence', []):
                            self.assertTrue(context.citation_valid(e))
                        for child in value.values():
                            walk(child)
                    elif isinstance(value, list):
                        for child in value:
                            walk(child)
                walk(output['response'])

    def test_omission_and_exact_repeated_locations(self):
        output = render_record(mapped()['records'][0], input_id='example', group_id='dev',
                               omit=('proponent', 'scope', 'quantity:0'))
        f = output['response']['records'][0]['fields']
        self.assertEqual(f['proposal_source']['availability'], 'ABSENT_IN_CONTEXT')
        self.assertEqual(f['reporting_scope']['availability'], 'ABSENT_IN_CONTEXT')
        self.assertEqual(f['vote_components']['value'][0]['quantity']['availability'], 'ABSENT_IN_CONTEXT')
        self.assertNotIn('SECURITY HOLDER', output['source_bytes'].decode())
        with self.assertRaises(ValueError):
            render_record(mapped()['records'][0], input_id='x', group_id='dev', omit=('direction:0',))


class PreparationControlTests(unittest.TestCase):
    def test_probe_usability_keeps_citation_and_truncation_failures_separate(self):
        packet = render_record(mapped()['records'][0], input_id='example', group_id='dev')
        bundle = packet['bundle']
        context = SourceContext({bundle['document_id']: packet['source_bytes']},
            {bundle['document_id']: [(0, len(packet['source_bytes']))]},
            views={(bundle['view_id'], b['block_id']): {'span': b['span'], 'text': b['prepared_text']} for b in bundle['blocks']})
        reference = synthetic_reference(packet['response']['records'][0], input_id='example', guide_version=ACTIVE_GUIDE)
        schedule = [dict(input_id='example', input_path='unused', input_sha256='unused',
                         reference_path='unused', reference_sha256='unused', binding_sha256='unused', split='development')]
        with tempfile.TemporaryDirectory() as directory, patch('proxybench.execution.probe_report.preflight', return_value=(b'', reference, context)):
            p = Path(directory) / 'example'
            p.mkdir()
            (p / 'capture.json').write_text(json.dumps({'terminated': True}))
            (p / 'raw.txt').write_text(json.dumps(packet['response']))
            result = score_probe(Path('.'), schedule, directory)
            self.assertEqual(result['contract_usable'], 1)
            bad = deepcopy(packet['response'])
            bad['records'][0]['fields']['issuer_name']['evidence'][0]['quote'] = 'not in source'
            (p / 'raw.txt').write_text(json.dumps(bad))
            result = score_probe(Path('.'), schedule, directory)
            self.assertEqual(result['contract_usable'], 0)
            self.assertEqual(result['whole_record_matches'], 1)
            self.assertIn('EVIDENCE_ERROR', result['results'][0]['failure_categories'])
            (p / 'capture.json').write_text(json.dumps({'terminated': False}))
            result = score_probe(Path('.'), schedule, directory)
            self.assertEqual(result['whole_record_matches'], 0)
            self.assertIn('TRUNCATION', result['results'][0]['failure_categories'])

    def test_quarantine_covers_every_rendering_of_failed_rule(self):
        records = [{'logical_id': logical, 'rule_version': rule, 'status': 'PROVISIONAL_GENERATED',
                    'mechanical_validation': 'PASS'} for logical, rule in [('a', 'v1'), ('a', 'v1'), ('b', 'v1'), ('c', 'v2')]]
        event = {'logical_id': 'a', 'reviewer': 'user', 'active_seconds': 80,
                 'status': 'RULE_FAILED', 'corrections': ['Alignment mismatch']}
        result = record_review(records, event)
        self.assertEqual([r['status'] for r in result], ['QUARANTINED'] * 3 + ['PROVISIONAL_GENERATED'])
        self.assertTrue(all(r['mechanical_validation'] == 'PASS' for r in result))
        self.assertEqual(records[0]['status'], 'PROVISIONAL_GENERATED')
        with self.assertRaises(ValueError):
            record_review(result, event | {'status': 'ACCEPTED'})

    def test_masks_and_no_silent_truncation(self):
        class Tokenizer:
            eos_token_id = 99
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3] if len(messages) == 1 else [1, 2, 3, 44, 45, 99]
        result = sequence(Tokenizer(), 'source', 'answer', context_cap=6)
        self.assertEqual(result['labels'], [-100, -100, -100, 44, 45, 99])
        padded = pad_sequence(result, 8, 0)
        self.assertEqual(padded['labels'][-3:], [99, -100, -100])
        self.assertEqual(padded['attention_mask'][-2:], [0, 0])
        with self.assertRaises(ValueError):
            sequence(Tokenizer(), 'source', 'answer', context_cap=5)

    def test_memory_threshold_resets_and_timeouts(self):
        limits = dict(stop_host_bytes=100, phase_seconds=30, total_seconds=60)
        self.assertEqual(limit_reason(0, 90, 0, 0, limits), (1, None))
        self.assertEqual(limit_reason(1, 110, 0, 0, limits), (0, None))
        self.assertEqual(limit_reason(1, 90, 0, 0, limits), (2, 'HOST_MEMORY_LIMIT'))
        self.assertEqual(limit_reason(0, 110, 30, 30, limits)[1], 'PHASE_TIMEOUT')
        self.assertEqual(limit_reason(0, 110, 1, 60, limits)[1], 'AGGREGATE_TIMEOUT')

    def test_supervisor_terminates_worker_and_refuses_existing_run(self):
        limits = dict(start_host_bytes=100, stop_host_bytes=10, phase_seconds=.1,
                      total_seconds=20, device_margin_bytes=100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('proxybench.execution.resources.host_memory', return_value={'available_bytes': 1000}), \
                    patch('proxybench.execution.resources.device_memory', return_value={'free_bytes': 1000}):
                result = supervise([sys.executable, '-c', 'import time; time.sleep(20)'], root / 'run', limits, ledger=root / 'ledger')
                self.assertEqual(result, 'PHASE_TIMEOUT')
                saved = json.loads((root / 'run/result.json').read_text())
                self.assertLess(saved['elapsed_seconds'], 6)
                self.assertIsNotNone(saved['returncode'])
                with self.assertRaises(FileExistsError):
                    supervise([sys.executable, '-c', 'pass'], root / 'run', limits, ledger=root / 'ledger')


if __name__ == '__main__':
    unittest.main()
