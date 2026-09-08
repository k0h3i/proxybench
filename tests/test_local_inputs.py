"""Local XML parsing, training masks, and resource limits."""
import json
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import sys
from proxybench.sources.npx_xml import FORM_NS, VOTE_NS, XMLRejected, parse_xml
from proxybench.training.sequences import sequence, pad_sequence
from proxybench.execution.resources import limit_reason, supervise

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


class ResourceTests(unittest.TestCase):
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
