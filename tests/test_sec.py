from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.sources.sec import fetch, index_rows


class Response(BytesIO):
    def __init__(self, raw, status=200):
        super().__init__(raw)
        self.status = status
        self.headers = {'Content-Type': 'text/plain'}


class SecTests(unittest.TestCase):
    def test_cache_and_filing_limit(self):
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.sources.sec.time.sleep'):
            root = Path(tmp)
            with patch('proxybench.sources.sec.urlopen', return_value=Response(b'filing')) as call:
                first = fetch('https://www.sec.gov/a', root/'a.txt', root/'state.json', 'Declared client', max_filings=1)
                self.assertEqual(fetch('https://www.sec.gov/a', root/'a.txt', root/'state.json', 'Declared client', max_filings=1), first)
                self.assertEqual(call.call_count, 1)
            with self.assertRaises(RuntimeError):
                fetch('https://www.sec.gov/b', root/'b.txt', root/'state.json', 'Declared client', max_filings=1)

    def test_block_stops_all_future_requests_and_preserves_response(self):
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.sources.sec.time.sleep'):
            root = Path(tmp)
            with patch('proxybench.sources.sec.urlopen', return_value=Response(b'blocked', 403)) as call:
                for url in ['https://www.sec.gov/a', 'https://www.sec.gov/b']:
                    with self.assertRaises(RuntimeError):
                        fetch(url, root/'a.txt', root/'state.json', 'Private contact', kind='index')
                self.assertEqual(call.call_count, 1)
            state = (root/'state.json').read_text()
            self.assertNotIn('Private contact', state)
            self.assertTrue(json.loads(state)['blocked'])
            self.assertEqual(next(root.glob('*.partial-*')).read_bytes(), b'blocked')

    def test_oversize_preserves_partial_and_counts_excluded_filing(self):
        with tempfile.TemporaryDirectory() as tmp, patch('proxybench.sources.sec.time.sleep'):
            root = Path(tmp)
            with patch('proxybench.sources.sec.urlopen', return_value=Response(b'123456')):
                with self.assertRaises(RuntimeError):
                    fetch('https://www.sec.gov/a', root/'a.txt', root/'state.json', 'Declared client', max_document_bytes=3)
            state = json.loads((root/'state.json').read_text())
            self.assertEqual(state['filings'], 1)
            self.assertEqual(state['bytes'], 3)
            self.assertFalse((root/'a.txt').exists())

    def test_master_index_keeps_real_registrant_and_amendments(self):
        raw = b'CIK|Company Name|Form Type|Date Filed|Filename\n12|A Fund|N-PX|2007-08-01|edgar/data/12/0000999999-07-000001.txt\n12|A Fund|N-PX/A|2007-08-02|edgar/data/12/0000999999-07-000002.txt\n'
        rows = list(index_rows(raw))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['cik'], '12')
        self.assertEqual(rows[1]['form'], 'N-PX/A')
